"""LoRA / QLoRA supervised fine-tuning of Qwen2.5-1.5B-Instruct on the CSE-Tutor SFT mix.

Loss is computed on the assistant answer only (system + user tokens are masked), using the same
Qwen chat template and system prompt as the evaluation harness.

Hardware:
  * NVIDIA GPU: 4-bit NF4 QLoRA with --qlora (bitsandbytes), otherwise bf16/fp16 LoRA
  * Intel Arc (XPU): bf16 LoRA (bitsandbytes 4-bit is not dependable on XPU)
  * CPU: fp32 LoRA (only for smoke tests)

Runs:
  FT-B (full mix):           python train/train_lora.py --run-name ft-b
  FT-A (no generator slice): python train/train_lora.py --run-name ft-a --exclude-source generator
  Timing / smoke test:       python train/train_lora.py --run-name smoke --max-steps 10 --limit 200

The adapter is saved to outputs/<run-name>/adapter (git-ignored) with a train_summary.json.
Evaluate with: python eval/run_eval.py --run-name FT-B --adapter outputs/ft-b/adapter ...
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFT_DIR = ROOT / "data" / "external" / "sft"


def load_records(path: Path, exclude: set[str], limit: int | None, seed: int) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    rows = [r for r in rows if r["source"] not in exclude]
    if limit:
        rows = random.Random(seed).sample(rows, min(limit, len(rows)))
    return rows


def tokenize(rows: list[dict], tok, max_len: int) -> list[dict]:
    """input_ids + labels with everything before the assistant answer masked to -100."""
    out = []
    for r in rows:
        msgs = r["messages"]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs, tokenize=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"]
        if f_ids[:len(p_ids)] != p_ids:  # template must extend the prompt exactly
            raise ValueError(f"chat template prefix mismatch for {r['id']}")
        f_ids = f_ids[:max_len]
        labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
        if all(l == -100 for l in labels):
            continue
        out.append({"input_ids": f_ids, "labels": labels})
    return out


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            labels[i, :L] = torch.tensor(b["labels"])
            mask[i, :L] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train-file", default=str(SFT_DIR / "sft_train_full.jsonl"))
    ap.add_argument("--val-file", default=str(SFT_DIR / "sft_val_full.jsonl"))
    ap.add_argument("--exclude-source", nargs="*", default=[], help="e.g. generator (FT-A ablation)")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8, help="effective batch = batch-size x grad-accum")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--qlora", action="store_true", help="4-bit NF4 base model (NVIDIA + bitsandbytes only)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--limit", type=int, help="subsample training rows (smoke tests)")
    ap.add_argument("--val-limit", type=int, default=300)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init-adapter", help="continue training this LoRA adapter (e.g. outputs/ft-b/adapter)")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device, dtype = "xpu", torch.bfloat16
    else:
        device, dtype = "cpu", torch.float32
    print(f"device={device} dtype={dtype} qlora={args.qlora}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    exclude = set(args.exclude_source)
    train_rows = load_records(Path(args.train_file), exclude, args.limit, args.seed)
    val_rows = load_records(Path(args.val_file), exclude, args.val_limit, args.seed)
    train_ds, val_ds = tokenize(train_rows, tok, args.max_len), tokenize(val_rows, tok, args.max_len)
    n_tokens = sum(len(x["input_ids"]) for x in train_ds)
    print(f"train {len(train_ds)} examples ({n_tokens:,} tokens), val {len(val_ds)}; "
          f"excluded sources: {sorted(exclude) or 'none'}", flush=True)

    load_kwargs = {"dtype": dtype}
    if args.qlora:
        if device != "cuda":
            raise SystemExit("--qlora needs an NVIDIA GPU with bitsandbytes")
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype)
    load_kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.config.use_cache = False
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if args.init_adapter:
        # Continued fine-tuning: start from an existing adapter (e.g. FT-B) instead of a fresh LoRA.
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    out_dir = ROOT / "outputs" / args.run_name
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * total_steps)),  # 3% warmup
        weight_decay=0.0,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        train_sampling_strategy="group_by_length",  # less padding per batch
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=Collator(tok.pad_token_id))
    print(f"steps/epoch={steps_per_epoch}, effective batch={args.batch_size * args.grad_accum}", flush=True)

    t0 = time.time()
    resume = any(out_dir.glob("checkpoint-*")) if out_dir.exists() else False
    result = trainer.train(resume_from_checkpoint=resume or None)
    elapsed = time.time() - t0
    metrics = trainer.evaluate()

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    summary = {
        "run_name": args.run_name, "model": args.model, "device": device, "dtype": str(dtype), "qlora": args.qlora,
        "excluded_sources": sorted(exclude), "init_adapter": args.init_adapter, "train_examples": len(train_ds), "train_tokens": n_tokens,
        "val_examples": len(val_ds), "epochs": args.epochs, "max_steps": args.max_steps,
        "global_steps": result.global_step, "train_loss": result.training_loss, "eval_loss": metrics.get("eval_loss"),
        "lr": args.lr, "effective_batch": args.batch_size * args.grad_accum, "lora": {
            "r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "seconds": round(elapsed, 1), "seconds_per_step": round(elapsed / max(result.global_step, 1), 2),
        "log_history": trainer.state.log_history,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "log_history"}, indent=2), flush=True)


if __name__ == "__main__":
    sys.exit(main())
