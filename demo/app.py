"""CSE-Tutor demo: chat with the fine-tuned model and try it on freshly generated, verified problems.

  python demo/app.py                       # newest adapter (FT-E, else FT-D, FT-C, FT-B), http://127.0.0.1:7860
  python demo/app.py --adapter outputs/ft-a/adapter
  python demo/app.py --share               # temporary public Gradio link

The adapter is kept separate from the base weights, so a checkbox can switch it off and show the
base model's answer to the same question.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from threading import Thread

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.generators import cn_gen, dbms_gen, dl_gen, os_gen  # noqa: E402
from data.generators.common import SYSTEM_PROMPT  # noqa: E402
from data.generators.grading import grade  # noqa: E402
from data.plain_math import plain_math  # noqa: E402

GEN_INSTR = "Solve step by step. End with a line starting with 'Final answer:'."
SUBJECTS = {"Operating Systems": os_gen, "Computer Networks": cn_gen, "DBMS": dbms_gen, "Deep Learning": dl_gen}
EXAMPLES = [
    "Explain the difference between deadlock prevention and deadlock avoidance, with an example.",
    "What is the time complexity of building a binary heap from n elements, and why is it not O(n log n)?",
    "Write a SQL query to find the second highest salary from an Employee(id, name, salary) table.",
    "Why does TCP use a three-way handshake instead of two?",
    "Write a Python function that checks whether a string of brackets is balanced.",
    "Explain why residual connections help train very deep networks.",
]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


class Tutor:
    def __init__(self, model: str, adapter: str):
        self.device = pick_device()
        self.tok = AutoTokenizer.from_pretrained(model)
        base = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.float32 if self.device == "cpu" else torch.float16, device_map={"": self.device})
        self.model = PeftModel.from_pretrained(base, adapter).eval()

    def stream(self, messages: list[dict], use_adapter: bool, max_new_tokens: int = 1024):
        prompt = self.tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, *messages],
                                              add_generation_prompt=True, tokenize=False)
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)

        def run():
            with torch.no_grad():
                if use_adapter:
                    self.model.generate(**enc, streamer=streamer, max_new_tokens=max_new_tokens, do_sample=False)
                else:
                    with self.model.disable_adapter():
                        self.model.generate(**enc, streamer=streamer, max_new_tokens=max_new_tokens, do_sample=False)

        Thread(target=run, daemon=True).start()
        text = ""
        for piece in streamer:
            text += piece
            yield text


def build(tutor: Tutor, adapter_name: str) -> gr.Blocks:
    with gr.Blocks(title="CSE-Tutor 1.5B") as app:
        gr.Markdown(f"# CSE-Tutor 1.5B\nQwen2.5-1.5B-Instruct with the **{adapter_name}** LoRA adapter, "
                    f"running on `{tutor.device}`. Untick the box to compare with the base model.")
        use_adapter = gr.Checkbox(value=True, label=f"Use the fine-tuned adapter ({adapter_name})")

        with gr.Tab("Ask the tutor"):
            chat = gr.Chatbot(height=480, label="CSE-Tutor")
            msg = gr.Textbox(placeholder="Ask about DSA, programming, DBMS, OS or networks…", label="Question")
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
            gr.Examples(EXAMPLES, inputs=msg)

            def respond(question, history, adapter_on):
                history = (history or []) + [{"role": "user", "content": question},
                                             {"role": "assistant", "content": ""}]
                for text in tutor.stream(history[:-1], adapter_on):
                    history[-1]["content"] = plain_math(text)  # math as plain text, no LaTeX
                    yield "", history

            send.click(respond, [msg, chat, use_adapter], [msg, chat])
            msg.submit(respond, [msg, chat, use_adapter], [msg, chat])
            clear.click(lambda: [], None, chat)

        with gr.Tab("Verified practice problems"):
            gr.Markdown("A new problem is generated with a **computed** answer, so the model's final answer "
                        "is graded automatically. Problems use the held-out test parameters.")
            state = gr.State(None)
            with gr.Row():
                subject = gr.Dropdown(list(SUBJECTS), value="DBMS", label="Subject")
                new = gr.Button("New problem")
            question = gr.Markdown()
            solve = gr.Button("Solve with the model", variant="primary")
            answer = gr.Markdown()
            verdict = gr.Markdown()

            def new_problem(subj):
                rec = SUBJECTS[subj].generate(1, seed=random.randrange(10**9), split="test")[0]
                q = rec["messages"][1]["content"]
                return rec, q, "", ""

            def solve_problem(rec, adapter_on):
                if rec is None:
                    yield "Generate a problem first.", ""
                    return
                text = ""
                messages = [{"role": "user", "content": rec["messages"][1]["content"] + "\n\n" + GEN_INSTR}]
                for text in tutor.stream(messages, adapter_on):
                    yield plain_math(text), ""
                ok = grade(rec["final"], text)
                ref = rec["messages"][2]["content"].split("**Final answer:**")[-1].strip()
                yield plain_math(text), f"### {'✅ Correct' if ok else '❌ Incorrect'}\n**Verified answer:** {ref}"

            new.click(new_problem, subject, [state, question, answer, verdict])
            solve.click(solve_problem, [state, use_adapter], [answer, verdict])
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    default = ROOT / "outputs" / next(r for r in ("ft-e", "ft-d", "ft-c", "ft-b") if (ROOT / "outputs" / r / "adapter").exists()) / "adapter"
    ap.add_argument("--adapter", default=str(default))
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    tutor = Tutor(args.model, args.adapter)
    name = Path(args.adapter).parent.name.upper()
    build(tutor, name).queue().launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
