"""CSE-Tutor chat server: a chat web interface for the fine-tuned model.

  python demo/server.py                 # http://127.0.0.1:8000
  python demo/server.py --port 8080

The base model is loaded once; the LoRA adapters found in outputs/ (FT-C, FT-B, FT-A) are attached as
named adapters, so the interface can switch between them, or turn them off for the base model,
without reloading anything. Answers stream token by token (server-sent events).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from peft import PeftModel
from pydantic import BaseModel
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList,
                          TextIteratorStreamer)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.generators.common import SYSTEM_PROMPT  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
ADAPTERS = {  # id -> (label, description), in menu order; only those present on disk are offered
    "ft-e": ("CSE-Tutor FT-E", "FT-C + full explanation stage (recommended)"),
    "ft-d": ("CSE-Tutor FT-D", "FT-C + short explanation stage"),
    "ft-c": ("CSE-Tutor FT-C", "FT-B + Deep Learning stage"),
    "ft-b": ("CSE-Tutor FT-B", "full CSE mix, 1 epoch"),
    "ft-a": ("CSE-Tutor FT-A", "ablation: no generator data"),
}
TUTOR_PROMPT = SYSTEM_PROMPT  # exactly the prompt used in training and evaluation


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


class Stop(StoppingCriteria):
    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.event.is_set()


class Engine:
    def __init__(self, model: str):
        self.device = pick_device()
        self.tok = AutoTokenizer.from_pretrained(model)
        base = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.float32 if self.device == "cpu" else torch.float16, device_map={"": self.device})
        self.models = [{"id": "base", "label": "Qwen2.5-1.5B base", "description": "no fine-tuning"}]
        self.model = None
        for aid, (label, desc) in ADAPTERS.items():
            path = ROOT / "outputs" / aid / "adapter"
            if not path.exists():
                continue
            if self.model is None:
                self.model = PeftModel.from_pretrained(base, path, adapter_name=aid)
            else:
                self.model.load_adapter(path, adapter_name=aid)
            self.models.insert(len(self.models) - 1, {"id": aid, "label": label, "description": desc})
        self.model = (self.model or base).eval()
        self.lock = threading.Lock()  # one generation at a time on a single GPU
        self.stop = threading.Event()
        print(f"device={self.device}; models: {[m['id'] for m in self.models]}", flush=True)

    def stream(self, messages: list[dict], model_id: str, max_new_tokens: int, temperature: float):
        prompt = self.tok.apply_chat_template([{"role": "system", "content": TUTOR_PROMPT}, *messages],
                                              add_generation_prompt=True, tokenize=False)
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        sampling = {"do_sample": True, "temperature": temperature, "top_p": 0.9} if temperature > 0 else {"do_sample": False}
        kwargs = dict(**enc, streamer=streamer, max_new_tokens=max_new_tokens,
                      stopping_criteria=StoppingCriteriaList([Stop(self.stop)]), **sampling)

        def run():
            with torch.no_grad():
                if model_id == "base" or not isinstance(self.model, PeftModel):
                    if isinstance(self.model, PeftModel):
                        with self.model.disable_adapter():
                            self.model.generate(**kwargs)
                    else:
                        self.model.generate(**kwargs)
                else:
                    self.model.set_adapter(model_id)
                    self.model.generate(**kwargs)

        with self.lock:
            self.stop.clear()
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            yield from streamer
            worker.join()


class ChatRequest(BaseModel):
    messages: list[dict]
    model: str = "ft-c"
    max_new_tokens: int = 1024
    temperature: float = 0.0


def build_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="CSE-Tutor")
    app.mount("/vendor", StaticFiles(directory=STATIC / "vendor"), name="vendor")  # Markdown libraries, works offline
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/models")
    def models():
        return {"models": engine.models, "device": engine.device}

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        ids = {m["id"] for m in engine.models}
        model_id = req.model if req.model in ids else engine.models[0]["id"]
        msgs = [{"role": m["role"], "content": m["content"]} for m in req.messages
                if m.get("role") in ("user", "assistant")][-12:]  # keep the context short

        def events():
            t0, n = time.time(), 0
            for piece in engine.stream(msgs, model_id, min(req.max_new_tokens, 2048), req.temperature):
                n += 1
                yield f"data: {json.dumps({'delta': piece})}\n\n"
            yield f"data: {json.dumps({'done': True, 'seconds': round(time.time() - t0, 1), 'chunks': n})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/api/stop")
    def stop():
        engine.stop.set()
        return {"stopped": True}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(build_app(Engine(args.model)), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
