"""Local CPU BGE embedding service; isolated from the generation model GPU."""
import argparse
import os
from pathlib import Path
import time
import torch
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--port", type=int, default=20200)
args = parser.parse_args()
torch.set_num_threads(8)
model = AutoModel.from_pretrained(args.model, local_files_only=True).eval().to("cpu")
tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
app = FastAPI()

class Input(BaseModel):
    texts: list[str]
    query: bool = False

@app.get("/health")
def health():
    return {"model": "BAAI/bge-small-en-v1.5", "device": str(next(model.parameters()).device), "dimension": 384}

@app.post("/embed")
def embed(body: Input):
    texts = [("Represent this sentence for searching relevant passages: " + text) if body.query else text for text in body.texts]
    vectors, tokens = [], 0
    started = time.monotonic()
    for start in range(0, len(texts), 32):
        encoded = tokenizer(texts[start:start + 32], padding=True, truncation=True, max_length=512, return_tensors="pt")
        tokens += int(encoded["attention_mask"].sum())
        with torch.inference_mode():
            output = model(**encoded).last_hidden_state[:, 0]
            output = torch.nn.functional.normalize(output, p=2, dim=1)
            vectors.extend(output.tolist())
    return {"vectors": vectors, "input_tokens": tokens, "dimension": 384,
            "duration_seconds": time.monotonic() - started, "device": "cpu"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1)
