import json, hashlib
from pathlib import Path

import torch
import torch.nn as nn


class Vocab():
    def __init__(self):
        self.specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.specials_mapping = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}

    # returns itos given input training text
    def build_vocab(self, text: str) -> list[str]:
        return self.specials + sorted(set(text))

    def vocab_hash(self, itos: list[str]) -> str:
        encoded_string = json.dumps({"version": 1, "itos" : itos}, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded_string).hexdigest()[:16]

    def save_vocab(self, itos: list, path: Path):
        data = {"version": 1, "itos" : itos, "specials": self.specials_mapping, "hash": self.vocab_hash(itos)}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data"
    DATA_DIR.mkdir(exist_ok=True)

    save_vocab(build_vocab("hi hi hi  hi a aaaaabcdefghiji@%#$639t73983wg-[p[p]]"), DATA_DIR / "test_vocab.json")


class Embedder(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return None