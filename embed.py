import json, hashlib
from pathlib import Path


SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]
SPECIALS_MAPPING = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}

# returns itos given input training text
def build_vocab(text: str) -> list[str]:
    return SPECIALS + sorted(set(text))

def vocab_hash(itos: list[str]) -> str:
    encoded_string = json.dumps({"version": 1, "itos" : itos}, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded_string).hexdigest()[:16]

def save_vocab(itos: list, path: Path):
    data = {"version": 1, "itos" : itos, "specials": SPECIALS_MAPPING, "hash": vocab_hash(itos)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# save_vocab(build_vocab("hi hi hi  hi a aaaaabcdefghiji@%#$639t73983wg-[p[p]]"), Path("test_vocab.json"))