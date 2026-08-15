import os
import json
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path("data")

def download_raw(dataset_name="neuralchemy/prompt-injection-Threat-Matrix", config="multiclass"):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(dataset_name, config)
    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(exist_ok=True)
    for split in ds.keys():
        out = raw_dir / f"{split}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for ex in ds[split]:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\\n")
    return raw_dir

def preprocess_text(text, lower=True):
    if lower:
        text = text.lower()
    # normalize whitespace
    text = \" \".join(text.split())
    return text

def prepare_processed(raw_dir: Path, out_dir: Path = Path("data/processed"), seed: int = 42):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Simple converter: read jsonl, apply preprocess, write to processed parquet or jsonl
    for f in raw_dir.glob("*.jsonl"):
        out_path = out_dir / f.name
        with f.open(\"r\", encoding=\"utf-8\") as rf, out_path.open(\"w\", encoding=\"utf-8\") as wf:
            for line in rf:
                obj = json.loads(line)
                obj[\"text\"] = preprocess_text(obj.get(\"text\",\"\"), lower=True)
                wf.write(json.dumps(obj, ensure_ascii=False) + \"\\n\")
    return out_dir

if __name__ == \"__main__\":
    raw = download_raw()
    prepare_processed(raw)

