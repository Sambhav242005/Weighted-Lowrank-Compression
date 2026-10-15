"""Explicitly authorized preparation; never called automatically by the runner."""
import argparse
import json
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

GEMMA = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
GPT2 = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
WIKITEXT = "b08601e04326c79dfdd32d625aee71d232d685c3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", choices=["all", "gemma", "gpt2", "wikitext"], default="all")
    args = parser.parse_args()
    root = Path("paper-assets")
    root.mkdir(exist_ok=True)
    manifest = root / "asset-revisions.json"
    revisions = {"gemma": GEMMA, "gpt2": GPT2, "wikitext": WIKITEXT}
    if manifest.exists() and json.loads(manifest.read_text()) != revisions:
        raise RuntimeError("Existing asset manifest has different revisions")
    if not manifest.exists():
        manifest.write_text(json.dumps(revisions, indent=2) + "\n")
    for repo, rev in [("google/gemma-3-1b-it", GEMMA), ("openai-community/gpt2", GPT2)]:
        label = "gemma" if "gemma" in repo else "gpt2"
        if args.asset not in ["all", label]:
            continue
        print(f"Preparing {repo}@{rev}", flush=True)
        snapshot = snapshot_download(repo, revision=rev, allow_patterns=[
            "*.json", "*.safetensors", "*.model", "vocab.*", "merges.txt",
            "tokenizer*", "special_tokens_map.json",
        ])
        print(f"Prepared {snapshot}", flush=True)
    dest = root / f"wikitext-{WIKITEXT}"
    if args.asset in ["all", "wikitext"] and not dest.exists():
        print(f"Preparing WikiText-2@{WIKITEXT}", flush=True)
        data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", revision=WIKITEXT)
        data.save_to_disk(str(dest))
    print(f"Preparation complete: {args.asset}", flush=True)


if __name__ == "__main__":
    out = Path("results/paper-v2") / ("asset-preparation-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    (out / "command.txt").write_text("rtk " + " ".join(sys.argv) + "\n")
    try:
        main()
        (out / "status.json").write_text(json.dumps({"state": "completed"}))
    except Exception as exc:
        (out / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        (out / "status.json").write_text(json.dumps({"state": "failed", "error": str(exc)}))
        raise
