"""Explicit preparation of the two user-selected pinned architecture assets."""
import json
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    selection = json.loads(Path('paper-assets/architecture-selection-20261014.json').read_text())
    expected = {'Qwen/Qwen3.5-0.8B', 'LiquidAI/LFM2.5-1.2B-Instruct'}
    if {r['model'] for r in selection['records']} != expected:
        raise ValueError('unexpected model selection')
    output = Path('paper-assets')/('architecture-preparation-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
    output.mkdir(exist_ok=False)
    records = []
    try:
        for row in selection['records']:
            revision = row['revision']
            if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
                raise ValueError('revision must be an exact commit')
            path = snapshot_download(repo_id=row['model'], revision=revision,
                allow_patterns=['*.json', '*.safetensors', 'tokenizer*', 'vocab*', 'merges*',
                                'special_tokens*', 'chat_template*', 'README.md', 'LICENSE*'])
            records.append({'model': row['model'], 'revision': revision, 'path': path})
            (output/'assets.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(records[-1]), flush=True)
        (output/'status.json').write_text(json.dumps({'state': 'completed'}))
    except Exception:
        (output/'stderr.log').write_text(traceback.format_exc(), encoding='utf-8')
        (output/'status.json').write_text(json.dumps({'state': 'failed'}))
        raise
    print(output, flush=True)


if __name__ == '__main__':
    main()
