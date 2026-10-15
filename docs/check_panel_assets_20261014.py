"""CPU-only actual-checkpoint loading and target preflight; no downloads."""
import gc
import json
import uuid
from pathlib import Path

import torch

from src.compression_core import sha256_file
from src.paper_architecture_panel import load_model, target_recipes, validate_config


def main():
    torch.set_num_threads(1)
    config = validate_config(json.loads(Path('configs/paper_architecture_panel_v3.json').read_text()))
    output = Path('results/paper-v2')/('architecture-asset-preflight-'+uuid.uuid4().hex[:8])
    output.mkdir(exist_ok=False)
    records = []
    for record in config['models']:
        model, info = load_model(record)
        if any(p.device.type != 'cpu' for p in model.parameters()):
            raise RuntimeError('preflight must remain on CPU')
        recipes = target_recipes(model)
        path = Path(record['model_path'])
        hashes = {str(p.relative_to(path)): sha256_file(p) for p in path.glob('*.safetensors')}
        hashes['config.json'] = sha256_file(path/'config.json')
        records.append({'model': record['model'], 'revision': record['model_revision'],
            'parameter_count': sum(p.numel() for p in model.parameters()), 'targets': recipes,
            'loading_info': info, 'asset_hashes': hashes, 'device': 'cpu'})
        (output/'preflight.json').write_text(json.dumps(records, indent=2))
        print(json.dumps({'model': record['model'], 'targets': len(recipes),
            'rank': recipes[0]['rank'], 'parameters': records[-1]['parameter_count']}), flush=True)
        del model
        gc.collect()
    (output/'status.json').write_text(json.dumps({'state': 'completed'}))
    print(output, flush=True)


if __name__ == '__main__':
    main()
