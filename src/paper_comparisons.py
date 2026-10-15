"""Offline, additive regularization and matched component comparisons.

Window CKA is an explicitly secondary diagnostic: centered within each window,
then token-weighted across windows; it is not a whole-corpus CKA estimator.
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from .compression_core import PackedLowRankLinear, sha256_file
from .paper_math import factors, weighted_factors, asvd_factors
from .paper_protocol import calibration_starts, evaluate, token_hash, windows


def load_assets(config):
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], local_files_only=True)
    data = load_from_disk(config['dataset_path'])
    if set(data) != {'train', 'validation', 'test'}:
        raise ValueError('native train/validation/test splits required')
    tokens = {s: tokenizer('\n\n'.join(data[s]['text']), add_special_tokens=False,
                           return_tensors='pt')['input_ids'][0] for s in data}
    if len({token_hash(t) for t in tokens.values()}) != 3:
        raise ValueError('split hash collision')
    teacher = AutoModelForCausalLM.from_pretrained(config['model_path'],
        local_files_only=True, torch_dtype=torch.float16).to(config['device']).eval()
    teacher.requires_grad_(False)
    return teacher, tokens


def target_modules(model):
    targets = [(n, m) for n, m in model.named_modules()
               if n.endswith('self_attn.o_proj') or n.endswith('attn.c_proj')]
    if not targets:
        raise ValueError('attention output targets missing')
    return targets


def canonical_weight(module):
    # HF GPT-2 Conv1D stores [in,out], Linear stores [out,in].
    return module.weight.detach().T if module.__class__.__name__ == 'Conv1D' else module.weight.detach()


class CalibrationInputs(dict):
    """Input mapping with bias-inclusive measured teacher outputs from same passes."""
    def __init__(self, inputs, outputs):
        super().__init__(inputs)
        self.outputs = outputs


@torch.inference_mode()
def capture_inputs(teacher, train_tokens, seed, progress=None):
    captures = {n: [] for n, _ in target_modules(teacher)}
    output_captures = {n: [] for n, _ in target_modules(teacher)}
    handles = []
    for name, module in target_modules(teacher):
        def hook(_module, args, name=name):
            captures[name].append(args[0].detach().reshape(-1, args[0].shape[-1]).half().cpu())
        handles.append(module.register_forward_pre_hook(hook))
        def output_hook(_module, _args, result, name=name):
            output_captures[name].append(result.detach().reshape(-1, result.shape[-1]).half().cpu())
        handles.append(module.register_forward_hook(output_hook))
    try:
        for index, start in enumerate(calibration_starts(len(train_tokens), seed)):
            if progress:
                progress(f'capture seed={seed} chunk={index}')
            teacher(input_ids=train_tokens[start:start+512][None].to(next(teacher.parameters()).device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return CalibrationInputs({n: torch.cat(values).T.contiguous() for n, values in captures.items()},
                             {n: torch.cat(values).T.contiguous() for n, values in output_captures.items()})


def fit_checkpoint(teacher, inputs, config, specification, directory, progress=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint = directory / 'factors.pt'
    started = time.monotonic()
    cuda = next(teacher.parameters()).is_cuda
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    state, recipes = {}, []
    for name, module in target_modules(teacher):
        if progress:
            progress(f'fit {specification} target={name}')
        w = canonical_weight(module).float()
        rank = min(w.shape) if specification['method'] == 'fullrank_ridge' else config['rank']
        if specification['method'] == 'svd':
            left, right = factors(w, rank)
        elif specification['method'] == 'asvd':
            left, right = asvd_factors(w, inputs[name].to(w.device), rank, alpha=0.5)
        else:
            measured = specification['method'] in {'legacy_ridge', 'fullrank_ridge'}
            if measured and not hasattr(inputs, 'outputs'):
                raise ValueError('legacy solver requires captured measured teacher outputs')
            left, right = weighted_factors(w, inputs[name].to(w.device), rank,
                beta=specification.get('beta', 0.0), ridge_fraction=specification.get('ridge_fraction', 0.0),
                rcond=config['pseudoinverse_rcond'],
                outputs=inputs.outputs[name].to(w.device) if measured else None,
                bias=module.bias if measured else None)
        convention = config.get('factor_convention', 'unbalanced')
        if convention == 'balanced':
            # Equalize paired factor-column norms without changing U@V.
            scale = (right.norm(dim=1)/left.norm(dim=0).clamp_min(1e-30)).clamp_min(1e-30).sqrt()
            left, right = left*scale, right/scale[:, None]
        elif convention != 'unbalanced':
            raise ValueError('unknown factor convention')
        left, right = left.half().cpu(), right.half().cpu()
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise FloatingPointError('nonfinite stored factors')
        state[name] = {'u': left, 'v': right,
                       'bias': module.bias.detach().cpu() if module.bias is not None else None}
        recipes.append({'name': name, 'shape': list(w.shape), 'rank': rank,
                        'factor_parameters': left.numel()+right.numel(),
                        'factor_tensor_bytes': 2*(left.numel()+right.numel()),
                        'bias_parameters': module.bias.numel() if module.bias is not None else 0})
        del w, left, right
    if cuda:
        torch.cuda.synchronize()
    if progress:
        progress('factor fitting complete; checkpoint serialization begins')
    torch.save(state, checkpoint)
    if progress:
        progress('factor checkpoint serialization complete')
    metadata = {'specification': specification, 'precision': 'float16', 'targets': recipes,
                'factor_convention': config.get('factor_convention', 'unbalanced'),
                'output_source': 'measured_teacher_bias_subtracted' if specification['method'] in {'legacy_ridge', 'fullrank_ridge'} else 'teacher_exact_WX',
                'budget_role': 'unmatched full-rank shrinkage control' if specification['method'] == 'fullrank_ridge' else 'matched fixed-rank comparison',
                'factor_parameters': sum(r['factor_parameters'] for r in recipes),
                'serialized_bytes': checkpoint.stat().st_size, 'sha256': sha256_file(checkpoint),
                'fit_seconds': time.monotonic()-started,
                'peak_fitting_allocated_bytes': torch.cuda.max_memory_allocated() if cuda else None,
                'peak_fitting_reserved_bytes': torch.cuda.max_memory_reserved() if cuda else None}
    from .paper_benchmark import write
    write(directory / 'metadata.json', metadata)
    return checkpoint, metadata


def student_from_checkpoint(teacher, checkpoint, names=None):
    student = copy.deepcopy(teacher)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    selected = set(state) if names is None else set(names)
    if not selected <= set(state):
        raise ValueError('requested replacement missing in checkpoint')
    for name in selected:
        value = state[name]
        packed = PackedLowRankLinear.from_factors(value['u'], value['v'], value['bias'], bits=16)
        parent, child = name.rsplit('.', 1)
        setattr(student.get_submodule(parent), child, packed.to(next(teacher.parameters()).device))
    return student.eval()


def block_modules(model):
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return list(model.transformer.h)
    for name, module in model.named_modules():
        if name.endswith('layers') and isinstance(module, torch.nn.ModuleList):
            return list(module)
    raise ValueError('decoder block structure unavailable')


class Geometry:
    """Accumulate sums across all scored positions; no final-prompt overwrite."""
    def __init__(self):
        self.error = self.reference = self.candidate = self.cosine = self.cka = 0.0
        self.tokens = 0
        self.valid_cka_tokens = 0

    def add(self, reference, candidate):
        x, y = reference.float().reshape(-1, reference.shape[-1]), candidate.float().reshape(-1, candidate.shape[-1])
        if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise FloatingPointError('invalid diagnostic activations')
        if torch.equal(x, y):
            # Exact observed tensor equality permits skipping duplicate norm
            # reductions and centered-Gram GEMMs. Preserve cosine's epsilon
            # behavior for zero/tiny norms rather than assuming distance zero.
            squared_norm = x.double().square().sum().item()
            self.reference += squared_norm
            self.candidate += squared_norm
            self.cosine += (1-F.cosine_similarity(x, y, dim=-1)).double().sum().item()
            self.tokens += len(x)
            if len(x) > 1 and (x-x.mean(0)).abs().amax().item() > 0:
                self.cka += len(x)
                self.valid_cka_tokens += len(x)
            return
        self.error += (x-y).double().square().sum().item()
        self.reference += x.double().square().sum().item()
        self.candidate += y.double().square().sum().item()
        self.cosine += (1-F.cosine_similarity(x, y, dim=-1)).double().sum().item()
        self.tokens += len(x)
        if len(x) > 1:
            x, y = x-x.mean(0), y-y.mean(0)
            # CKA is invariant to independent positive rescaling. Normalize
            # centered features before float32 GEMM, then reduce in float64:
            # both tiny variance and large finite FP16 activations remain safe.
            scale_x, scale_y = x.abs().amax(), y.abs().amax()
            if scale_x.item() > 0 and scale_y.item() > 0:
                x, y = x/scale_x, y/scale_y
                gx, gy = (x @ x.T).double(), (y @ y.T).double()
                norm_x, norm_y = gx.square().sum().sqrt(), gy.square().sum().sqrt()
                if norm_x.item() > 0 and norm_y.item() > 0:
                    similarity = (gx*gy).sum()/(norm_x*norm_y)
                    if not torch.isfinite(similarity):
                        raise FloatingPointError('nonfinite linear CKA')
                    self.cka += similarity.clamp(0, 1).item()*len(x)
                    self.valid_cka_tokens += len(x)

    def result(self):
        return {'normalized_squared_error': self.error/max(self.reference, 1e-30),
                'cosine_distance': self.cosine/max(self.tokens, 1),
                'norm_ratio': math.sqrt(self.candidate/max(self.reference, 1e-30)),
                'window_linear_cka': self.cka/self.valid_cka_tokens if self.valid_cka_tokens else None,
                'valid_cka_tokens': self.valid_cka_tokens, 'tokens': self.tokens}


@torch.inference_mode()
def evaluate_diagnostics(student, teacher, tokens, config, progress=None):
    residual = [Geometry() for _ in block_modules(teacher)]
    local = {n: Geometry() for n, _ in target_modules(teacher)}
    refs, candidates, local_refs, local_candidates = {}, {}, {}, {}
    handles = []
    # During teacher passes, run candidate modules on identical teacher inputs.
    for name, module in target_modules(teacher):
        candidate_module = student.get_submodule(name)
        def local_hook(_m, args, output, name=name, candidate_module=candidate_module):
            local_refs[name] = output.detach()
            local_candidates[name] = output.detach() if student is teacher else candidate_module(args[0]).detach()
        handles.append(module.register_forward_hook(local_hook))
    for model, storage in [(teacher, refs), (student, candidates)]:
        if model is teacher and student is teacher:
            # Dense baseline uses one forward and copies reference diagnostics.
            continue
        for index, block in enumerate(block_modules(model)):
            def hook(_m, _args, output, index=index, storage=storage):
                storage[index] = (output[0] if isinstance(output, tuple) else output).detach()
            handles.append(block.register_forward_hook(hook))
    device = next(teacher.parameters()).device
    total = count = kl_sum = 0
    nll_blocks, block_sum, block_count = [], 0.0, 0
    try:
        for index, (start, end, first) in enumerate(windows(len(tokens), config['context'], config['stride'])):
            if progress:
                progress(f'evaluation window={index} targets={count}')
            ids = tokens[start:end][None].to(device)
            reference = teacher(input_ids=ids, use_cache=False).logits[0].float()
            logits = reference if student is teacher else student(input_ids=ids, use_cache=False).logits[0].float()
            offset = first-start-1
            losses = F.cross_entropy(logits[offset:-1], ids[0, offset+1:], reduction='none')
            ref_log = reference[offset:-1].log_softmax(-1)
            kl = (ref_log.exp()*(ref_log-logits[offset:-1].log_softmax(-1))).sum(-1)
            if not torch.isfinite(losses).all() or not torch.isfinite(kl).all():
                raise FloatingPointError('NaN/Inf evaluation')
            total += losses.double().sum().item()
            kl_sum += kl.double().sum().item()
            count += len(losses)
            for loss in losses.double().cpu().tolist():
                block_sum += loss
                block_count += 1
                if block_count == 4096:
                    nll_blocks.append({'nll_sum': block_sum, 'targets': block_count})
                    block_sum, block_count = 0.0, 0
            for i, geometry in enumerate(residual):
                if student is teacher:
                    continue
                geometry.add(refs[i][:, offset:-1], candidates[i][:, offset:-1])
            for name, geometry in local.items():
                geometry.add(local_refs[name][:, offset:-1], local_candidates[name][:, offset:-1])
            refs.clear(); candidates.clear(); local_refs.clear(); local_candidates.clear()
    finally:
        for handle in handles:
            handle.remove()
    if block_count:
        nll_blocks.append({'nll_sum': block_sum, 'targets': block_count})
    if count != len(tokens)-1:
        raise ValueError('evaluation target count mismatch')
    drift = [g.result() for g in residual] if student is not teacher else [
        {'normalized_squared_error': 0.0, 'cosine_distance': 0.0, 'norm_ratio': 1.0,
         'window_linear_cka': 1.0, 'tokens': count} for _ in residual]
    return {'nll_sum': total, 'targets': count, 'nll': total/count, 'ppl': math.exp(total/count),
            'logit_kl': kl_sum/count, 'nll_blocks': nll_blocks,
            'local_output': {n: g.result() for n, g in local.items()}, 'residual_by_block': drift,
            'primary_drift': sum(g['normalized_squared_error'] for g in drift)/len(drift),
            'diagnostic_scope': 'primary geometry uses all scored shifted positions; CKA is token-weighted over nonzero centered covariance windows, with valid_cka_tokens reported'}


def regularization_specs(config):
    specs = [{'method': 'svd'}, {'method': 'fullrank_ridge', 'beta': config['beta'],
                               'ridge_fraction': config['ridge_fraction']}]
    specs += [{'method': 'anchored', 'beta': beta, 'ridge_fraction': 0.0}
              for beta in [0.0, 0.03, 0.1, 0.3, 1.0]]
    specs += [{'method': 'legacy_ridge', 'beta': 0.1, 'ridge_fraction': ridge}
              for ridge in [0.0, 0.001, 0.01]]
    recorded = {'method': 'legacy_ridge', 'beta': config['beta'], 'ridge_fraction': config['ridge_fraction']}
    if recorded not in specs:
        specs.append(recorded)
    specs.append({'method': 'asvd'})  # validation-only comparator selection before C/D
    return specs


def select_result(results, methods, tie_tolerance=1e-6):
    eligible = [r for r in results if r['specification']['method'] in methods]
    best = min(r['metrics']['nll'] for r in eligible)
    tied = [r for r in eligible if r['metrics']['nll'] <= best+tie_tolerance]
    return min(tied, key=lambda r: (r['specification'].get('ridge_fraction', 0.0),
                                  r['specification'].get('beta', 0.0), r['id']))


def validate_selection_provenance(selection, config, manifests, teacher_hash, targets, dataset_hash=None):
    """Fail before test use if any frozen selection evidence changed."""
    if selection.get('state') != 'frozen' or selection.get('selection_split') != 'validation':
        raise ValueError('selection must be frozen from validation')
    for key in ['model_revision', 'dataset_revision', 'rank', 'precision']:
        if selection.get(key) != config[key]:
            raise ValueError(f'frozen selection mismatch: {key}')
    if selection.get('factor_convention', 'unbalanced') != config.get('factor_convention', 'unbalanced'):
        raise ValueError('factor convention changed after validation selection')
    if selection.get('teacher_hash') != teacher_hash:
        raise ValueError('teacher changed after validation selection')
    if dataset_hash is not None and selection.get('dataset_content_hash') != dataset_hash:
        raise ValueError('raw pinned dataset contents changed after validation selection')
    expected_splits = {name: manifests[name]['hash'] for name in ['train', 'validation', 'test']}
    if selection.get('split_hashes') != expected_splits:
        raise ValueError('split contents changed after validation selection')
    if selection.get('calibration_token_hashes') != manifests['calibration_hashes']:
        raise ValueError('calibration token contents changed after validation selection')
    if selection.get('targets') != targets:
        raise ValueError('target recipes changed after validation selection')
    if selection.get('selection_seed') != 11 or config['seeds'] != [11, 23, 37]:
        raise ValueError('selection/calibration seed protocol changed')


def run_comparisons(config, stage):
    from .paper_benchmark import create_output, environment, write, tensor_digest, command_line
    from .paper_followups import dataset_content_hash
    if stage not in {'regularization', 'baselines'}:
        raise ValueError('unsupported comparison stage')
    output = create_output(config, stage)
    started = time.monotonic()
    def progress(message):
        cap = config.get('runtime_remaining_seconds', {}).get(stage, config['stage_gpu_hours'][stage]*3600)
        if time.monotonic()-started > cap:
            raise TimeoutError(f'{stage} cap exceeded; preserve incomplete comparison')
        print(message, flush=True)
        with (output/'stdout.log').open('a', encoding='utf-8') as handle:
            handle.write(f'{time.monotonic()-started:.3f} {message}\n')
    (output/'stderr.log').touch()
    results = []
    try:
        env = environment(config)
        write(output/'preflight.json', env)
        if not env['ready']:
            raise RuntimeError(f"preflight incomplete: {env['missing']}")
        if config['precision'] != 'float16':
            raise ValueError('matched FP16 factor precision required')
        selection = None
        if stage == 'baselines':
            path = Path(config['selection_path'])
            selection = json.loads(path.read_text(encoding='utf-8'))
            if selection['state'] != 'frozen' or any(selection[key] != config[key] for key in
                    ['model_revision', 'dataset_revision', 'rank', 'precision']):
                raise ValueError('frozen same-model validation selection required')
            if selection.get('factor_convention', 'unbalanced') != config.get('factor_convention', 'unbalanced'):
                raise ValueError('factor convention changed after validation selection')
            write(output/'frozen-selection.json', selection)
        dataset_hash = dataset_content_hash(config['dataset_path'])
        if selection and selection.get('dataset_content_hash') != dataset_hash:
            raise ValueError('raw pinned dataset contents changed after validation selection')
        teacher, tokens = load_assets(config)
        manifests = {s: {'tokens': len(t), 'hash': token_hash(t)} for s, t in tokens.items()}
        manifests['calibration_positions'] = {str(s): calibration_starts(len(tokens['train']), s) for s in config['seeds']}
        manifests['calibration_hashes'] = {str(seed): token_hash(torch.cat([
            tokens['train'][position:position+512] for position in manifests['calibration_positions'][str(seed)]
        ])) for seed in config['seeds']}
        if selection and selection['validation_hash'] != manifests['validation']['hash']:
            raise ValueError('validation changed after selection')
        write(output/'token-manifests.json', manifests)
        original = tensor_digest(teacher)
        target_recipes = [{'name': n, 'shape': list(canonical_weight(m).shape), 'rank': config['rank'],
                           'factor_parameters': config['rank']*sum(canonical_weight(m).shape)}
                          for n, m in target_modules(teacher)]
        if selection:
            validate_selection_provenance(selection, config, manifests, original, target_recipes, dataset_hash)
        write(output/'contract.json', {'run_id': output.name, 'command': command_line(),
              'environment': env, 'configuration': config, 'splits': manifests, 'dataset_content_hash': dataset_hash,
              'targets': [{'name': n, 'shape': list(canonical_weight(m).shape), 'rank': config['rank'],
                           'factor_parameters': config['rank']*sum(canonical_weight(m).shape)}
                          for n, m in target_modules(teacher)], 'group_size': 'all attention outputs',
              'baseline': 'SVD and fixed-rank ASVD / whitening-only components, same FP16/rank',
              'teacher_hash': original, 'selection_hash': sha256_file(config['selection_path']) if selection else None,
              'expected_artifacts': ['config.json', 'contract.json', 'stdout.log', 'stderr.log', 'metrics.json', 'report.md', 'factors.pt hashes']})
        write(output/'status.json', {'state': 'running'})
        split = 'validation' if stage == 'regularization' else 'test'
        baseline = evaluate(teacher, tokens[split], context=config['context'], stride=config['stride'], progress=progress)
        write(output/'teacher.json', baseline)
        for seed in ([11] if stage == 'regularization' else config['seeds']):
            inputs = capture_inputs(teacher, tokens['train'], seed, progress)
            specs = regularization_specs(config) if stage == 'regularization' else [
                {'method': 'svd'}, {'method': 'asvd'},
                {'method': 'whitening', 'beta': 0.0, 'ridge_fraction': 0.0},
                selection['selected_anchor']['specification'],
                {'method': 'legacy_ridge', 'beta': config['beta'], 'ridge_fraction': config['ridge_fraction']}]
            if stage == 'baselines' and selection['selected_solver']['specification'] not in specs:
                specs.append(selection['selected_solver']['specification'])
            for index, spec in enumerate(specs):
                if seed != 11 and spec['method'] == 'svd':
                    continue
                identifier = f"seed-{seed}-{index:02d}-{spec['method']}"
                checkpoint, metadata = fit_checkpoint(teacher, inputs, config, spec, output/identifier, progress)
                student = student_from_checkpoint(teacher, checkpoint)
                metric = evaluate_diagnostics(student, teacher, tokens[split], config, progress) if stage == 'baselines' else evaluate(
                    student, tokens[split], context=config['context'], stride=config['stride'], teacher=teacher, progress=progress)
                metric['teacher_relative_ppl_change'] = metric['ppl']/baseline['ppl']-1
                result = {'id': identifier, 'seed': seed, 'specification': spec, 'checkpoint': str(checkpoint.resolve()),
                          'metadata': metadata, 'metrics': metric}
                results.append(result)
                write(output/'metrics.json', results)
                del student
                torch.cuda.empty_cache()
                if tensor_digest(teacher) != original:
                    raise RuntimeError('teacher mutated')
            del inputs
        progress('all declared comparisons complete; final budget check')
        if stage == 'regularization':
            selected_anchor = select_result(results, {'anchored'})
            selected_solver = select_result(results, {'anchored', 'legacy_ridge'})
            strongest = select_result(results, {'asvd'})
            # Whitening is the beta=0 anchored component under the declared pseudoinverse.
            whitening = next(r for r in results if r['specification'].get('beta') == 0 and r['specification']['method'] == 'anchored')
            strongest = min([strongest, whitening], key=lambda r: (r['metrics']['nll'], r['id']))
            strongest = copy.deepcopy(strongest)
            strongest['validation_source_specification'] = dict(strongest['specification'])
            strongest['validation_source_result_id'] = strongest['id']
            if strongest['specification']['method'] == 'anchored':
                strongest['specification']['method'] = 'whitening'
            write(output/'selection.json', {'state': 'frozen', 'selection_split': 'validation',
                'validation_hash': manifests['validation']['hash'], 'model_revision': config['model_revision'],
                'dataset_revision': config['dataset_revision'], 'rank': config['rank'], 'precision': 'float16',
                'tie_nll_tolerance': 1e-6, 'selection_seed': 11,
                'teacher_hash': original, 'dataset_content_hash': dataset_hash,
                'split_hashes': {name: manifests[name]['hash'] for name in ['train', 'validation', 'test']},
                'calibration_token_hashes': manifests['calibration_hashes'], 'targets': target_recipes,
                'factor_convention': config.get('factor_convention', 'unbalanced'), 'selected_anchor': selected_anchor,
                'selected_solver': selected_solver, 'strongest_baseline': strongest})
        write(output/'status.json', {'state': 'completed', 'elapsed_seconds': time.monotonic()-started})
        rows = ['| Method | Seed | NLL | PPL | KL | Primary drift |',
                '|---|---:|---:|---:|---:|---:|']
        for result in results:
            m = result['metrics']
            rows.append(f"| {result['id']} | {result['seed']} | {m['nll']:.6f} | {m['ppl']:.4f} | "
                        f"{m['logit_kl']:.6f} | {m.get('primary_drift', 'not measured')} |")
        diagnostic = ''
        if stage == 'regularization':
            full = next(r for r in results if r['specification']['method'] == 'fullrank_ridge')
            diagnostic = (f"\nFull-rank shrinkage validation NLL change versus teacher: "
                          f"{full['metrics']['nll']-baseline['nll']:+.6f}. Selected solver change: "
                          f"{selected_solver['metrics']['nll']-baseline['nll']:+.6f}. "
                          'A matching sign alone does not establish mechanistic equivalence.\n')
        (output/'report.md').write_text(f'# {stage}\n\nCompleted {len(results)} comparisons on {split}. '
            'Hypothesis status: inconclusive pending paired uncertainty, three-seed global comparisons and hardware gates. '
            'Component baselines are not complete ASVD/SVD-LLM pipelines.\n\n'+ '\n'.join(rows)+'\n'+diagnostic+
            '\nNo promotion claim: hardware limits and second architecture remain separate gates.\n', encoding='utf-8')
        return 0
    except Exception as exc:
        (output/'stderr.log').write_text(traceback.format_exc(), encoding='utf-8')
        state = 'incomplete' if isinstance(exc, TimeoutError) else 'failed'
        write(output/'status.json', {'state': state, 'error': str(exc), 'elapsed_seconds': time.monotonic()-started})
        (output/'report.md').write_text(f'# {stage}\n\n{state}: {exc}\nPartial results preserved.\n', encoding='utf-8')
        print(f'{stage} {state}: {output}: {exc}', file=sys.stderr)
        return 1
