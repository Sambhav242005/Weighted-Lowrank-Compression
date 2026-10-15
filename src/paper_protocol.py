"""Deterministic calibration and exactly-once shifted-target evaluation."""
import hashlib
import math

import torch
import torch.nn.functional as F


def token_hash(tokens):
    return hashlib.sha256(tokens.long().cpu().numpy().tobytes()).hexdigest()


def calibration_starts(length, seed, *, count=16, chunk=512):
    if length // chunk < count:
        raise ValueError("insufficient non-overlapping calibration chunks")
    g = torch.Generator().manual_seed(seed)
    return (torch.randperm(length // chunk, generator=g)[:count] * chunk).tolist()


def windows(length, context=512, stride=256):
    if not 1 <= stride < context:
        raise ValueError("stride must be positive and smaller than context")
    previous_end = 1
    for start in range(0, max(0, length - 1), stride):
        end = min(start + context, length)
        first_target = max(previous_end, start + 1)
        if first_target < end:
            yield start, end, first_target
        previous_end = end
        if end == length:
            break


@torch.inference_mode()
def evaluate(model, tokens, *, context=512, stride=256, teacher=None, progress=None):
    device = next(model.parameters()).device
    total, count, kl_total = 0.0, 0, 0.0
    nll_blocks, block_sum, block_count = [], 0.0, 0
    for index, (start, end, first) in enumerate(windows(len(tokens), context, stride)):
        ids = tokens[start:end].unsqueeze(0).to(device)
        logits = model(input_ids=ids, use_cache=False).logits[0]
        offset = first - start - 1
        scored = logits[offset:-1].float()
        labels = ids[0, offset + 1:]
        losses = F.cross_entropy(scored, labels, reduction="none")
        if not torch.isfinite(losses).all():
            raise FloatingPointError("NaN/Inf NLL")
        total += losses.double().sum().item()
        count += len(labels)
        for loss in losses.double().cpu().tolist():
            block_sum += loss
            block_count += 1
            if block_count == 4096:
                nll_blocks.append({"nll_sum": block_sum, "targets": block_count})
                block_sum, block_count = 0.0, 0
        if teacher is not None:
            ref = teacher(input_ids=ids, use_cache=False).logits[0, offset:-1].float()
            ref_log = ref.log_softmax(-1)
            kl = (ref_log.exp() * (ref_log - scored.log_softmax(-1))).sum(-1)
            if not torch.isfinite(kl).all():
                raise FloatingPointError("NaN/Inf KL")
            kl_total += kl.double().sum().item()
        if progress and index % 20 == 0:
            progress(f"evaluation window={index} targets={count}")
    if block_count:
        nll_blocks.append({"nll_sum": block_sum, "targets": block_count})
    if count != len(tokens) - 1 or count <= 0:
        raise ValueError(f"target count mismatch: {count} versus {len(tokens)-1}")
    return {"nll_sum": total, "targets": count, "nll": total/count,
            "ppl": math.exp(total/count), "logit_kl": kl_total/count if teacher is not None else None,
            "nll_blocks": nll_blocks}
