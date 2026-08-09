"""
Baseline model evaluation: perplexity, logit distributions, downstream tasks.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class EvalResult:
    perplexity: float
    logit_mean: float
    logit_std: float
    logit_max: float
    logit_min: float
    logit_entropy: float
    top1_accuracy: float
    top5_accuracy: float
    top10_accuracy: float
    token_losses: List[float]
    n_tokens: int


def compute_perplexity(
    model: GPT2LMHeadModel,
    tokenizer,
    texts: List[str],
    max_length: int = 512,
    device: str = "cpu",
) -> float:
    """Compute perplexity on a list of texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=max_length).to(device)
            labels = inputs["input_ids"].clone()

            # Causal-LM loss is computed on labels[:, 1:] after the model
            # shifts the sequence. Padding must therefore be ignored before
            # both loss accumulation and token accounting.
            if "attention_mask" in inputs:
                labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)
            n_tokens = int(labels[:, 1:].ne(-100).sum().item())
            if n_tokens == 0:
                continue
            
            outputs = model(**inputs, labels=labels)
            # outputs.loss is cross-entropy averaged over valid shifted tokens
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens
    
    if total_tokens == 0:
        raise ValueError("perplexity requires at least one valid target token")
    avg_loss = total_loss / total_tokens
    return float(np.exp(avg_loss))


def compute_logit_stats(
    model: GPT2LMHeadModel,
    tokenizer,
    texts: List[str],
    max_length: int = 256,
    device: str = "cpu",
) -> Dict:
    """Compute logit distribution statistics."""
    model.eval()
    all_logits = []
    all_targets = []
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_correct_top10 = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=max_length).to(device)
            outputs = model(**inputs)
            logits = outputs.logits.cpu()  # [1, seq_len, vocab_size]
            targets = inputs["input_ids"].cpu()  # [1, seq_len]
            
            # Shift: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            
            # Flatten
            flat_logits = shift_logits.view(-1, shift_logits.shape[-1])
            flat_targets = shift_targets.view(-1)
            
            all_logits.append(flat_logits)
            all_targets.append(flat_targets)
            
            # Accuracy
            top1 = flat_logits.argmax(dim=-1)
            top5 = flat_logits.topk(5, dim=-1).indices
            top10 = flat_logits.topk(10, dim=-1).indices
            
            total_correct_top1 += (top1 == flat_targets).sum().item()
            total_correct_top5 += (top5 == flat_targets.unsqueeze(1)).any(dim=1).sum().item()
            total_correct_top10 += (top10 == flat_targets.unsqueeze(1)).any(dim=1).sum().item()
            total_tokens += flat_targets.numel()
    
    all_logits = torch.cat(all_logits, dim=0)
    
    # Logit statistics
    logit_mean = float(all_logits.mean())
    logit_std = float(all_logits.std())
    logit_max = float(all_logits.max())
    logit_min = float(all_logits.min())
    
    # Entropy of softmax
    probs = F.softmax(all_logits, dim=-1)
    log_probs = F.log_softmax(all_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    
    return {
        "logit_mean": logit_mean,
        "logit_std": logit_std,
        "logit_max": logit_max,
        "logit_min": logit_min,
        "logit_entropy": float(entropy),
        "top1_accuracy": total_correct_top1 / total_tokens,
        "top5_accuracy": total_correct_top5 / total_tokens,
        "top10_accuracy": total_correct_top10 / total_tokens,
        "n_tokens": total_tokens,
    }


def get_baseline(
    model_name: str = "gpt2",
    n_eval_texts: int = 50,
    max_length: int = 256,
    device: str = "cpu",
) -> tuple:
    """Get full baseline evaluation of the model."""
    print(f"Loading {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    # Get evaluation texts
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [item["text"].strip() for item in dataset if len(item["text"].strip()) > 50]
    texts = texts[:n_eval_texts]
    
    print(f"Computing perplexity on {len(texts)} texts...")
    perplexity = compute_perplexity(model, tokenizer, texts, max_length, device)
    print(f"Baseline perplexity: {perplexity:.2f}")
    
    print("Computing logit statistics...")
    logit_stats = compute_logit_stats(model, tokenizer, texts, max_length, device)
    
    print(f"  Top-1 accuracy: {logit_stats['top1_accuracy']:.4f}")
    print(f"  Top-5 accuracy: {logit_stats['top5_accuracy']:.4f}")
    print(f"  Top-10 accuracy: {logit_stats['top10_accuracy']:.4f}")
    print(f"  Logit entropy: {logit_stats['logit_entropy']:.4f}")
    
    result = EvalResult(
        perplexity=perplexity,
        logit_mean=logit_stats["logit_mean"],
        logit_std=logit_stats["logit_std"],
        logit_max=logit_stats["logit_max"],
        logit_min=logit_stats["logit_min"],
        logit_entropy=logit_stats["logit_entropy"],
        top1_accuracy=logit_stats["top1_accuracy"],
        top5_accuracy=logit_stats["top5_accuracy"],
        top10_accuracy=logit_stats["top10_accuracy"],
        token_losses=[],
        n_tokens=logit_stats["n_tokens"],
    )
    
    return result, model, tokenizer, texts


def compute_logit_divergence(
    logits_original: torch.Tensor,
    logits_substituted: torch.Tensor,
) -> Dict:
    """Compute divergence between original and substituted model logits."""
    # KL divergence: KL(original || substituted)
    probs_orig = F.softmax(logits_original, dim=-1)
    log_probs_orig = F.log_softmax(logits_original, dim=-1)
    log_probs_sub = F.log_softmax(logits_substituted, dim=-1)
    
    # KL(orig || sub)
    kl_orig_sub = (probs_orig * (log_probs_orig - log_probs_sub)).sum(dim=-1).mean()
    
    # KL(sub || orig)
    probs_sub = F.softmax(logits_substituted, dim=-1)
    kl_sub_orig = (probs_sub * (log_probs_sub - log_probs_orig)).sum(dim=-1).mean()
    
    # Cosine similarity of logit vectors
    cos_sim = F.cosine_similarity(
        logits_original.view(-1, logits_original.shape[-1]),
        logits_substituted.view(-1, logits_substituted.shape[-1]),
        dim=-1
    ).mean()
    
    # Top-1 agreement
    top1_orig = logits_original.argmax(dim=-1)
    top1_sub = logits_substituted.argmax(dim=-1)
    top1_agreement = (top1_orig == top1_sub).float().mean()
    
    return {
        "kl_orig_to_sub": float(kl_orig_sub),
        "kl_sub_to_orig": float(kl_sub_orig),
        "cosine_similarity": float(cos_sim),
        "top1_agreement": float(top1_agreement),
    }
