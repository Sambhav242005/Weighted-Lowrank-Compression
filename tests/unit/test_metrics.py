import pytest
import torch
import torch.nn.functional as F
from transformers import BatchEncoding, GPT2Config, GPT2LMHeadModel

from src.baseline_eval import compute_logit_divergence, compute_perplexity


class FixedPaddedTokenizer:
    def __call__(self, text, return_tensors, truncation, max_length):
        del text, return_tensors, truncation, max_length
        return BatchEncoding(
            {
                "input_ids": torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
            }
        )


def make_tiny_model():
    torch.manual_seed(23)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=7,
            n_positions=4,
            n_ctx=4,
            n_embd=8,
            n_layer=1,
            n_head=2,
            n_inner=16,
            pad_token_id=0,
        )
    )
    return model.eval()


def test_identical_logits_have_zero_divergence_and_full_agreement():
    torch.manual_seed(29)
    logits = torch.randn(2, 3, 5)

    metrics = compute_logit_divergence(logits, logits.clone())

    assert metrics["kl_orig_to_sub"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["kl_sub_to_orig"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["cosine_similarity"] == pytest.approx(1.0, abs=1e-7)
    assert metrics["top1_agreement"] == pytest.approx(1.0)


def test_perplexity_counts_only_valid_shifted_non_padding_targets():
    model = make_tiny_model()
    tokenizer = FixedPaddedTokenizer()
    input_ids = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
    shift_targets = input_ids[:, 1:].contiguous().view(-1)
    shift_targets[attention_mask[:, 1:].contiguous().view(-1) == 0] = -100
    expected_loss = F.cross_entropy(shift_logits, shift_targets, ignore_index=-100)
    expected_perplexity = torch.exp(expected_loss).item()

    actual = compute_perplexity(model, tokenizer, ["offline"], max_length=4)

    assert actual == pytest.approx(expected_perplexity, rel=1e-6)
