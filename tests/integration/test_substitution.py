import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.substitution import substitute_weight_in_model


def make_tiny_gpt2():
    torch.manual_seed(41)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=19,
            n_positions=8,
            n_ctx=8,
            n_embd=8,
            n_layer=2,
            n_head=2,
            n_inner=16,
            pad_token_id=0,
        )
    ).eval()


def test_substitution_copies_teacher_and_changes_only_student():
    teacher = make_tiny_gpt2()
    teacher_state = {
        name: parameter.detach().clone()
        for name, parameter in teacher.named_parameters()
    }
    original_weight = teacher.transformer.h[0].attn.c_proj.weight.detach().clone()
    replacement = torch.zeros_like(original_weight)

    student = substitute_weight_in_model(teacher, "layer0.attn.W_O", replacement)

    assert student is not teacher
    assert torch.equal(student.transformer.h[0].attn.c_proj.weight, replacement)
    assert all(
        torch.equal(parameter, teacher_state[name])
        for name, parameter in teacher.named_parameters()
    )

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        teacher_logits = teacher(input_ids=input_ids).logits
        student_logits = student(input_ids=input_ids).logits

    assert not torch.equal(teacher_logits, student_logits)
