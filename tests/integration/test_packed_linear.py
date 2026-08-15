import torch

from src.compression_core import PackedLowRankLinear, checkpoint_has_dense_weight


def test_serialized_packed_layer_can_be_loaded_without_dense_weight():
    torch.manual_seed(3)
    original = PackedLowRankLinear.from_factors(
        torch.randn(4, 2), torch.randn(2, 6), bits=4
    )
    restored = PackedLowRankLinear.from_factors(
        torch.zeros(4, 2), torch.zeros(2, 6), bits=4
    )
    restored.load_state_dict(original.state_dict())
    inputs = torch.randn(2, 6)
    assert torch.allclose(restored(inputs), original(inputs))
    assert not checkpoint_has_dense_weight(restored.state_dict())


def test_teacher_parameters_are_not_mutated_by_constructing_student():
    teacher_u = torch.randn(4, 2)
    teacher_v = torch.randn(2, 6)
    teacher_snapshot = (teacher_u.clone(), teacher_v.clone())
    _student = PackedLowRankLinear.from_factors(teacher_u, teacher_v, bits=8)
    assert torch.equal(teacher_u, teacher_snapshot[0])
    assert torch.equal(teacher_v, teacher_snapshot[1])
