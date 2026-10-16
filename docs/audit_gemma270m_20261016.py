"""Read-only checkpoint/evidence audit after the isolated Gemma run completes."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args()
    root = args.run
    output = root / "audit.json"
    if output.exists():
        raise FileExistsError("Preserve existing audit")
    cfg = json.loads((root / "resolved-config.json").read_text())
    metrics = json.loads((root / "metrics.json").read_text())
    status = json.loads((root / "status.json").read_text())
    artifact_root = Path(cfg.get("recovery", {}).get("source_run", root))
    assert status["state"] == "completed", status
    assert cfg["dtype"] == "float32" and cfg["seeds"] == [11, 23]
    assert all(sha(path) == expected for path, expected in cfg["source_hashes"].items())
    torch.set_num_threads(4)
    dense = torch.load(artifact_root / "dense-full.pt", map_location="cpu", weights_only=True, mmap=True)
    # The teacher's saved FP32 weights must exactly equal the pinned BF16 originals upcast to FP32.
    from safetensors import safe_open
    assets = json.loads(Path("paper-assets/gemma270m-study-assets-20261016.json").read_text())
    with safe_open(str(Path(assets["snapshot"]) / "model.safetensors"), framework="pt", device="cpu") as source:
        teacher_checks = 0
        for key in source.keys():
            assert key in dense, key
            assert torch.equal(dense[key], source.get_tensor(key).float()), key
            teacher_checks += 1
    assert dense["lm_head.weight"].data_ptr() == dense["model.embed_tokens.weight"].data_ptr()
    checks = {}
    for method in metrics["storage"]:
        file = artifact_root / (method + "-full.pt")
        assert file.stat().st_size == metrics["storage"][method]["full_checkpoint_bytes"]
        assert sha(file) == metrics["storage"][method]["full_sha256"]
        checkpoint = torch.load(file, map_location="cpu", weights_only=True, mmap=True)
        assert checkpoint["lm_head.weight"].data_ptr() == checkpoint["model.embed_tokens.weight"].data_ptr()
        unchanged = 0
        for key, value in dense.items():
            if ".self_attn.o_proj." not in key:
                assert torch.equal(checkpoint[key], value), (method, key)
                unchanged += 1
        if method != "dense":
            family, seed = method.rsplit("_seed", 1)
            replacements = torch.load(artifact_root / f"replacements-seed{seed}.pt", map_location="cpu", weights_only=True)
            subkey = "whitening" if family == "whitening" else "relu"
            for i, state in enumerate(replacements[subkey]):
                for k, tensor in state.items():
                    assert torch.equal(checkpoint[f"model.layers.{i}.self_attn.o_proj.{k}"], tensor)
        checks[method] = {"checkpoint_hash_matches": True, "unchanged_non_target_tensors": unchanged,
                          "reload_matches_saved_replacements": True, "tied_embedding_preserved": True}
        del checkpoint
    for seed, fit in metrics["fitting"].items():
        assert len(fit) == 18
        assert all(r["full_rank_validation_relative_rmse"] < .01 for r in fit)
        assert all(r["linear_parameters"] == 457600 and r["relu_parameters"] == 458515 for r in fit)
        assert all(r["relu_parameters"] <= r["dense_parameters"] * .7 for r in fit)
    expected = cfg["splits"]["test"]["tokens"] - 1
    for q in metrics["quality"].values():
        assert q["targets"] == expected and len(q["block_relative_rmse"]) == 18
        assert all(math.isfinite(v) for v in q["block_relative_rmse"])
    for methods in metrics["performance"].values():
        assert len(methods) == 5
        for timings in methods.values():
            for workload, timing in timings.items():
                if isinstance(timing, dict):
                    assert len(timing["samples_ms"]) == 20
                    assert all(math.isfinite(x) and x > 0 for x in timing["samples_ms"])
    assert all(r["last_logit_relative_rmse"] < 1e-4 and r["last_token_agreement"]
               for r in metrics["cpu_gpu_parity"].values())
    a = torch.load(artifact_root / "replacements-seed11.pt", map_location="cpu", weights_only=True)
    b = torch.load(artifact_root / "replacements-seed23.pt", map_location="cpu", weights_only=True)
    assert all(torch.equal(a["whitening"][i][key], b["whitening"][i][key])
               for i in range(18) for key in a["whitening"][i])
    audit = {"state": "passed", "teacher_tensors_matching_upstream": teacher_checks,
             "checkpoint_checks": checks, "test_targets_per_method": expected,
             "source_hashes_match": True, "whitening_reuse_verified": True,
             "limitations": ["One architecture; two optimization seeds and shared calibration windows",
                             "Common ceiling, not exact parameter equality (ReLU biases)",
                             "CPU likelihood is not evaluated over the full test; parity uses one window",
                             "Fixed benchmark order and observed repeated-baseline timing variability",
                             "Process RSS is cumulative; reserved GPU memory includes allocator history",
                             "Standard PyTorch kernels, no custom fused CUDA implementation"]}
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": "passed", "test_targets_per_method": expected,
                      "teacher_tensors_matching_upstream": teacher_checks}))


if __name__ == "__main__":
    main()
