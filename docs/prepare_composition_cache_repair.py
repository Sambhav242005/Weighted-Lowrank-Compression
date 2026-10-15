"""Prepare a proposed source patch without editing active experiment source."""
from pathlib import Path
import difflib


HELPERS = '''
def _composition_source_artifacts(config):
    """Validate an optional completed-C boundary manifest before reusing metrics."""
    directory = Path(config["baseline_run_path"])
    declared = config.get("baseline_artifact_integrity_path")
    integrity = Path(declared) if declared else directory/"artifact-integrity.json"
    verified = False
    if declared or integrity.exists():
        manifest = json.loads(integrity.read_text(encoding="utf-8"))
        artifacts = manifest.get("artifacts", {})
        for filename in ("metrics.json", "contract.json", "teacher.json"):
            if not artifacts.get(filename) or sha256_file(directory/filename) != artifacts[filename]:
                raise ValueError(f"completed Stage C artifact changed: {filename}")
        verified = True
    return {"run_id": directory.name, "metrics_sha256": sha256_file(directory/"metrics.json"),
            "boundary_integrity_verified": verified,
            "integrity_manifest_sha256": sha256_file(integrity) if verified else None}


def _composition_record(label, seed, rank, group, names, entry, source_artifacts,
                        evaluate_fresh, *, validated_source):
    """Only the exact all-target condition can reuse validated completed-C metrics."""
    if not validated_source:
        raise ValueError("fresh teacher/data/target/checkpoint validation is required")
    group = tuple(group)
    if not group or any(i < 0 or i >= len(names) for i in group) or group != tuple(range(group[0], group[-1]+1)):
        raise ValueError("composition group must be a valid consecutive ordered group")
    full_stack = group == tuple(range(len(names)))
    recipe_by_name = {recipe["name"]: recipe for recipe in entry["metadata"]["targets"]}
    selected_recipes = [recipe_by_name[names[index]] for index in group]
    if any(recipe["rank"] != rank for recipe in selected_recipes):
        raise ValueError("composition rank differs from cached factors")
    reused = {}
    if full_stack:
        metrics = copy.deepcopy(entry["metrics"])
        reused = {"reused_from_run": source_artifacts["run_id"], "source_record_id": entry["id"],
                  "source_metrics_sha256": source_artifacts["metrics_sha256"],
                  "source_boundary_integrity_verified": source_artifacts["boundary_integrity_verified"],
                  "source_integrity_manifest_sha256": source_artifacts["integrity_manifest_sha256"],
                  "independent_replication": False,
                  "evaluation_origin": "reused completed Stage C full-stack measurement"}
    else:
        metrics = evaluate_fresh()
        reused = {"evaluation_origin": "fresh complete native test evaluation"}
    if len(metrics["residual_by_block"]) != len(names):
        raise ValueError("source/fresh drift block count mismatch")
    metrics["group_end_drift"] = copy.deepcopy(metrics["residual_by_block"][group[-1]])
    metrics["downstream_drift"] = copy.deepcopy(metrics["residual_by_block"][group[-1]+1:])
    metadata = {"factor_parameters": sum(recipe["factor_parameters"] for recipe in selected_recipes),
                "targets": copy.deepcopy(selected_recipes), "precision": entry["metadata"]["precision"],
                "factor_convention": entry["metadata"].get("factor_convention", "unbalanced"),
                "serialized_bytes": entry["metadata"]["serialized_bytes"] if full_stack else None,
                "storage_scope": "full-stack factor checkpoint" if full_stack else "subgroup bytes not measured"}
    return {"label": label, "specification": copy.deepcopy(entry["specification"]), "seed": seed,
            "rank": rank, "group": list(group), "split": "test", "checkpoint": entry["checkpoint"],
            "checkpoint_sha256": entry["metadata"]["sha256"], "metadata": metadata,
            "metrics": metrics, **reused}

'''


def proposed_source(source):
    source = source.replace("import math\n", "import math\nimport copy\n", 1)
    marker = "def run_followups(config, stage):\n"
    assert source.count(marker) == 1
    source = source.replace(marker, HELPERS+"\n"+marker, 1)
    old = '''            teacher_result = evaluate(teacher, tokens["test"], progress=progress)
            write(output/"teacher.json", teacher_result)
            def execute(label, seed, rank, group, split):
                spec = frozen[label]
                entry = source[label, seed]
                student = student_from_checkpoint(teacher, entry["checkpoint"], names=[names[i] for i in group])
                metrics = evaluate_diagnostics(student, teacher, tokens[split], config, progress)
                metrics["group_end_drift"] = metrics["residual_by_block"][group[-1]]
                metrics["downstream_drift"] = metrics["residual_by_block"][group[-1]+1:]
                result = persist({"label": label, "specification": spec, "seed": seed, "rank": rank,
                    "group": list(group), "split": split, "checkpoint": entry["checkpoint"],
                    "checkpoint_sha256": entry["metadata"]["sha256"], "metrics": metrics})
                del student
                torch.cuda.empty_cache()
                return result
'''
    new = '''            source_artifacts = _composition_source_artifacts(config)
            teacher_result = evaluate(teacher, tokens["test"], progress=progress)
            write(output/"teacher.json", teacher_result)
            def execute(label, seed, rank, group, split):
                if split != "test":
                    raise ValueError("composition protocol requires the native test split")
                entry = source[label, seed]
                def evaluate_fresh():
                    student = student_from_checkpoint(teacher, entry["checkpoint"], names=[names[i] for i in group])
                    try:
                        return evaluate_diagnostics(student, teacher, tokens[split], config, progress)
                    finally:
                        del student
                        torch.cuda.empty_cache()
                result = _composition_record(label, seed, rank, group, names, entry, source_artifacts,
                    evaluate_fresh, validated_source=True)
                return persist(result)
'''
    assert source.count(old) == 1, "active composition source differs; review proposal before rebasing"
    return source.replace(old, new, 1)


if __name__ == "__main__":
    original = Path("src/paper_followups.py").read_text(encoding="utf-8")
    proposed = proposed_source(original)
    compile(proposed, "proposed-paper-followups.py", "exec")
    patch = "".join(difflib.unified_diff(original.splitlines(True), proposed.splitlines(True),
                     fromfile="a/src/paper_followups.py", tofile="b/src/paper_followups.py"))
    path = Path("docs/composition_cache_repair.patch")
    if path.exists():
        raise FileExistsError(path)
    path.write_text(patch, encoding="utf-8")
    print(path)
