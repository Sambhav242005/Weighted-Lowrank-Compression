"""
Self-Improving Research Loop
============================
Orchestrates experiments, logs results, mines failures for next direction.

Loop:
1. Run experiment (drift_profiler.py or joint_group_compress.py)
2. Parse results → extract metrics
3. Classify: success / partial / failure
4. Mine failures: extract failure patterns → candidate next experiments
5. Update frontier.json, rejected.jsonl, THREAD.md
6. Return next candidate

Does NOT auto-run next experiment — requires human approval.
"""

import json, sys, os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESEARCH_LOOP_DIR = PROJECT_ROOT / "research-loop"
FRONTIER_PATH = RESEARCH_LOOP_DIR / "frontier.json"
REJECTED_PATH = RESEARCH_LOOP_DIR / "rejected.jsonl"
THREAD_PATH = PROJECT_ROOT / "THREAD.md"
CANDIDATES_DIR = RESEARCH_LOOP_DIR / "candidates"

# Ensure dirs exist
RESULTS_DIR.mkdir(exist_ok=True)
RESEARCH_LOOP_DIR.mkdir(exist_ok=True)
CANDIDATES_DIR.mkdir(exist_ok=True)


@dataclass
class ExperimentResult:
    """Parsed result from a completed experiment."""
    experiment_id: str
    experiment_type: str  # "drift_profiler" | "joint_group_compress"
    timestamp: str
    group_size: int
    replaced_layers: List[int]
    ppl_orig: float
    ppl_sub: float
    ppl_delta_pct: float
    compression_ratio: float
    metrics: Dict  # additional metrics specific to the experiment


@dataclass
class Candidate:
    """A proposed next experiment."""
    candidate_id: str
    hypothesis: str
    experiment_type: str
    config: Dict
    rationale: str
    expected_outcome: str
    decision_rule: str
    status: str = "pending"  # pending | running | completed | rejected


def load_frontier() -> List[Dict]:
    """Load Pareto frontier."""
    if FRONTIER_PATH.exists():
        with open(FRONTIER_PATH) as f:
            return json.load(f)
    return []


def save_frontier(frontier: List[Dict]):
    """Save Pareto frontier."""
    with open(FRONTIER_PATH, 'w') as f:
        json.dump(frontier, f, indent=2)


def append_rejected(result: Dict):
    """Append to rejected log."""
    with open(REJECTED_PATH, 'a') as f:
        f.write(json.dumps(result) + "\n")


def classify_result(result: ExperimentResult) -> str:
    """
    Classify experiment result.
    - success: PPL delta < -5%, composition works
    - partial: PPL delta < 0%, some drift but manageable
    - failure: PPL delta > 0%, drift breaks composition
    """
    if result.ppl_delta_pct < -5:
        return "success"
    elif result.ppl_delta_pct < 0:
        return "partial"
    elif result.ppl_delta_pct < 50:
        return "failure"
    else:
        return "catastrophic_failure"


def mine_failure_patterns(results: List[ExperimentResult]) -> List[Dict]:
    """
    Mine failure patterns from completed experiments.
    Returns candidate next experiments based on observed failures.
    """
    candidates = []
    
    # Pattern 1: Drift accumulates linearly with group size
    if len(results) >= 3:
        group_sizes = sorted(set(r.group_size for r in results))
        deltas_by_size = {}
        for r in results:
            deltas_by_size.setdefault(r.group_size, []).append(r.ppl_delta_pct)
        
        avg_deltas = {gs: np.mean(deltas_by_size[gs]) for gs in group_sizes}
        
        if len(avg_deltas) >= 2:
            deltas = [avg_deltas[gs] for gs in sorted(avg_deltas.keys())]
            if all(deltas[i] < deltas[i+1] for i in range(len(deltas)-1)):
                # Monotonic increase — drift is composition-dependent
                candidates.append(Candidate(
                    candidate_id=f"failure-drift-{datetime.now().strftime('%Y%m%d-%H%M')}",
                    hypothesis="Drift accumulation is sublinear — testing intermediate group sizes",
                    experiment_type="joint_group_compress",
                    config={"group_sizes": [3, 5, 6], "rank": 128},
                    rationale="Observed monotonic drift increase. Test if intermediate sizes show sublinear pattern.",
                    expected_outcome="Intermediate group sizes show sublinear PPL increase",
                    decision_rule="If PPL delta < average of neighbors, drift is sublinear"
                ))
    
    # Pattern 2: Some layers cause disproportionate drift
    layer_deltas = {}
    for r in results:
        if r.group_size == 1:
            layer_deltas[r.replaced_layers[0]] = r.ppl_delta_pct
    
    if layer_deltas:
        worst_layer = max(layer_deltas, key=layer_deltas.get)
        best_layer = min(layer_deltas, key=layer_deltas.get)
        
        if layer_deltas[worst_layer] > 2 * layer_deltas[best_layer]:
            candidates.append(Candidate(
                candidate_id=f"failure-asymmetry-{datetime.now().strftime('%Y%m%d-%H%M')}",
                hypothesis="Drift is asymmetric across layers — some layers are more sensitive",
                experiment_type="joint_group_compress",
                config={"group_sizes": [2], "target_layers": [worst_layer, best_layer], "rank": 128},
                rationale=f"Layer {worst_layer} has {layer_deltas[worst_layer]:+.2f}% delta vs {best_layer} at {layer_deltas[best_layer]:+.2f}%",
                expected_outcome="Sensitive layers show worse compression, suggesting layer-specific representations",
                decision_rule="If group including worst layer has >2x PPL delta vs group with best layer, layer sensitivity is real"
            ))
    
    # Pattern 3: Independent vs joint comparison
    if len(results) >= 2:
        joint_results = [r for r in results if r.experiment_type == "joint_group_compress"]
        independent_results = [r for r in results if r.experiment_type == "drift_profiler"]
        
        if joint_results and independent_results:
            candidates.append(Candidate(
                candidate_id=f"failure-compare-{datetime.now().strftime('%Y%m%d-%H%M')}",
                hypothesis="Joint compression reduces drift through shared structure",
                experiment_type="joint_group_compress",
                config={"group_sizes": [4, 8], "rank": 128},
                rationale="Direct comparison of independent vs joint at same group sizes",
                expected_outcome="Joint compression shows lower PPL delta than independent at same group size",
                decision_rule="If joint PPL delta < independent PPL delta, shared structure exists"
            ))
    
    return candidates


def update_thread(results: List[ExperimentResult], candidates: List[Candidate]):
    """Update THREAD.md with latest findings."""
    lines = []
    lines.append("# Research Thread\n")
    lines.append(f"Last updated: {datetime.now().isoformat()}\n")
    lines.append("## Completed Experiments\n")
    
    for r in results:
        status = classify_result(r)
        lines.append(f"### {r.experiment_id}")
        lines.append(f"- Type: {r.experiment_type}")
        lines.append(f"- Group size: {r.group_size}")
        lines.append(f"- PPL: {r.ppl_orig:.2f} -> {r.ppl_sub:.2f} ({r.ppl_delta_pct:+.2f}%)")
        lines.append(f"- Classification: {status}")
        lines.append("")
    
    lines.append("## Candidate Next Experiments\n")
    for c in candidates:
        lines.append(f"### {c.candidate_id}")
        lines.append(f"- Hypothesis: {c.hypothesis}")
        lines.append(f"- Rationale: {c.rationale}")
        lines.append(f"- Decision rule: {c.decision_rule}")
        lines.append("")
    
    with open(THREAD_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def parse_drift_profiler_results(results_path: Path) -> List[ExperimentResult]:
    """Parse drift_profiler.json into ExperimentResult objects."""
    if not results_path.exists():
        return []
    
    with open(results_path) as f:
        data = json.load(f)
    
    results = []
    for r in data:
        # Compute average drift across layers
        layer_drift = r.get("layer_drift", [])
        avg_cosine = np.mean([d["cosine"] for d in layer_drift]) if layer_drift else 0.0
        avg_cka = np.mean([d["cka"] for d in layer_drift]) if layer_drift else 0.0
        final_mse = layer_drift[-1]["mse"] if layer_drift else 0.0
        
        results.append(ExperimentResult(
            experiment_id=f"drift-{r['group_size']}-L{''.join(str(l) for l in r['replaced_layers'])}",
            experiment_type="drift_profiler",
            timestamp=datetime.now().isoformat(),
            group_size=r["group_size"],
            replaced_layers=r["replaced_layers"],
            ppl_orig=r["ppl_orig"],
            ppl_sub=r["ppl_sub"],
            ppl_delta_pct=r["ppl_delta_pct"],
            compression_ratio=3.0,
            metrics={
                "avg_cosine": avg_cosine,
                "avg_cka": avg_cka,
                "final_mse": final_mse,
            }
        ))
    
    return results


def parse_joint_results(results_path: Path) -> List[ExperimentResult]:
    """Parse joint_group_compress.json into ExperimentResult objects."""
    if not results_path.exists():
        return []
    
    with open(results_path) as f:
        data = json.load(f)
    
    results = []
    for r in data:
        results.append(ExperimentResult(
            experiment_id=f"joint-{r['group_size']}-L{''.join(str(l) for l in r['replaced_layers'])}",
            experiment_type="joint_group_compress",
            timestamp=datetime.now().isoformat(),
            group_size=r["group_size"],
            replaced_layers=r["replaced_layers"],
            ppl_orig=r["ppl_orig"],
            ppl_sub=r["ppl_sub"],
            ppl_delta_pct=r["ppl_delta_pct"],
            compression_ratio=r["compression_ratio"],
            metrics={
                "total_params": r["total_params"],
                "original_params": r["original_params"],
            }
        ))
    
    return results


def run_analysis():
    """Run analysis on all completed experiments."""
    print("=" * 70)
    print("SELF-IMPROVING RESEARCH LOOP — ANALYSIS")
    print("=" * 70)
    
    # Load all results
    drift_results = parse_drift_profiler_results(RESULTS_DIR / "drift_profiler.json")
    joint_results = parse_joint_results(RESULTS_DIR / "joint_group_compress.json")
    
    all_results = drift_results + joint_results
    
    if not all_results:
        print("No experiment results found. Run drift_profiler.py or joint_group_compress.py first.")
        return
    
    print(f"Found {len(all_results)} experiment results")
    
    # Classify
    print("\n--- Classification ---")
    for r in all_results:
        status = classify_result(r)
        print(f"  {r.experiment_id}: {status} (PPL D = {r.ppl_delta_pct:+.2f}%)")
    
    # Mine failures
    print("\n--- Mining failure patterns ---")
    candidates = mine_failure_patterns(all_results)
    
    for c in candidates:
        print(f"\n  Candidate: {c.candidate_id}")
        print(f"    Hypothesis: {c.hypothesis}")
        print(f"    Rationale: {c.rationale}")
        print(f"    Decision rule: {c.decision_rule}")
    
    # Save candidates
    for c in candidates:
        candidate_path = CANDIDATES_DIR / f"{c.candidate_id}.json"
        with open(candidate_path, 'w') as f:
            json.dump(asdict(c), f, indent=2)
        print(f"\n  Saved candidate to {candidate_path}")
    
    # Update thread
    update_thread(all_results, candidates)
    print(f"\nUpdated {THREAD_PATH}")
    
    # Report
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Total experiments: {len(all_results)}")
    print(f"Candidates generated: {len(candidates)}")
    print(f"Status: Waiting for human approval to run next experiment")


import numpy as np

if __name__ == "__main__":
    run_analysis()
