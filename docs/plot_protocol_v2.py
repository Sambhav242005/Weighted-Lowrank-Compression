"""Plot retained completed C/D/F measurements; never fit, download or use CUDA.

Example (after runs complete):
rtk python docs/plot_protocol_v2.py --baselines RUN_C --composition RUN_D --practical RUN_F
Requires an already-installed matplotlib. Outputs a fresh docs/figures directory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import uuid


def load_run(directory):
    directory = Path(directory).resolve()
    status = json.loads((directory/"status.json").read_text(encoding="utf-8"))
    if status.get("state") != "completed":
        raise ValueError(f"plot requires completed run: {directory}")
    metrics = directory/"metrics.json"
    records = json.loads(metrics.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"no comparison records: {directory}")
    return records, {"directory": str(directory), "metrics_sha256": hashlib.sha256(metrics.read_bytes()).hexdigest(),
                     "contract_sha256": hashlib.sha256((directory/"contract.json").read_bytes()).hexdigest()}


def canonical(spec):
    value = dict(spec)
    if value.get("method") == "whitening" or (value.get("method") == "anchored" and
            value.get("beta") == 0 and value.get("ridge_fraction", 0) == 0):
        value.update(method="whitening", beta=0.0, ridge_fraction=0.0)
    return value


def quality_groups(records):
    groups = {}
    for record in records:
        spec = canonical(record.get("specification", {}))
        scope = {key: record[key] for key in ("group", "rank", "split") if key in record}
        key = json.dumps({"specification": spec, "scope": scope}, sort_keys=True)
        group = groups.setdefault(key, {})
        seed = record.get("seed")
        if seed in group:
            prior = group[seed]
            def digest(r):
                return r.get("metadata", {}).get("sha256") or r.get("checkpoint_sha256")
            if not digest(record) or digest(record) != digest(prior) or record.get("metrics") != prior.get("metrics"):
                raise ValueError(f"conflicting/unverified quality aliases: {key}, seed={seed}")
        else:
            group[seed] = record
    return groups


def drift_figure(plt, groups, output, title):
    valid = [(key, entries) for key, entries in groups.items()
             if all(r.get("metrics", {}).get("residual_by_block") for r in entries.values())]
    if not valid:
        raise ValueError("no measured per-block residual drift; no figure generated")
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for key, entries in valid:
        rows = [[block["normalized_squared_error"] for block in r["metrics"]["residual_by_block"]]
                for r in entries.values()]
        if len({len(row) for row in rows}) != 1 or any(not math.isfinite(x) for row in rows for x in row):
            raise ValueError("inconsistent/nonfinite drift measurements")
        mean = [statistics.mean(values) for values in zip(*rows)]
        std = [statistics.stdev(values) if len(values) > 1 else 0 for values in zip(*rows)]
        descriptor = json.loads(key)
        label = json.dumps(descriptor["specification"], sort_keys=True)
        if descriptor["scope"]:
            label += " " + json.dumps(descriptor["scope"], sort_keys=True)
        x = list(range(len(mean)))
        line, = ax.plot(x, mean, label=f"{label}; {len(rows)} seed(s)")
        if len(rows) > 1:
            ax.fill_between(x, [m-s for m,s in zip(mean,std)], [m+s for m,s in zip(mean,std)],
                            color=line.get_color(), alpha=.12)
    ax.set(xlabel="Block index", ylabel="Normalized residual squared error", title=title)
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1,1))
    ax.grid(alpha=.2)
    for suffix in ("png", "svg"):
        fig.savefig(output.with_suffix("."+suffix), dpi=180)
    plt.close(fig)


def storage_figure(plt, records, output):
    fig, ax = plt.subplots(figsize=(8,5), layout="constrained")
    count = 0
    for record in records:
        size = record.get("whole_checkpoint", {}).get("bytes")
        nll = record.get("calibration", {}).get("raw_test", {}).get("nll")
        if size is None or nll is None:
            continue
        if not math.isfinite(size) or size <= 0 or not math.isfinite(nll):
            raise ValueError("nonfinite storage/quality measurement")
        label = f"{record.get('label')} seed={record.get('seed','reference')} bits={record.get('bits','dense')}"
        ax.scatter(size/(1024**2), nll, label=label)
        count += 1
    if not count:
        plt.close(fig)
        raise ValueError("no measured whole-checkpoint bytes and raw-test NLL; no storage figure")
    ax.set(xlabel="Actual serialized whole-model checkpoint (MiB)", ylabel="Raw native-test NLL",
           title="Measured quality versus whole-model storage")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1,1))
    ax.grid(alpha=.2)
    for suffix in ("png", "svg"):
        fig.savefig(output.with_suffix("."+suffix), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in ("baselines", "composition", "practical"):
        parser.add_argument("--"+stage)
    args = parser.parse_args()
    selected = {stage: load_run(getattr(args, stage)) for stage in ("baselines", "composition", "practical")
                if getattr(args, stage)}
    if not selected:
        parser.error("provide at least one completed C/D/F run directory")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(__file__).resolve().parent/"figures"/(datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"sources": {stage: evidence for stage, (_, evidence) in selected.items()},
                "interpretation": "Measured results only; shaded bands are sample seed standard deviations, not confidence intervals.",
                "scope": "Whole-model serialized bytes include archive overhead; no compression-success claim follows from a plot."}
    (output/"provenance.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    for stage in ("baselines", "composition"):
        if stage in selected:
            drift_figure(plt, quality_groups(selected[stage][0]), output/(stage+"-drift"),
                         "Measured residual drift versus depth ("+stage+")")
    if "practical" in selected:
        storage_figure(plt, selected["practical"][0], output/"quality-storage")
    print(output)


if __name__ == "__main__":
    main()
