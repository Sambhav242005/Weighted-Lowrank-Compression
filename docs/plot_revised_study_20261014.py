"""Scientific plots from a completed revised report; no model/CUDA imports.

Called by the finalizer, or: rtk python docs/plot_revised_study_20261014.py REPORT_DIR
Plots retain run identity. Missing measurements produce explicit unavailable flags.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def method_label(spec):
    """Readable legend only; exact original parameters stay in provenance."""
    method=spec.get("method")
    if method=="svd":
        return "SVD"
    if method=="asvd":
        return "ASVD"
    if method=="whitening":
        return "Whitening"
    if method=="anchored":
        return f"Anchor (beta={spec.get('beta',0):g})"
    if method in {"legacy_ridge","fullrank_ridge"}:
        prefix="Full-rank ridge" if method=="fullrank_ridge" else "Extra ridge"
        return f"{prefix} (beta={spec.get('beta',0):g}, ridge={spec.get('ridge_fraction',0):g})"
    return str(method or "Unspecified method")


def drift_series(run):
    if run.get("state")!="completed" or not run.get("run_id","").startswith("baselines-"):
        return []
    result=[]
    for group in run.get("method_groups",[]):
        rows=[]
        for record in group.get("records",[]):
            values=[b.get("normalized_squared_error") for b in record.get("metrics",{}).get("residual_by_block",[])]
            if not values:
                continue
            if not all(finite(v) and v>=0 for v in values):
                raise ValueError("nonfinite/negative drift: "+run["run_id"])
            rows.append({"seed":record.get("seed"),"values":values})
        if not rows:
            continue
        if len({len(r["values"]) for r in rows})!=1 or len({r["seed"] for r in rows})!=len(rows):
            raise ValueError("inconsistent block counts or duplicate seeds")
        if group.get("scope",{}).get("split","test")!="test":
            continue
        mean=[statistics.mean(x) for x in zip(*(r["values"] for r in rows))]
        sd=[statistics.stdev(x) if len(x)>1 else 0 for x in zip(*(r["values"] for r in rows))]
        result.append({"specification":group["specification"],"scope":group.get("scope",{}),
                       "per_seed":rows,"mean":mean,"sample_std":sd})
    return result


def storage_points(run):
    if run.get("state")!="completed" or not run.get("run_id","").startswith("practical-"):
        return []
    points=[]
    for group in run.get("method_groups",[]):
        for record in group.get("records",[]):
            size=record.get("whole_checkpoint",{}).get("bytes")
            nll=record.get("calibration",{}).get("raw_test",{}).get("nll")
            if size is None or nll is None:
                continue
            if not finite(size) or size<=0 or not finite(nll) or nll<0:
                raise ValueError("invalid actual whole-checkpoint bytes or raw-test NLL")
            points.append({"label":record.get("label"),"seed":record.get("seed"),"bits":record.get("bits","dense"),
                           "precision":record.get("precision",run.get("contract",{}).get("configuration",{}).get("precision","unrecorded")),
                           "whole_checkpoint_bytes":size,"raw_test_nll":nll,
                           "whole_checkpoint":record.get("whole_checkpoint"),
                           "scope":"actual serialized whole-model bytes, including archive overhead"})
    return points


def architecture_series(run):
    """Within one model/run, pair each row with its native same-split teacher."""
    if not run.get("run_id","").startswith("architecture-replication-"):
        return None
    contract=run.get("contract",{})
    model=contract.get("configuration",{}).get("model") or run.get("status",{}).get("model")
    if not model:
        raise ValueError("architecture model identity missing")
    records=[r for g in run.get("method_groups",[]) for r in g.get("records",[])]
    references={}
    for row in records:
        if row.get("label")=="teacher" or row.get("specification",{}).get("method")=="teacher":
            split=row.get("split")
            if split in references and references[split]!=row.get("metrics"):
                raise ValueError("conflicting same-split teacher references")
            references[split]=row.get("metrics",{})
    validation=run.get("validation-teacher",{})
    if validation:
        if validation.get("split")!="validation":
            raise ValueError("validation teacher has wrong split")
        expected_hash=contract.get("splits",{}).get("validation",{}).get("hash")
        if not expected_hash or validation.get("token_hash")!=expected_hash:
            raise ValueError("validation teacher split hash mismatch")
        references["validation"]=validation.get("metrics",{})
    groups={}
    unavailable=[]
    for row in records:
        if row.get("specification",{}).get("method")=="teacher":
            continue
        split=row.get("split")
        frontier=split=="validation"
        if split not in ("test","validation"):
            continue
        if frontier and (row.get("rank_divisor") not in (2,3,4) or row.get("independent_replication") is not False):
            raise ValueError("validation frontier must retain declared divisor and non-confirmation role")
        if not frontier and row.get("rank_divisor") is not None:
            raise ValueError("primary test row has frontier divisor")
        reference=references.get(split)
        if not reference:
            unavailable.append({"split":split,"seed":row.get("seed"),"reason":"same-split teacher unavailable"})
            continue
        measured=row.get("metrics",{})
        expected_count=contract.get("splits",{}).get(split,{}).get("tokens")
        if not isinstance(expected_count,int) or reference.get("targets")!=expected_count-1 or measured.get("targets")!=reference.get("targets"):
            raise ValueError("native split target counts mismatch")
        if row.get("teacher_reference_split")!=split or row.get("teacher_reference_targets")!=reference["targets"]:
            raise ValueError("row teacher reference uses wrong split/targets")
        if not finite(measured.get("nll")) or not finite(reference.get("nll")):
            raise ValueError("nonfinite architecture NLL")
        delta=measured["nll"]-reference["nll"]
        if not finite(row.get("teacher_relative_nll_difference")) or not math.isclose(delta,row["teacher_relative_nll_difference"],rel_tol=1e-10,abs_tol=1e-10):
            raise ValueError("recorded teacher-relative NLL disagrees with same-split reference")
        recipes=row.get("target_recipes")
        if not recipes or recipes!=row.get("metadata",{}).get("targets"):
            raise ValueError("exact target recipes missing or inconsistent")
        dense=factors=0
        for recipe in recipes:
            shape=recipe.get("shape")
            rank=recipe.get("rank")
            if not isinstance(shape,list) or len(shape)!=2 or not all(isinstance(n,int) and n>0 for n in shape) or not isinstance(rank,int) or not 0<rank<=min(shape):
                raise ValueError("invalid architecture rank/shape recipe")
            dense+=math.prod(shape)
            factors+=rank*sum(shape)
            if recipe.get("factor_parameters")!=rank*sum(shape):
                raise ValueError("recipe factor count mismatch")
        storage=row.get("storage",{})
        if storage.get("dense_target_parameters")!=dense or storage.get("factor_parameters")!=factors or row.get("metadata",{}).get("factor_parameters")!=factors:
            raise ValueError("actual target factor parameter accounting mismatch")
        spec=row.get("specification",{})
        scope={"split":split,"rank_divisor":row.get("rank_divisor"),"target_recipes":recipes,
               "specification":spec,"factor_parameters":factors,"dense_target_parameters":dense}
        key=json.dumps(scope,sort_keys=True)
        group=groups.setdefault(key,{**scope,"model":model,"retained_target_parameter_fraction":factors/dense,"per_seed":[]})
        seed=row.get("seed")
        if seed in {p["seed"] for p in group["per_seed"]}:
            raise ValueError("duplicate seed in architecture plot scope")
        group["per_seed"].append({"seed":seed,"teacher_relative_nll_difference":delta,
            "teacher_nll":reference["nll"],"measured_nll":measured["nll"],"targets":measured["targets"],
            "checkpoint_sha256":row.get("metadata",{}).get("sha256")})
    for group in groups.values():
        values=[p["teacher_relative_nll_difference"] for p in group["per_seed"]]
        group.update(mean=statistics.mean(values),sample_std=statistics.stdev(values) if len(values)>1 else None,
                     seed_count=len(values),three_seeds={p["seed"] for p in group["per_seed"]}=={11,23,37})
    counts={"test":sum(r.get("split")=="test" for r in records),"validation":sum(r.get("split")=="validation" for r in records)}
    return {"model":model,"run_id":run["run_id"],"state":run.get("state"),
            "partial":run.get("state")!="completed" or counts!={"test":8,"validation":18},"record_counts":counts,
            "primary_test":[g for g in groups.values() if g["split"]=="test"],
            "validation_frontier":[g for g in groups.values() if g["split"]=="validation"],"unavailable":unavailable,
            "interpretation":"Within-model same-split teacher-relative NLL; validation frontier is not independent confirmation. Target factor fraction excludes bias, untargeted weights and serialization overhead."}


def generate_figures(report_directory):
    report_directory=Path(report_directory).resolve()
    report_path=report_directory/"report.json"
    report=json.loads(report_path.read_text(encoding="utf-8"))
    output=report_directory/"figures"
    output.mkdir(exist_ok=False)
    manifest={"generated_utc":datetime.now(timezone.utc).isoformat(),
              "report_input_sha256":hashlib.sha256(report_path.read_bytes()).hexdigest(),
              "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "figures":[],"unavailable":[],"inputs":[],
              "presentation_supersession":report.get("presentation_supersession",{}),
              "interpretation":"Exploratory measured cohorts only. Drift bands are sample seed SD, not confidence intervals; no cross-run pooling."}
    jobs=[]
    for run in report.get("runs",[]):
        if not run.get("run_id","").startswith(("baselines-","practical-","architecture-replication-")):
            continue
        if run["run_id"] in report.get("presentation_supersession",{}).get("superseded",{}):
            # The reporter validated the exact reused prefix and source hashes.
            # Keep the old attempt in provenance, without drawing it twice.
            manifest["inputs"].append({"run_id":run["run_id"],"state":run.get("state"),"path":run.get("path"),
                "metrics":run.get("artifacts",{}).get("metrics.json"),"contract":run.get("artifacts",{}).get("contract.json"),
                "display":"preserved source attempt; drawn through validated continuation",
                "supersession":report["presentation_supersession"]["superseded"][run["run_id"]]})
            continue
        try:
            if run["run_id"].startswith("architecture-replication-"):
                architecture=architecture_series(run)
                manifest["inputs"].append({"run_id":run["run_id"],"state":run.get("state"),"path":run.get("path"),
                    "metrics":run.get("artifacts",{}).get("metrics.json"),"contract":run.get("artifacts",{}).get("contract.json"),
                    "validation_teacher":run.get("artifacts",{}).get("validation-teacher.json"),"figure_data":architecture})
                for field,kind in (("primary_test","architecture-primary"),("validation_frontier","architecture-frontier")):
                    if architecture[field]:
                        jobs.append((kind,run["run_id"],{**architecture,"groups":architecture[field]}))
                    else:
                        manifest["unavailable"].append({"run_id":run["run_id"],"kind":kind,"reason":"no measured rows with valid same-split teacher/parameter accounting"})
                continue
            kind="drift" if run["run_id"].startswith("baselines-") else "storage"
            data=drift_series(run) if kind=="drift" else storage_points(run)
            if not data:
                manifest["unavailable"].append({"run_id":run["run_id"],"kind":kind,
                    "reason":"cohort incomplete or required measured drift/whole-checkpoint NLL data missing"})
                continue
            manifest["inputs"].append({"run_id":run["run_id"],"state":run.get("state"),"path":run.get("path"),
                "metrics":run.get("artifacts",{}).get("metrics.json"),"contract":run.get("artifacts",{}).get("contract.json"),
                "figure_data":data})
            jobs.append((kind,run["run_id"],data))
        except Exception as exc:
            manifest["unavailable"].append({"run_id":run.get("run_id"),"reason":str(exc)})
    if not any(kind=="drift" for kind,_,_ in jobs):
        manifest["unavailable"].append({"kind":"drift","reason":"no complete Gemma C cohort with measured per-block drift"})
    if not any(kind=="storage" for kind,_,_ in jobs):
        manifest["unavailable"].append({"kind":"storage","reason":"no complete Stage F with actual whole-model bytes and raw native-test NLL"})
    if jobs:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            manifest["unavailable"].append({"kind":"rendering","reason":"existing matplotlib unavailable: "+str(exc)})
            jobs=[]
    for kind,run_id,data in jobs:
        fig=None
        try:
            fig,ax=plt.subplots(figsize=(10,6),layout="constrained")
            if kind=="drift":
                for series in data:
                    x=list(range(len(series["mean"])))
                    label=method_label(series["specification"])
                    line,=ax.plot(x,series["mean"],linewidth=2,label=label+f" ({len(series['per_seed'])} seed(s))")
                    for row in series["per_seed"]:
                        ax.plot(x,row["values"],color=line.get_color(),alpha=.25,linewidth=.8,linestyle=":")
                    if len(series["per_seed"])>1:
                        ax.fill_between(x,[m-s for m,s in zip(series["mean"],series["sample_std"])],
                                        [m+s for m,s in zip(series["mean"],series["sample_std"])],color=line.get_color(),alpha=.12)
                ax.set(xlabel="Transformer block index",ylabel="Normalized residual squared error",
                       title="Measured drift versus depth — "+run_id)
                caption="Gemma Stage C residual drift: individual seeds (dotted), mean (solid), sample seed SD (shaded). Single retained cohort; strict FP16 readiness remains failed."
            elif kind=="storage":
                groups={}
                for point in data:
                    groups.setdefault((point["label"],point["bits"],point["precision"]),[]).append(point)
                for (label,bits,precision),points in groups.items():
                    ax.scatter([p["whole_checkpoint_bytes"]/(1024**2) for p in points],
                               [p["raw_test_nll"] for p in points],label=f"{label}, {bits} bits / {precision}",s=45)
                    for point in points:
                        if point["seed"] is not None:
                            ax.annotate(str(point["seed"]),(point["whole_checkpoint_bytes"]/(1024**2),point["raw_test_nll"]),
                                        textcoords="offset points",xytext=(4,4),fontsize=7)
                ax.set(xlabel="Actual serialized whole-model checkpoint (MiB)",ylabel="Raw native-test NLL",
                       title="Measured quality versus storage — "+run_id)
                caption="Gemma Stage F raw native-test NLL versus actual whole-model checkpoint bytes; seed numbers label points. Factor-only full-rank controls excluded; no speed or lossless-compression claim."
            else:
                status=("PARTIAL cohort: "+str(data["state"])+f"; records test {data['record_counts']['test']}/8, validation {data['record_counts']['validation']}/18") if data["partial"] else "completed exploratory cohort"
                groups=data["groups"]
                ax.axhline(0,color="gray",linewidth=.8,linestyle="--",label="Same-split dense teacher")
                if kind=="architecture-primary":
                    groups=sorted(groups,key=lambda g:(method_label(g["specification"]),g["factor_parameters"]))
                    ticks=[]
                    for index,group in enumerate(groups):
                        ax.errorbar(index,group["mean"],yerr=group["sample_std"],fmt="o",capsize=4,color="black")
                        for offset,point in enumerate(group["per_seed"]):
                            dx=(offset-(len(group["per_seed"])-1)/2)*.07
                            ax.scatter(index+dx,point["teacher_relative_nll_difference"],s=25,alpha=.6)
                            ax.annotate(str(point["seed"]),(index+dx,point["teacher_relative_nll_difference"]),
                                        textcoords="offset points",xytext=(3,3),fontsize=7)
                        ticks.append(method_label(group["specification"])+f"\nn={group['seed_count']}; factors={group['factor_parameters']:,}\nretained target fraction={group['retained_target_parameter_fraction']:.4f}")
                    ax.set_xticks(range(len(groups)),ticks,fontsize=8)
                    ax.set(xlabel="Primary test methods at recorded target parameter budgets",ylabel="Native-test NLL minus native-test teacher NLL",
                           title=data["model"]+" — primary test quality\n"+status)
                    caption=f"{data['model']}: primary native-test teacher-relative NLL for {status}; individual paired seed points, mean and sample seed SD when at least two seeds exist. Actual target factor counts/fractions shown. Strict promotion remains disabled."
                else:
                    methods={}
                    for group in groups:
                        methods.setdefault(json.dumps(group["specification"],sort_keys=True),[]).append(group)
                    for key,points in methods.items():
                        points=sorted(points,key=lambda g:g["retained_target_parameter_fraction"])
                        line,=ax.plot([g["retained_target_parameter_fraction"] for g in points],[g["mean"] for g in points],
                                      marker="o",label=method_label(json.loads(key)))
                        for group in points:
                            x=group["retained_target_parameter_fraction"]
                            if group["sample_std"] is not None:
                                ax.errorbar(x,group["mean"],yerr=group["sample_std"],color=line.get_color(),capsize=4)
                            for point in group["per_seed"]:
                                ax.scatter(x,point["teacher_relative_nll_difference"],s=18,alpha=.35,color=line.get_color())
                            ax.annotate(f"divisor {group['rank_divisor']}; n={group['seed_count']}\n{group['factor_parameters']:,} factors",
                                        (x,group["mean"]),textcoords="offset points",xytext=(5,6),fontsize=7)
                    ax.set(xlabel="Retained target weight parameter fraction: rank*(out+in)/(out*in)",
                           ylabel="Validation NLL minus validation teacher NLL",
                           title=data["model"]+" — validation-only rank frontier\n"+status)
                    caption=f"{data['model']}: validation-only rank frontier for {status}. Each point aggregates only one exact rank recipe and available paired seeds; error bars are sample seed SD. X is actual retained target weight parameter fraction, not whole-model storage or a rank compression factor. Validation is not independent confirmation."
            ax.grid(alpha=.2)
            ax.legend(fontsize=7,loc="upper left",bbox_to_anchor=(1,1))
            stem=run_id+"-"+kind
            for extension in ("png","svg"):
                fig.savefig(output/(stem+"."+extension),dpi=180)
            manifest["figures"].append({"run_id":run_id,"kind":kind,"png":"figures/"+stem+".png",
                "svg":"figures/"+stem+".svg","caption":caption,
                "png_sha256":hashlib.sha256((output/(stem+".png")).read_bytes()).hexdigest(),
                "svg_sha256":hashlib.sha256((output/(stem+".svg")).read_bytes()).hexdigest()})
        except Exception as exc:
            manifest["unavailable"].append({"run_id":run_id,"kind":kind,"reason":"rendering error: "+str(exc)})
        finally:
            if fig is not None:
                plt.close(fig)
    (output/"provenance.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_directory")
    print(json.dumps(generate_figures(parser.parse_args().report_directory),indent=2))
