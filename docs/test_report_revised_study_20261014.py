"""CPU/stdlib tests: rtk python -m unittest discover -s docs -p test_report_revised_study_20261014.py"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import importlib.util
import sys
from unittest.mock import patch

try:
    from .report_revised_study_20261014 import bootstrap, main, summarize, budget_audit, export_documents, apply_interruption_accounting, digest, manuscript_text, practical_same_bit_costs, presentation_supersession
    from .plot_revised_study_20261014 import drift_series, storage_points, generate_figures, method_label, architecture_series
except ImportError:
    from report_revised_study_20261014 import bootstrap, main, summarize, budget_audit, export_documents, apply_interruption_accounting, digest, manuscript_text, practical_same_bit_costs, presentation_supersession
    from plot_revised_study_20261014 import drift_series, storage_points, generate_figures, method_label, architecture_series


class ReportTest(unittest.TestCase):
    def test_display_supersession_requires_unique_verified_exact_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"architecture-replication-source"
            successor=root/"architecture-replication-recovery"
            source.mkdir(); successor.mkdir()
            rows=[{"seed":11,"metrics":{"nll":2.}}]
            for name in ("config.json","status.json","contract.json"):
                (source/name).write_text("{}")
            (source/"metrics.json").write_text(json.dumps(rows))
            (successor/"metrics.json").write_text(json.dumps(rows+[{"new":True}]))
            contract={"configuration":{"model":"test","model_revision":"m","dataset_revision":"d","tokenizer_revision":"t"},
                      "splits":{"test":{"hash":"test"}},"targets":[{"name":"out","rank":1}],"baseline":{"method":"whitening"}}
            old={"run_id":source.name,"path":str(source),"state":"interrupted","contract":contract}
            recovery={"source_run":str(source),"reused_records":1,"independent_replication":False,
                      "source_artifacts":{name:digest(source/name) for name in ("config.json","status.json","contract.json","metrics.json")}}
            new={"run_id":successor.name,"path":str(successor),"state":"running","contract":{**copy.deepcopy(contract),"recovery":recovery}}
            valid=presentation_supersession(root,[old,new])
            self.assertEqual(valid["superseded"][old["run_id"]]["successor"],new["run_id"])
            self.assertEqual(valid["superseded"][old["run_id"]]["successor_state"],"running")
            self.assertEqual(valid["warnings"],[])
            for kind in ("pin","split","hash","independent","count","branch","prefix"):
                changed=copy.deepcopy(new)
                candidates=[old,changed]
                if kind=="pin": changed["contract"]["configuration"]["model_revision"]="wrong"
                if kind=="split": changed["contract"]["splits"]={"test":{"hash":"wrong"}}
                if kind=="hash": changed["contract"]["recovery"]["source_artifacts"]["metrics.json"]="wrong"
                if kind=="independent": changed["contract"]["recovery"]["independent_replication"]=True
                if kind=="count": changed["contract"]["recovery"]["reused_records"]=0
                if kind=="branch": candidates.append({**copy.deepcopy(changed),"run_id":"architecture-replication-branch"})
                if kind=="prefix": (successor/"metrics.json").write_text(json.dumps([{"changed":True}]))
                with self.subTest(kind=kind):
                    result=presentation_supersession(root,candidates)
                    self.assertEqual(result["superseded"],{})
                    self.assertTrue(result["warnings"])

    def test_practical_cost_pairs_require_same_bit_device_placement_and_workload(self):
        timing={"mode":"decode","prefill_tokens":128,"decode_tokens":128,"batch_size":1,"warmups":10,
                "measurements":30,"use_cache":True,"median_seconds":2.,"p95_seconds":4.,
                "peak_allocated_bytes":100,"peak_reserved_bytes":200}
        baseline={"label":"baseline","specification":{"method":"whitening"},"seed":11,"bits":8,
                  "factor_checkpoint_bytes":1000,"whole_checkpoint":{"bytes":2000},"timing":[timing]}
        solver=copy.deepcopy(baseline)
        solver.update(label="solver",specification={"method":"anchored","beta":.03},factor_checkpoint_bytes=900)
        solver["timing"][0]["median_seconds"]=3.
        contract={"configuration":{"device":"cuda","precision":"float16"},"environment":{"gpu":"test GPU"}}
        data=practical_same_bit_costs([baseline,solver],contract)
        pair=data["comparisons"][0]
        self.assertEqual(len(data["comparisons"]),1)
        self.assertEqual(pair["ratios"]["median_latency"],1.5)
        self.assertEqual(pair["ratios"]["factor_bytes"],.9)
        self.assertEqual(pair["ratios"]["whole_bytes"],1.)
        self.assertFalse(pair["hardware_gate"])
        for changes in ({"bits":16},{"seed":23},{"device":"cpu"},{"placement":{"cpu":"offloaded"}},
                        {"gpu":"other GPU"},{"compute_precision":"bfloat16"}):
            with self.subTest(changes=changes):
                self.assertEqual(practical_same_bit_costs([baseline,{**solver,**changes}],contract)["comparisons"],[])
        for field,value in (("mode","prefill"),("prefill_tokens",512),("decode_tokens",1),("warmups",0),("measurements",20),("batch_size",2),("use_cache",False)):
            other=copy.deepcopy(solver)
            other["timing"][0][field]=value
            with self.subTest(field=field):
                self.assertEqual(practical_same_bit_costs([baseline,other],contract)["comparisons"],[])
        self.assertEqual(practical_same_bit_costs([baseline,solver,solver],contract)["comparisons"],[])
        self.assertEqual(practical_same_bit_costs([baseline,solver],{"configuration":contract["configuration"]})["comparisons"],[])

    def test_scoped_negative_prose_requires_matched_primary_three_seed_evidence(self):
        comparison={"scope":{"split":"test"},"candidate":{"method":"anchored","beta":.03},
                    "baseline":{"method":"whitening"},"nll_and_drift_worse_all_three":True,
                    "per_seed":[{"seed":s,"targets_match":True,"actual_factor_parameters_match":True} for s in (11,23,37)]}
        run={"run_id":"architecture-replication-test","state":"interrupted","contract":{"configuration":{"model":"Qwen/model"}},
             "comparisons":[comparison],"method_groups":[]}
        report={"generated_utc":"test","orchestration":{"state":"stopped"},"mode":"test","runs":[run],
                "budget":{"stages":{},"interpretation":"test fixture"},"decision":"inconclusive"}
        docs=Path(__file__).resolve().parent
        self.assertIn("Scoped finding",manuscript_text(report,docs))
        for change in ("validation","unmatched","mixed"):
            bad=copy.deepcopy(report)
            value=bad["runs"][0]["comparisons"][0]
            if change=="validation":
                value["scope"]["split"]="validation"
            elif change=="unmatched":
                value["per_seed"][0]["actual_factor_parameters_match"]=False
            else:
                value["nll_and_drift_worse_all_three"]=False
            self.assertNotIn("Scoped finding",manuscript_text(bad,docs))

    def test_interruption_sidecar_identity_hash_and_charge_are_bound_to_immutable_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run=root/"architecture-replication-interrupted"
            run.mkdir()
            for name in ("status.json","metrics.json","contract.json","config.json"):
                (run/name).write_text("{}")
            record={"run_id":run.name,"path":str(run),"state":"running","status":{"state":"running","elapsed_seconds":1},
                    "contract":{"configuration":{"model":"Qwen/Qwen3.5-0.8B"}}}
            entry={"run_id":run.name,"run":str(run),"stage":"replication","model":"Qwen/Qwen3.5-0.8B",
                   "state":"interrupted by reboot","started_epoch":10.,"new_boot_epoch":20.,"charged_seconds":10.,
                   "basis":"launch-to-boot wall interval","independent_replication":False,
                   "source_artifacts":{name:digest(run/name) for name in ("status.json","metrics.json","contract.json","config.json")}}
            path=root/"interruption.json"
            def write(value):
                path.write_text(json.dumps({"schema":"researchcompression-interruption-accounting-v1","runs":[value]}))
            write(entry)
            rows=[copy.deepcopy(record)]
            apply_interruption_accounting(root,rows,path)
            audit=budget_audit(root,rows)["stages"]["architecture"]
            self.assertEqual(audit["recorded_charged_seconds"],10.)
            self.assertEqual(audit["unknown_cost_runs"],[])
            self.assertEqual(rows[0]["state"],"interrupted")
            self.assertEqual(rows[0]["observed_state"],"running")
            self.assertEqual(rows[0]["status"],record["status"])
            self.assertEqual((run/"status.json").read_text(),"{}")
            corruptions=[{"run_id":"wrong"},{"run":str(root/"wrong")},{"charged_seconds":-1},
                         {"charged_seconds":9.},{"charged_seconds":float("inf")},
                         {"source_artifacts":{**entry["source_artifacts"],"status.json":"wrong"}}]
            for corruption in corruptions:
                with self.subTest(corruption=corruption):
                    write({**entry,**corruption})
                    unchanged=[copy.deepcopy(record)]
                    with self.assertRaises(ValueError):
                        apply_interruption_accounting(root,unchanged,path)
                    self.assertEqual(unchanged,[record])

    def test_export_failure_is_explicit_preserves_other_format_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def latex(directory):
                path=directory/"manuscript.tex"
                with path.open("x") as stream:
                    stream.write("standalone test fixture")
                return path
            def missing_pdf_dependency(directory):
                raise ModuleNotFoundError("reportlab unavailable in test fixture")
            status=export_documents(root,exporters={"latex":latex,"pdf":missing_pdf_dependency})
            self.assertEqual(status["state"],"failed")
            self.assertEqual(status["exports"]["latex"]["state"],"prepared")
            self.assertIn("reportlab",status["exports"]["pdf"]["error"])
            self.assertFalse(status["latex_compilation_verified"])
            before=(root/"export-status.json").read_bytes()
            with self.assertRaises(FileExistsError):
                export_documents(root,exporters={"latex":latex,"pdf":missing_pdf_dependency})
            self.assertEqual((root/"export-status.json").read_bytes(),before)

    def test_main_propagates_export_failure_without_altering_orchestration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            orchestration=root/"orchestration"
            orchestration.mkdir()
            status=json.dumps({"state":"stopped"})
            (orchestration/"status.json").write_text(status)
            failure={"state":"failed","interpretation":"injected exporter dependency failure",
                     "exports":{"pdf":{"state":"failed","error":"ModuleNotFoundError: reportlab"}}}
            with patch.object(sys.modules[main.__module__],"export_documents",return_value=failure):
                self.assertEqual(main(["--root",tmp,"--orchestration",str(orchestration)]),1)
            report=next(root.glob("revised-report-*"))
            self.assertIn("ModuleNotFoundError: reportlab",(report/"report.md").read_text())
            self.assertEqual((orchestration/"status.json").read_text(),status)

    def architecture_fixture(self,model="Qwen/model",teacher_shift=0):
        teacher={"label":"teacher","specification":{"method":"teacher"},"seed":11,"split":"test", "metrics":{"nll":2+teacher_shift,"targets":10}}
        records=[teacher]
        for split,divisor,rank,base,targets in [("test",None,2,2.2,10),("validation",2,2,4.6,8),("validation",4,1,4.8,8)]:
            for seed,offset in [(11,0),(23,.2)]:
                recipe={"name":"attention.out","shape":[4,4],"rank":rank,"factor_parameters":rank*8}
                reference=(2 if split=="test" else 4)+teacher_shift
                nll=base+offset+teacher_shift
                records.append({"label":"solver","specification":{"method":"anchored","beta":.03},"seed":seed,
                    "split":split,"rank":rank,"rank_divisor":divisor,"target_recipes":[recipe],
                    "independent_replication":False if divisor else True,
                    "metrics":{"nll":nll,"targets":targets},"teacher_reference_split":split,"teacher_reference_targets":targets,
                    "teacher_relative_nll_difference":nll-reference,
                    "metadata":{"targets":[recipe],"factor_parameters":rank*8,"sha256":f"{split}-{rank}-{seed}"},
                    "storage":{"dense_target_parameters":16,"factor_parameters":rank*8}})
        return {"run_id":"architecture-replication-test","state":"running","contract":{
            "configuration":{"model":model},"splits":{"test":{"tokens":11,"hash":"test-hash"},"validation":{"tokens":9,"hash":"val-hash"}}},
            "validation-teacher":{"split":"validation","token_hash":"val-hash","metrics":{"nll":4+teacher_shift,"targets":8}},
            "method_groups":summarize(records)}

    def test_architecture_plot_split_rank_model_and_partial_scope(self):
        run=self.architecture_fixture()
        data=architecture_series(run)
        self.assertTrue(data["partial"])
        self.assertEqual(len(data["primary_test"]),1)
        self.assertEqual(len(data["validation_frontier"]),2)
        self.assertAlmostEqual(data["primary_test"][0]["mean"],.3)
        self.assertEqual(sorted(g["retained_target_parameter_fraction"] for g in data["validation_frontier"]),[.5,1.])
        self.assertTrue(all(g["seed_count"]==2 for g in data["validation_frontier"]))
        self.assertTrue(all(g["split"]=="validation" for g in data["validation_frontier"]))
        other=architecture_series(self.architecture_fixture("LFM/model",10))
        self.assertNotEqual(other["model"],data["model"])
        self.assertAlmostEqual(other["primary_test"][0]["mean"],data["primary_test"][0]["mean"])
        corrupted=copy.deepcopy(run)
        corrupted["validation-teacher"]["metrics"]["targets"]=10
        with self.assertRaisesRegex(ValueError,"target counts"):
            architecture_series(corrupted)
        corrupted=copy.deepcopy(run)
        corrupted["validation-teacher"]["token_hash"]="test-hash"
        with self.assertRaisesRegex(ValueError,"split hash"):
            architecture_series(corrupted)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"),"existing matplotlib unavailable")
    def test_plot_preserves_superseded_source_provenance_without_duplicate_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=self.architecture_fixture()
            source["state"]="interrupted"
            successor=copy.deepcopy(source)
            successor.update(run_id="architecture-replication-recovery",state="running")
            evidence={"successor":successor["run_id"],"source_state":"interrupted","successor_state":"running",
                      "source_artifacts":{"metrics.json":"verified-source-fixture"},"reason":"validated exact prefix"}
            selection={"superseded":{source["run_id"]:evidence},"warnings":[]}
            (root/"report.json").write_text(json.dumps({"runs":[source,successor],"presentation_supersession":selection}))
            manifest=generate_figures(root)
            self.assertEqual(len(manifest["figures"]),2)
            self.assertTrue(all(f["run_id"]==successor["run_id"] for f in manifest["figures"]))
            self.assertEqual(manifest["presentation_supersession"],selection)
            preserved=next(r for r in manifest["inputs"] if r["run_id"]==source["run_id"])
            self.assertEqual(preserved["state"],"interrupted")
            self.assertEqual(preserved["supersession"]["source_artifacts"],evidence["source_artifacts"])

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"),"existing matplotlib unavailable")
    def test_architecture_primary_and_frontier_render_independent_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            qwen=self.architecture_fixture()
            lfm=self.architecture_fixture("LFM/model",10)
            lfm["run_id"]="architecture-replication-other"
            lfm["state"]="completed"
            (root/"report.json").write_text(json.dumps({"runs":[qwen,lfm]}))
            manifest=generate_figures(root)
            self.assertEqual(len(manifest["figures"]),4)
            self.assertEqual({f["kind"] for f in manifest["figures"]},{"architecture-primary","architecture-frontier"})
            self.assertTrue(any("PARTIAL" in f["caption"] for f in manifest["figures"] if f["run_id"]==qwen["run_id"]))
            self.assertTrue(all((root/f["png"]).is_file() and (root/f["svg"]).is_file() for f in manifest["figures"]))
            self.assertEqual(len(manifest["inputs"]),2)

    def test_legacy_runtime_matches_original_log_fallback_and_invalid_stays_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            old=root/"readiness-legacy"
            old.mkdir()
            (old/"stdout.log").write_text("2026-10-13T10:00:00+00:00 started\n2026-10-13T10:01:18.336+00:00 failed\n")
            row={"run_id":old.name,"path":str(old),"state":"failed","status":{"state":"failed"}}
            audit=budget_audit(root,[row])["stages"]["readiness"]
            self.assertAlmostEqual(audit["recorded_charged_seconds"],138.336)
            self.assertEqual(audit["unknown_cost_runs"],[])
            self.assertIn("matching original spent_seconds",audit["attempts"][0]["basis"])
            for log in ("", "not-an-ISO-time failed\n", "2026-10-13T10:00:00+00:00 start\ninvalid end\n"):
                (old/"stdout.log").write_text(log)
                invalid=budget_audit(root,[row])["stages"]["readiness"]
                self.assertEqual(invalid["unknown_cost_runs"],[old.name])
                self.assertIsNone(invalid["attempts"][0]["seconds"])

    def test_readable_method_labels(self):
        self.assertEqual(method_label({"method":"svd"}),"SVD")
        self.assertEqual(method_label({"method":"asvd"}),"ASVD")
        self.assertEqual(method_label({"method":"whitening"}),"Whitening")
        self.assertEqual(method_label({"method":"anchored","beta":.03}),"Anchor (beta=0.03)")
        self.assertEqual(method_label({"method":"legacy_ridge","beta":.1,"ridge_fraction":.01}),"Extra ridge (beta=0.1, ridge=0.01)")
    def drift_fixture(self):
        return {"run_id":"baselines-test","state":"completed","method_groups":[{
            "specification":{"method":"whitening"},"scope":{},"records":[
                {"seed":seed,"metrics":{"residual_by_block":[{"normalized_squared_error":x} for x in values]}}
                for seed,values in [(11,[1.,2.]),(23,[3.,4.])]]}]}

    def test_plot_stats_finite_checks_and_actual_storage_scope(self):
        run=self.drift_fixture()
        series=drift_series(run)[0]
        self.assertEqual(series["mean"],[2,3])
        self.assertAlmostEqual(series["sample_std"][0],2**.5)
        run["method_groups"][0]["records"][0]["metrics"]["residual_by_block"][0]["normalized_squared_error"]=float("nan")
        with self.assertRaises(ValueError):
            drift_series(run)
        self.assertEqual(storage_points({"run_id":"precision-readiness-test","state":"completed","method_groups":[]}),[])
        self.assertEqual(drift_series({**self.drift_fixture(),"state":"incomplete"}),[])

    def test_plot_no_data_is_explicit_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/"report.json").write_text(json.dumps({"runs":[]}))
            manifest=generate_figures(path)
            self.assertFalse(manifest["figures"])
            self.assertEqual({m["kind"] for m in manifest["unavailable"]},{"drift","storage"})
            self.assertFalse(list((path/"figures").glob("*.png")))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"),"existing matplotlib unavailable")
    def test_png_svg_rendering_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            run=self.drift_fixture()
            run["artifacts"]={"metrics.json":{"sha256":"fixture-only"}}
            (path/"report.json").write_text(json.dumps({"runs":[run]}))
            manifest=generate_figures(path)
            self.assertEqual(len(manifest["figures"]),1)
            figure=manifest["figures"][0]
            self.assertTrue((path/figure["png"]).read_bytes().startswith(b"\x89PNG"))
            self.assertIn("<svg",(path/figure["svg"]).read_text())
            self.assertEqual(manifest["inputs"][0]["metrics"]["sha256"],"fixture-only")

    def test_fixed_block_uncertainty_does_not_bootstrap_partial_tail(self):
        a=[{"nll_sum":4096,"targets":4096}]*2+[{"nll_sum":100,"targets":1}]
        b=[{"nll_sum":8192,"targets":4096}]*2+[{"nll_sum":1,"targets":1}]
        result=bootstrap(a,b)
        self.assertEqual(result["confidence_interval_95"],[-1,-1])
        self.assertEqual(result["excluded_partial_targets"],1)
        self.assertEqual(result["resamples"],2000)
        self.assertFalse(bootstrap(a,b[:-1])["available"])

    def test_alias_conflict_not_extra_seed(self):
        row={"seed":11,"specification":{"method":"whitening"},"metadata":{"sha256":"same","factor_parameters":12},"metrics":{"nll":1}}
        alias={**copy.deepcopy(row),"specification":{"method":"anchored","beta":0}}
        clean=summarize([row,alias])
        self.assertEqual(clean[0]["seeds"],[11])
        alias["metrics"]["nll"]=2
        conflict=summarize([row,alias])
        self.assertEqual(conflict[0]["seeds"],[])
        self.assertEqual(len(conflict[0]["conflicts"]),1)

    def test_validation_frontier_never_pools_rank_or_test(self):
        rows=[]
        for split,divisor in [("test",None),("validation",2),("validation",3),("validation",4)]:
            for seed in (11,23,37):
                row={"seed":seed,"specification":{"method":"anchored","beta":.03},"split":split,
                     "metadata":{"sha256":f"{split}-{divisor}-{seed}"},"metrics":{"nll":1}}
                if divisor:
                    row.update(rank_divisor=divisor,recipes=[{"name":"out","rank":1024//divisor}])
                rows.append(row)
        groups=summarize(rows)
        self.assertEqual(len(groups),4)
        self.assertTrue(all(g["three_seeds"] for g in groups))

    def test_terminal_guard_and_partial_artifacts_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            orchestration=root/"revised-orchestration-2"
            orchestration.mkdir()
            (orchestration/"status.json").write_text(json.dumps({"state":"running precision"}))
            with self.assertRaises(RuntimeError):
                main(["--root",tmp,"--orchestration",str(orchestration)])
            (orchestration/"status.json").write_text(json.dumps({"state":"completed with incomplete cohorts","error":"cap exhausted"}))
            precision=root/"precision-readiness-partial"
            precision.mkdir()
            status=json.dumps({"state":"failed","elapsed_seconds":5,"completed_variants":["teacher_bf16"]})
            (precision/"status.json").write_text(status)
            (precision/"metrics.json").write_text(json.dumps({"teacher_bf16":{"nll":2,"identity_passed":True}}))
            bad=root/"architecture-replication-unreadable"
            bad.mkdir()
            (bad/"status.json").write_text("{malformed")
            self.assertEqual(main(["--root",tmp,"--orchestration",str(orchestration)]),0)
            report= json.loads(next(root.glob("revised-report-*/report.json")).read_text())
            self.assertTrue(report["promotion_disabled"])
            self.assertEqual(report["hypothesis_status"],"inconclusive")
            self.assertEqual(len(report["read_errors"]),1)
            self.assertIn(bad.name,report["budget"]["stages"]["architecture"]["unknown_cost_runs"])
            self.assertFalse(report["budget"]["exact_24_gpu_hour_compliance_established"])
            self.assertEqual((precision/"status.json").read_text(),status)
            output=next(root.glob("revised-report-*"))
            self.assertTrue((output/"manuscript.md").exists())
            self.assertTrue((output/"paper_protocol_v2.bib").exists())
            self.assertIn("completed with incomplete cohorts",(output/"manuscript.md").read_text())
            self.assertNotIn("will be transcribed",(output/"manuscript.md").read_text())
            self.assertTrue((output/"figures/provenance.json").exists())
            self.assertTrue((output/"manuscript.tex").exists())
            self.assertTrue((output/"manuscript.pdf").read_bytes().startswith(b"%PDF-"))
            from pypdf import PdfReader
            self.assertGreater(len(PdfReader(output/"manuscript.pdf").pages),0)
            exports=json.loads((output/"export-status.json").read_text())
            self.assertEqual(exports["state"],"prepared")
            self.assertFalse(exports["latex_compilation_verified"])
            self.assertIn("not compiled from LaTeX",exports["exports"]["pdf"]["format"])


if __name__=="__main__":
    unittest.main()
