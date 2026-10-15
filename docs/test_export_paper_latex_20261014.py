"""CPU-only regression tests; synthetic fixtures never enter experiment results."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).with_name("export_paper_latex_20261014.py")
SPEC = importlib.util.spec_from_file_location("latex_export", MODULE)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class ExportTests(unittest.TestCase):
    def test_escape(self):
        self.assertEqual(exporter.escape("a_b & 5% # {x} $"),r"a\_b \& 5\% \# \{x\} \$")
        self.assertEqual(exporter.inline(r"Error $\|W-A\|_F^2$ and `x_y`."),r"Error $\|W-A\|_F^2$ and \texttt{x\_y}.")

    def test_tables_split_without_omitting_columns(self):
        table = "| "+" | ".join("col"+str(i) for i in range(12))+" |\n"
        table += "| "+" | ".join("---" for _ in range(12))+" |\n"
        table += "| "+" | ".join("value"+str(i) for i in range(12))+" |\n"
        result = exporter.markdown_to_latex(table)
        self.assertEqual(result.count(r"\begin{longtable}"),4)
        for i in range(12): self.assertIn("value"+str(i),result)

    def test_wide_table_unique_row_ids_in_every_panel(self):
        source = "| Run | Variant | NLL | PPL | Error | Probe gate | NLL gate |\n|---|---|---|---|---|---|---|\n| same | first | 1 | 2 | 3 | passA | failA |\n| same | second | 4 | 5 | 6 | passB | failB |"
        result = exporter.markdown_to_latex(source)
        self.assertEqual(result.count("R001"),2)
        self.assertEqual(result.count("R002"),2)
        self.assertIn("R001 & same & first",result)
        self.assertIn("R001 & same & 3 & passA & failA",result)
        self.assertIn("R002 & same & 6 & passB & failB",result)

    def test_nested_bibliography(self):
        entries = exporter.bibliography_entries('@misc{k,title={{ASVD}: A {Nested} Title},year={2023}}')
        self.assertEqual(entries[0][1]["title"],"ASVD: A Nested Title")
        actual = MODULE.with_name("paper_protocol_v2.bib").read_text(encoding="utf-8")
        self.assertEqual(len(exporter.bibliography_entries(actual)),22)

    def test_scope_separation(self):
        data = {"inputs":[{"run_id":"architecture-replication-fixture","figure_data":{
            "model":"fixture_model","primary_test":[{"specification":{"method":"anchor"},"retained_target_parameter_fraction":.7,"mean":.2}],
            "validation_frontier":[{"specification":{"method":"anchor"},"retained_target_parameter_fraction":.5,"mean":.4}]}}]}
        result = exporter.embedded_figures(data)
        self.assertIn("Primary test",result)
        self.assertIn("Validation frontier (not confirmation)",result)
        self.assertNotIn("PPL ranking",result)
        self.assertEqual(result.count(r"\begin{picture}"),2)
        self.assertIn("Unmeasured/unavailable: storage",result)
        with self.assertRaises(ValueError): exporter.picture([("bad",[(1,float("nan"))])],"x","y")

    def test_export_valid_name_standalone_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="latex-export-test-") as temp:
            directory = Path(temp)
            (directory/"report.json").write_text(json.dumps({"promotion_disabled":True,"generated_utc":"fixture"}))
            (directory/"manuscript.md").write_text("# Fixture\n\n## Scope\nExploratory incomplete. $A=UV$.\n")
            (directory/"paper_protocol_v2.bib").write_bytes(MODULE.with_name("paper_protocol_v2.bib").read_bytes())
            output = exporter.export_report(directory)
            self.assertEqual(output.name,"manuscript.tex")
            content = output.read_text()
            self.assertEqual(content.count(r"\bibitem{"),22)
            self.assertNotIn(r"\includegraphics",content)
            self.assertNotIn(r"\bibliography{",content)
            self.assertIn("Exploratory incomplete",content)
            with self.assertRaises(FileExistsError): exporter.export_report(directory)
            self.assertEqual(output.read_text(),content)


if __name__ == "__main__": unittest.main()
