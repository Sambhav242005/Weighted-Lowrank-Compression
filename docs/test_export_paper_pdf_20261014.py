"""PDF export smoke checks on synthetic fixtures outside research results."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).with_name("export_paper_pdf_20261014.py")
SPEC = importlib.util.spec_from_file_location("pdf_review",SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PdfTests(unittest.TestCase):
    def test_markup_escapes_xml(self):
        self.assertEqual(MODULE.text_markup("x < y & z"),"x &lt; y &amp; z")
        self.assertEqual(MODULE.plain_math(r"H=XX^T+\beta nI; W\in\mathbb{R}^{o\times i}"),"H=XX^T+beta nI; W in R^{o x  i}")

    def test_export_readable_paginated_scope_no_overwrite(self):
        import fitz
        from pypdf import PdfReader
        with tempfile.TemporaryDirectory(prefix="pdf-export-fixture-") as temp:
            directory = Path(temp)
            (directory/"report.json").write_text(json.dumps({"promotion_disabled":True}))
            rows = ["# Fixture","## Incomplete fixture","Validation frontier is not confirmation.",
                    "| Attempt | State | NLL | Drift | Scope | Seed |","|---|---|---|---|---|---|"]
            rows += [f"| fixture_{i} | incomplete | unmeasured | unmeasured | validation only | 11 |" for i in range(80)]
            (directory/"manuscript.md").write_text("\n".join(rows))
            (directory/"paper_protocol_v2.bib").write_bytes(SOURCE.with_name("paper_protocol_v2.bib").read_bytes())
            output = MODULE.export_pdf(directory)
            reader = PdfReader(output)
            self.assertGreater(len(reader.pages),2)
            text = "\n".join(p.extract_text() for p in reader.pages)
            self.assertIn("fixture_79",text)
            self.assertIn("not confirmation",text)
            self.assertIn("Verified references",text)
            self.assertIn("clark2018arc",text)
            document = fitz.open(output)
            for page in document:
                for block in page.get_text("blocks"):
                    self.assertGreaterEqual(block[0],0)
                    self.assertLessEqual(block[2],page.rect.width+.5)
                    self.assertGreaterEqual(block[1],0)
                    self.assertLessEqual(block[3],page.rect.height+.5)
            document.close()
            reader.close()
            with self.assertRaises(FileExistsError): MODULE.export_pdf(directory)

    def test_wide_panels_repeat_identifiable_rows(self):
        from pypdf import PdfReader
        with tempfile.TemporaryDirectory(prefix="pdf-rowid-fixture-") as temp:
            directory = Path(temp)
            (directory/"report.json").write_text(json.dumps({"promotion_disabled":True}))
            (directory/"manuscript.md").write_text("# Fixture\n| Run | Variant | NLL | PPL | Error | Probe gate | NLL gate |\n|---|---|---|---|---|---|---|\n| same | first | 1 | 2 | 3 | passA | failA |\n| same | second | 4 | 5 | 6 | passB | failB |")
            (directory/"paper_protocol_v2.bib").write_bytes(SOURCE.with_name("paper_protocol_v2.bib").read_bytes())
            reader = PdfReader(MODULE.export_pdf(directory))
            text = "\n".join(p.extract_text() for p in reader.pages)
            reader.close()
            self.assertEqual(text.count("R001"),2)
            self.assertEqual(text.count("R002"),2)
            panels = text.split("match by Row ID.")[1:]
            self.assertIn("R001\nsame\nfirst",panels[0])
            self.assertIn("R002\nsame\nsecond",panels[0])
            self.assertIn("R001\nsame\n3\npassA\nfailA",panels[1])
            self.assertIn("R002\nsame\n6\npassB\nfailB",panels[1])


if __name__=="__main__": unittest.main()
