"""Check PDF text bounds and render representative pages for human inspection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import fitz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_directory", type=Path)
    args = parser.parse_args()
    report = args.report_directory.resolve()
    path = report / "manuscript.pdf"
    output = Path("docs/proofs") / report.name
    output.mkdir(parents=True, exist_ok=False)
    outside, pages = [], []
    with fitz.open(path) as document:
        selected = {0, len(document)-1}
        table_pages, figure_pages = [], []
        for index, page in enumerate(document):
            blocks = page.get_text("blocks")
            words = page.get_text("words")
            text = page.get_text()
            pages.append({"page": index+1, "words": len(words)})
            for block in blocks:
                if block[6] == 0 and block[4].strip():
                    if block[0] < -0.5 or block[1] < -0.5 or block[2] > page.rect.width+0.5 or block[3] > page.rect.height+0.5:
                        outside.append({"page": index+1, "box": list(block[:4]), "text": block[4][:180]})
            if "R001" in text:
                table_pages.append(index)
            if page.get_images():
                figure_pages.append(index)
        selected.update(sorted(table_pages, key=lambda i: pages[i]["words"], reverse=True)[:2])
        selected.update(figure_pages[:2])
        renders = []
        for index in sorted(selected):
            target = output / f"page-{index+1:02d}.png"
            document[index].get_pixmap(matrix=fitz.Matrix(1.3, 1.3), alpha=False).save(target)
            renders.append(str(target.resolve()))
    result = {"pdf": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "checked_at": datetime.now(timezone.utc).isoformat(), "pages": pages,
              "off_page_text": outside, "rendered_pages": renders,
              "interpretation": "Text bounds checked automatically; clipping, column completeness and visual readability require inspection of rendered pages."}
    (output / "bounds.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"output": str(output), "pages": len(pages), "off_page_text": outside, "renders": renders}, indent=2))


if __name__ == "__main__":
    main()
