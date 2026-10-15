"""Readable artifact-based review PDF using existing ReportLab, no GPU/install.

API export_pdf(report_directory)->Path. This is an independent PDF rendering
of the same manuscript source; it does not claim that LaTeX compilation passed.
"""
import argparse
import hashlib
from html import escape
import importlib.util
import json
from pathlib import Path
import re


def plain_math(value):
    """Readable ASCII equation notation; exact TeX remains alongside the PDF."""
    value = re.sub(r"\\in\b"," in ",value)
    value = re.sub(r"\\(?:mathbb|mathrm|mathbf|operatorname)\{([^{}]*)\}",r"\1",value)
    value = value.replace(r"\|","||").replace(r"\,"," ").replace(r"\!","")
    replacements = {"times":" x ","in":" in ","beta":"beta","rho":"rho","lambda":"lambda",
                    "dagger":"dagger","max":"max","min":"min","mean":"mean"}
    value = re.sub(r"\\([A-Za-z]+)",lambda m:replacements.get(m[1],m[1]),value)
    return value


def text_markup(value):
    value = str(value)
    value = re.sub(r"\$([^$\n]+)\$",lambda m:plain_math(m[1]),value)
    value = value.replace("—","--").replace("–","-").replace("−","-")
    value = re.sub(r"!\[([^]]*)\]\([^)]*\)",r"\1",value)
    value = re.sub(r"\[([^]]+)\]\(([^)]+)\)",r"\1 (\2)",value)
    value = escape(value)
    value = re.sub(r"\*\*([^*]+)\*\*",r"<b>\1</b>",value)
    value = re.sub(r"`([^`]+)`",r"<font name='Courier'>\1</font>",value)
    return value


def export_pdf(report_directory):
    # Imports deferred so API discovery does not require the PDF environment.
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, LongTable, TableStyle, Image
    directory = Path(report_directory).resolve()
    output = directory/"manuscript.pdf"
    if output.exists(): raise FileExistsError(output)
    report_path, source = directory/"report.json",directory/"manuscript.md"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("promotion_disabled") is not True:
        raise ValueError("review PDF requires promotion_disabled=true")
    font = "Times-Roman"
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    if font_path.is_file():
        font = "ReviewArial"
        if font not in pdfmetrics.getRegisteredFontNames(): pdfmetrics.registerFont(TTFont(font,str(font_path)))
    styles = getSampleStyleSheet()
    for name in ("Normal","BodyText","Title","Heading1","Heading2","Heading3"):
        styles[name].fontName = font
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 12
    styles["BodyText"].spaceAfter = 6
    cell = ParagraphStyle("ReviewCell",parent=styles["BodyText"],fontSize=6.7,leading=8.5,spaceAfter=0,splitLongWords=True)
    tiny = ParagraphStyle("ReviewSource",parent=styles["BodyText"],fontSize=7,leading=9,splitLongWords=True)
    width,height = A4
    available = width-80
    document = SimpleDocTemplate(str(output),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=45,bottomMargin=42,
        title="Activation-Weighted Low-Rank Compression: Likelihood, Drift, and Practical Cost",
        author="ResearchCompression review draft")
    story = [Paragraph("Activation-Weighted Low-Rank Compression: Likelihood, Drift, and Practical Cost",styles["Title"]),
        Paragraph("PREPRINT REVIEW DRAFT -- no venue assumed",styles["Heading2"]),
        Paragraph("No established global-preservation or practical-compression claim. Original FP16 readiness remains failed. "
        "Native-tokenizer perplexities are compared within models only. Partial and cancelled attempts remain evidence, not completed cohorts.",styles["BodyText"]),
        Paragraph("Equations use readable ASCII notation (beta, rho, lambda; dagger denotes the pseudoinverse). "
        "The accompanying standalone manuscript.tex provides exact typeset mathematical source. This PDF is independently rendered from the same Markdown and measured figures.",tiny)]
    lines = source.read_text(encoding="utf-8").splitlines()
    if lines and lines[0]=="---":
        end = next(i for i in range(1,len(lines)) if lines[i]=="---")
        lines = lines[end+1:]
    index = 0
    while index<len(lines):
        line = lines[index]
        if line.startswith("| "):
            rows = []
            while index<len(lines) and lines[index].startswith("|"):
                cells = [c.strip() for c in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-:]+",c) for c in cells): rows.append(cells)
                index += 1
            count = max(map(len,rows))
            if count>5:
                rows = [["Row ID"]+row for row in rows[:1]]+[[f"R{i:03d}"]+row for i,row in enumerate(rows[1:],1)]
                count += 1
                panels = [[0,1]+list(range(start,min(start+3,count))) for start in range(2,count,3)]
                story.append(Paragraph("Wide-table continuation: identical row IDs identify the same original record across every panel. IDs restart for each source table.",tiny))
            else:
                panels = [list(range(count))]
            for panel_index,columns in enumerate(panels,1):
                if len(panels)>1: story.append(Paragraph(f"Panel {panel_index} of {len(panels)}; match by Row ID.",tiny))
                values = [[Paragraph(text_markup((r+[""]*count)[c]),cell) for c in columns] for r in rows]
                table = LongTable(values,colWidths=[available/len(columns)]*len(columns),repeatRows=1,splitByRow=1,splitInRow=1,hAlign="LEFT")
                table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e7edf3")),
                    ("LINEBELOW",(0,0),(-1,0),.5,colors.HexColor("#536577")),("VALIGN",(0,0),(-1,-1),"TOP"),
                    ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
                    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
                    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f4f6f8")])]))
                story += [table,Spacer(1,9)]
            continue
        image = re.match(r"!\[([^]]*)\]\(([^)]+)\)",line)
        if image:
            image_path = (directory/image[2]).resolve()
            if not image_path.is_relative_to(directory): raise ValueError("external figure path")
            if image_path.is_file():
                item = Image(str(image_path))
                scale = min(available/item.imageWidth,380/item.imageHeight)
                item.drawWidth,item.drawHeight = item.imageWidth*scale,item.imageHeight*scale
                story += [item,Paragraph(text_markup(image[1]),tiny),Spacer(1,8)]
            else: story.append(Paragraph("Figure unavailable: "+text_markup(image[1]),tiny))
        elif line.startswith("#"):
            heading = re.match(r"(#+)\s+(.*)",line)
            if heading and len(heading[1])>1:
                story.append(Paragraph(text_markup(heading[2]),styles["Heading"+str(min(len(heading[1])-1,3))]))
        elif line.strip():
            story.append(Paragraph(text_markup(line),styles["BodyText"]))
        index += 1
    story.append(Paragraph("Paired-block uncertainty appendix",styles["Heading1"]))
    story.append(Paragraph("Intervals are candidate minus the same-split baseline, using evaluation blocks rather than individual tokens. "
        "Partial-tail targets remain included in full NLL and are excluded from fixed-block intervals. These are not calibration-seed intervals.",styles["BodyText"]))
    interval_count = 0
    for run in report.get("runs",[]):
        for comparison in run.get("comparisons",[]):
            values = [["Seed","Full NLL delta","Subset delta","95% interval","Blocks/resamples","Excluded targets"]]
            for row in comparison.get("per_seed",[]):
                uncertainty = row.get("uncertainty",{})
                if not uncertainty.get("available"): continue
                interval_count += 1
                values.append([str(v) for v in (row.get("seed"),uncertainty.get("full_test_difference"),
                    uncertainty.get("bootstrap_subset_difference"),uncertainty.get("confidence_interval_95"),
                    str(uncertainty.get("blocks"))+"/"+str(uncertainty.get("resamples")),uncertainty.get("excluded_partial_targets"))])
            if len(values)>1:
                story.append(Paragraph(text_markup(run.get("run_id","unknown")),styles["Heading3"]))
                story.append(Paragraph(text_markup(json.dumps({"candidate":comparison.get("candidate"),"baseline":comparison.get("baseline"),"scope":comparison.get("scope")},sort_keys=True)),tiny))
                table = LongTable([[Paragraph(text_markup(v),cell) for v in row] for row in values],
                    colWidths=[available/6]*6,repeatRows=1,splitByRow=1,splitInRow=1)
                table.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e7edf3")),
                    ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3)]))
                story += [table,Spacer(1,8)]
    if not interval_count: story.append(Paragraph("No measured paired-block intervals were supplied.",styles["BodyText"]))
    # Full checked bibliography is embedded; no external BibTeX runtime required.
    spec = importlib.util.spec_from_file_location("review_latex",Path(__file__).with_name("export_paper_latex_20261014.py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    references = module.bibliography_entries((directory/"paper_protocol_v2.bib").read_text(encoding="utf-8"))
    if len(references)!=22: raise ValueError("expected 22 verified references")
    story.append(Paragraph("Verified references",styles["Heading1"]))
    for index,(key,fields) in enumerate(references,1):
        entry = ". ".join(fields[k] for k in ("author","title","booktitle","journal","year","note","url") if fields.get(k))
        story.append(Paragraph(text_markup(f"[{index}] [{key}] {entry}"),tiny))
    story.append(Paragraph("Source integrity",styles["Heading1"]))
    for path in (report_path,source,directory/"paper_protocol_v2.bib",Path(__file__)):
        story.append(Paragraph(text_markup(path.name+" SHA256 "+hashlib.sha256(path.read_bytes()).hexdigest()),tiny))
    def footer(canvas,doc):
        canvas.saveState(); canvas.setFont("Helvetica",7)
        canvas.drawString(40,25,"ResearchCompression | exploratory review draft")
        canvas.drawRightString(width-40,25,str(doc.page)); canvas.restoreState()
    # Exclusive reservation enforces no overwrite even if another exporter races.
    with output.open("xb"): pass
    document.build(story,onFirstPage=footer,onLaterPages=footer)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_directory",type=Path)
    print(export_pdf(parser.parse_args(argv).report_directory))


if __name__=="__main__": main()
