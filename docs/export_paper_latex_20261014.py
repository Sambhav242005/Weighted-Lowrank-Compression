"""Stdlib standalone review-draft export; no external TeX assets or installs.

API: export_report(report_directory) -> Path. The finalizer must first write
report.json, manuscript.md, bibliography, and measured figure provenance.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def escape(text):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
                    "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    text = str(text).replace("±", r"\(\pm\)").replace("−", "-")
    text = text.replace("–", "--").replace("—", "---").replace("×", " x ")
    # Generated commands above are added after escaping user-controlled text.
    text = str(text).replace(r"\(\pm\)", "@@PLUSMINUS@@")
    return "".join(replacements.get(c, c) for c in text).replace("@@PLUSMINUS@@", r"\(\pm\)")


def inline(text):
    """Escape prose/code/URLs, retaining only generated manuscript dollar math."""
    tokens = []
    def keep(value):
        tokens.append(value)
        return "LATEXTOKEN" + str(len(tokens)-1) + "END"
    text = re.sub(r"`([^`]+)`", lambda m: keep(r"\texttt{"+escape(m[1])+"}"), text)
    text = re.sub(r"\$([^$\n]+)\$", lambda m: keep("$"+m[1].replace(r"\lambda_\max",r"\lambda_{\max}")+"$"), text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: keep(escape(m[1])+" ("+escape(m[2])+")"), text)
    text = re.sub(r"\[@([^\]]+)\]", lambda m: keep(r"\cite{"+m[1]+"}"), text)
    text = re.sub(r"\*\*([^*]+)\*\*", lambda m: keep(r"\textbf{"+escape(m[1])+"}"), text)
    result = escape(text)
    for i, value in enumerate(tokens):
        result = result.replace("LATEXTOKEN"+str(i)+"END", value)
    return result


def markdown_to_latex(markdown):
    lines = markdown.splitlines()
    if lines and lines[0] == "---":
        end = next((i for i in range(1,len(lines)) if lines[i] == "---"), 0)
        lines = lines[end+1:]
    out, index = [], 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("!["):
            # The same measured coordinates are embedded below; never external PNGs.
            index += 1
            continue
        if line.startswith("| "):
            table = []
            while index < len(lines) and lines[index].startswith("|"):
                cells = [c.strip() for c in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-:]+", c) for c in cells):
                    table.append(cells)
                index += 1
            count = max(map(len, table))
            # Row IDs identify original records even when the first column is
            # repeated across variants/scopes. The same IDs occur in each panel.
            if count > 5:
                table = [["Row ID"]+row for row in table[:1]] + [[f"R{i:03d}"]+row for i,row in enumerate(table[1:],1)]
                count += 1
                panels = [[0,1]+list(range(start,min(start+3,count))) for start in range(2,count,3)]
                out.append("Wide-table continuation: identical row IDs identify the same original record across every panel; IDs restart for each source table.")
            else:
                panels = [list(range(count))]
            for panel_index,columns in enumerate(panels,1):
                if len(panels)>1: out.append(f"\\par\\noindent Panel {panel_index} of {len(panels)}; match by Row ID.")
                width = .94 / len(columns)
                out += [r"{\scriptsize\setlength{\tabcolsep}{2pt}",
                        r"\begin{longtable}{" + ("p{"+f"{width:.4f}"+r"\textwidth}")*len(columns)+"}"]
                for row_index,row in enumerate(table):
                    padded = row + [""]*(count-len(row))
                    out.append(" & ".join(inline(padded[c]) for c in columns)+r" \\")
                    if row_index == 0:
                        out += [r"\hline\endhead"]
                out += [r"\end{longtable}}"]
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading:
            level, title = len(heading[1]), heading[2]
            if level != 1:
                command = "section" if level == 2 else "subsection" if level == 3 else "subsubsection"
                out.append("\\"+command+"{"+inline(title)+"}")
        elif line.startswith("- "):
            out.append(r"\par\noindent\textbullet\quad "+inline(line[2:])+r"\par")
        elif line.startswith("```"):
            index += 1
            code = []
            while index < len(lines) and not lines[index].startswith("```"):
                code.append(lines[index]); index += 1
            out += [r"\begin{quote}\ttfamily\small", escape("\n".join(code)), r"\end{quote}"]
        else:
            out.append(inline(line))
        index += 1
    return "\n".join(out)


def bibliography_entries(source):
    """Parse the checked conventional BibTeX fields with nested-brace support."""
    entries = []
    for match in re.finditer(r"@\w+\s*\{\s*([^,]+),", source):
        pos, depth = match.end(), 1
        start = pos
        while pos < len(source) and depth:
            if source[pos] == "{": depth += 1
            elif source[pos] == "}": depth -= 1
            pos += 1
        body, fields = source[start:pos-1], {}
        cursor = 0
        while cursor < len(body):
            field = re.search(r"(\w+)\s*=\s*", body[cursor:])
            if not field: break
            name = field[1].lower(); cursor += field.end()
            if cursor < len(body) and body[cursor] == "{":
                begin = cursor+1; cursor += 1; nesting = 1
                while cursor < len(body) and nesting:
                    if body[cursor] == "{": nesting += 1
                    elif body[cursor] == "}": nesting -= 1
                    cursor += 1
                value = body[begin:cursor-1]
            else:
                end = body.find(",",cursor)
                if end < 0: end = len(body)
                value = body[cursor:end].strip().strip('"'); cursor = end+1
            fields[name] = value.replace("{", "").replace("}", "")
        entries.append((match[1].strip(), fields))
    if len({key for key,_ in entries}) != len(entries):
        raise ValueError("duplicate bibliography keys")
    return entries


def number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def picture(series, xlabel, ylabel):
    """Native LaTeX picture: measured straight segments, no interpolated curves."""
    pairs = [(x,y) for _,points in series for x,y in points]
    if not pairs or not all(number(x) and number(y) for x,y in pairs):
        raise ValueError("plot requires finite measured coordinates")
    xmin,xmax = min(x for x,_ in pairs),max(x for x,_ in pairs)
    ymin,ymax = min(y for _,y in pairs),max(y for _,y in pairs)
    if xmin == xmax: xmin -= .5; xmax += .5
    if ymin == ymax: ymin -= .5; ymax += .5
    def coordinate(x,y): return 35+290*(x-xmin)/(xmax-xmin),25+140*(y-ymin)/(ymax-ymin)
    out = [r"\begin{center}\setlength{\unitlength}{1pt}", r"\begin{picture}(350,195)",
           r"\put(35,25){\line(1,0){290}}",r"\put(35,25){\line(0,1){140}}"]
    for series_index,(label,points) in enumerate(series,1):
        transformed = [coordinate(x,y) for x,y in points]
        for (x,y),(xx,yy) in zip(transformed,transformed[1:]):
            out.append(f"\\qbezier({x:.3f},{y:.3f})({(x+xx)/2:.3f},{(y+yy)/2:.3f})({xx:.3f},{yy:.3f})")
        for x,y in transformed:
            out.append(f"\\put({x:.3f},{y:.3f}){{\\circle*{{2}}}}")
        x,y = transformed[-1]
        out.append(f"\\put({x+3:.3f},{y+3:.3f}){{\\makebox(0,0)[l]{{\\tiny {series_index}}}}}")
    for x,y,label in [(35,12,f"{xmin:.5g}"),(305,12,f"{xmax:.5g}"),(0,25,f"{ymin:.5g}"),(0,165,f"{ymax:.5g}")]:
        out.append(f"\\put({x},{y}){{\\makebox(0,0)[l]{{\\tiny {escape(label)}}}}}")
    out += [r"\put(170,0){\makebox(0,0){\scriptsize "+escape(xlabel)+"}}",
            r"\put(35,185){\makebox(0,0)[l]{\scriptsize "+escape(ylabel)+"}}",
            r"\end{picture}\end{center}"]
    for index,(label,points) in enumerate(series,1):
        # Exact coordinate table makes overlapping monochrome curves auditable.
        out.append(r"\noindent\footnotesize "+escape(f"Series {index}: {label}; {len(points)} measured points.")+r"\par")
    return "\n".join(out)


def embedded_figures(manifest):
    out = [r"\section{Embedded measured figures}",
           "Coordinates below come from the figure provenance associated with this report. "
           "Lines connect recorded measurements; no fitted curve or fabricated points are added. "
           "Each attempt, model and split remains separate. Sample means are not confidence intervals."]
    seen = set()
    for item in manifest.get("inputs",[]):
        run, data = item.get("run_id","unknown"),item.get("figure_data")
        if run.startswith("baselines-") and isinstance(data,list):
            series = [(json.dumps(s.get("specification",{}),sort_keys=True),list(enumerate(s["mean"]))) for s in data if s.get("mean")]
            if series:
                out += [r"\subsection{Gemma residual drift: "+escape(run)+"}",picture(series,"Block index","Normalized residual squared error")]; seen.add("drift")
        elif run.startswith("practical-") and isinstance(data,list):
            series = [(str(p.get("label"))+" seed="+str(p.get("seed"))+" bits="+str(p.get("bits")),[(p["whole_checkpoint_bytes"]/2**20,p["raw_test_nll"])]) for p in data]
            if series:
                out += [r"\subsection{Gemma measured storage: "+escape(run)+"}",picture(series,"Whole checkpoint MiB","Raw native-test NLL")]; seen.add("storage")
        elif run.startswith("architecture-replication-") and isinstance(data,dict):
            for key,title in (("primary_test","Primary test"),("validation_frontier","Validation frontier (not confirmation)")):
                methods = {}
                for group in data.get(key,[]):
                    methods.setdefault(json.dumps(group.get("specification",{}),sort_keys=True),[]).append((group["retained_target_parameter_fraction"],group["mean"]))
                if methods:
                    out += [r"\subsection{"+escape(str(data.get("model"))+": "+title+" / "+run)+"}",picture([(label,sorted(points)) for label,points in methods.items()],"Target factor parameter fraction","Same-split teacher-relative NLL")]; seen.add("frontier" if key=="validation_frontier" else "architecture")
    for kind in ("drift","storage","frontier"):
        if kind not in seen:
            out.append(r"\par\textbf{Unmeasured/unavailable: "+escape(kind)+".} No substitute graph is invented.")
    for item in manifest.get("unavailable",[]):
        out.append(r"\par\small "+escape(json.dumps(item,sort_keys=True)))
    return "\n".join(out)


def export_report(report_directory):
    directory = Path(report_directory).resolve()
    output = directory/"manuscript.tex"
    if output.exists(): raise FileExistsError(output)
    required = [directory/name for name in ("report.json","manuscript.md","paper_protocol_v2.bib")]
    for path in required:
        if not path.is_file(): raise FileNotFoundError(path)
    report = json.loads(required[0].read_text(encoding="utf-8"))
    if report.get("promotion_disabled") is not True:
        raise ValueError("review exporter requires explicit promotion_disabled=true")
    markdown = required[1].read_text(encoding="utf-8")
    bibliography = bibliography_entries(required[2].read_text(encoding="utf-8"))
    if len(bibliography) != 22: raise ValueError("expected all 22 verified references")
    figure_path = directory/"figures"/"provenance.json"
    figures = json.loads(figure_path.read_text(encoding="utf-8")) if figure_path.is_file() else {}
    content = [r"\documentclass[10pt]{article}",r"\usepackage[T1]{fontenc}",
               r"\usepackage[margin=0.7in]{geometry}",r"\usepackage{amsmath,amssymb,longtable}",
               r"\setlength{\emergencystretch}{3em}",r"\sloppy",
               r"\title{Activation-Weighted Low-Rank Compression: Likelihood, Drift, and Practical Cost}",
               r"\author{ResearchCompression --- review draft}",r"\date{Draft generated "+escape(report.get("generated_utc","unrecorded"))+"}",
               r"\begin{document}\maketitle",
               r"\noindent\textbf{Preprint review draft. No venue assumed. No established global-preservation or practical-compression claim.}",
               "Original FP16 readiness remains failed; separate numerical controls do not repair that cohort. "
               "Native-tokenizer perplexities must not rank different models. Incomplete attempts and declared omissions remain visible.",
               markdown_to_latex(markdown),embedded_figures(figures),r"\appendix\section{Paired block uncertainty by attempt and scope}",
               "Each interval is candidate minus the same-split baseline. Evaluation-block uncertainty does not measure calibration-seed uncertainty. "
               "Full-test NLL includes the partial tail; fixed-block intervals exclude it. Missing intervals remain unavailable."]
    interval_count = 0
    for run in report.get("runs",[]):
        for comparison in run.get("comparisons",[]):
            rows = ["| Seed | Full NLL delta | Block-subset delta | 95 percent interval | Blocks / resamples | Excluded targets |",
                    "|---|---|---|---|---|---|"]
            for row in comparison.get("per_seed",[]):
                uncertainty = row.get("uncertainty",{})
                if not uncertainty.get("available"): continue
                interval_count += 1
                rows.append("| "+" | ".join(str(v) for v in (row.get("seed"),uncertainty.get("full_test_difference"),
                    uncertainty.get("bootstrap_subset_difference"),uncertainty.get("confidence_interval_95"),
                    str(uncertainty.get("blocks"))+" / "+str(uncertainty.get("resamples")),uncertainty.get("excluded_partial_targets")))+" |")
            if len(rows)>2:
                content += [r"\subsection{"+escape(run.get("run_id","unknown"))+"}",
                    escape(json.dumps({"candidate":comparison.get("candidate"),"baseline":comparison.get("baseline"),"scope":comparison.get("scope")},sort_keys=True)),
                    markdown_to_latex("\n".join(rows))]
    if not interval_count: content.append("No measured paired-block intervals were supplied.")
    content.append(r"\section{Export provenance}")
    for path in required + ([figure_path] if figure_path.is_file() else []) + [Path(__file__)]:
        content.append(r"\par\noindent\small "+escape(path.name+" SHA256 "+hashlib.sha256(path.read_bytes()).hexdigest()))
    content += [r"\section{Verified references}",r"\begin{thebibliography}{99}"]
    for key,fields in bibliography:
        if not re.fullmatch(r"[A-Za-z0-9:_-]+",key): raise ValueError("unsafe bibliography key")
        citation = ". ".join(fields[k] for k in ("author","title","booktitle","journal","year","note","url") if fields.get(k))
        content.append(r"\bibitem{"+key+"} "+escape(citation))
    content += [r"\end{thebibliography}",r"\end{document}"]
    with output.open("x",encoding="utf-8",newline="\n") as stream:
        stream.write("\n".join(content)+"\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_directory",type=Path)
    args = parser.parse_args(argv)
    print(export_report(args.report_directory))


if __name__ == "__main__": main()
