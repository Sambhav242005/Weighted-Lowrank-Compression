"""One-time additive IEEE paper update from audited, frozen GPU execution records."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EDITION = Path(__file__).resolve().parent
SOURCE = ROOT / "docs/paper-future-scope-20261015/ieee-conference.tex"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    before = (EDITION / "ieee-conference-before-gpu.tex").read_text(encoding="utf-8")
    assert SOURCE.read_text(encoding="utf-8") == before, "Do not overwrite a subsequent edit"
    paths = {
        "timing": ROOT / "results/gpu-lowrank-v2/20261016-mvb5e5qd/metrics.json",
        "quality": ROOT / "results/gpu-perplexity/20261016-mvb6ajxf/metrics.json",
        "timing_audit": ROOT / "results/gpu-lowrank-v2/20261016-mvb5e5qd/audit.json",
        "quality_audit": ROOT / "results/gpu-perplexity/20261016-mvb6ajxf/audit.json",
        "storage": ROOT / "results/gemma270m-relu/20261016-mvanbe94/metrics.json",
    }
    data = {key: json.loads(p.read_text()) for key, p in paths.items()}
    assert data["timing_audit"]["state"] == data["quality_audit"]["state"] == "passed"
    assert all(x["targets"] == 292281 for x in data["quality"]["full_test"].values())
    timings = data["timing"]["model"]
    quality = data["quality"]["full_test"]
    rows = [r"Dense reference & " + f"{quality['dense']['perplexity']:.4f}" + r" & -- & -- & -- \\"]
    extracted = []
    for label, method, qmethod in [("PyTorch factors", "pytorch", "pytorch_lowrank"),
            ("Two custom stages", "staged", "staged"), ("Graph (custom)", "graph_staged", "graph_staged"),
            ("Graph (cuBLAS)", "graph_cublas", "graph_cublas")]:
        times = [timings[w][method]["wall_ms"]["median"] for w in
                 ("prefill128", "prefill256", "prompt128_decode32")]
        q = quality[qmethod]["perplexity"]
        rows.append(label + " & " + f"{q:.4f}" + " & " + " & ".join(f"{x:.3f}" for x in times) + r" \\")
        extracted.append({"method": method, "quality_method": qmethod, "ppl": q, "latency_ms": times})
    protocol = r"""\subsection{Separate 270M GPU execution cohort}
An execution-only follow-up uses pretrained Gemma 3-270M in FP32 on an NVIDIA GeForce GTX 1050 (4\,GB), with PyTorch 2.7.1+cu118 and Transformers 4.55.4. It reuses one saved whitening fit (seed 11), replacing all 18 attention-output projections, each $1024\rightarrow640$, with rank-275 factors. No activation, refitting or anchor is added. The unchanged factors are executed with two PyTorch linear operations, two custom CUDA stages that compute the intermediate once, or per-projection CUDA Graphs of the custom stages or cuBLAS operations. Kernels use NVRTC 11.8 \cite{nvidia2022nvrtc}; graphs use PyTorch's capture/replay interface \cite{pytorch27graphs}. All new paths retain GPU cuBLAS for multi-token prefill.\par\smallskip
This cohort uses its own pinned token manifests: 16 training and eight validation windows of 256 tokens. Native WikiText-2 test evaluation uses context/stride 256/128, includes the final tail, and scores each of 292,281 shifted targets once. A separate teacher-forced diagnostic scores test positions 128--255 after a 128-token prompt to exercise the single-token execution paths. It is not a full-test estimate. Batch-one GPU timings use five warmups, 20 synchronized samples per condition and rotating method order. Prefill is measured at 128/256 tokens; the other workload includes a 128-token prompt plus 32 cached steps, rather than decode alone. Initialization and capture are excluded from warmed timings. This cohort is not pooled with the earlier FP16/BF16 or three-model quality comparisons.\par\smallskip
"""
    results = r"""\subsection{Equivalent factor execution does not establish acceleration}
Table~\ref{tab:gpu270-execution} reports the separate 270M cohort. All factor execution paths have measured native-test PPL 58.305858, versus 47.223106 for the dense teacher. Full-sequence PPL uses their shared cuBLAS prefill path. The 128-target teacher-forced diagnostic exercises the new decode paths: its PPL is 58.921703--58.921705 for the factorized methods and 42.895796 for dense. These subset values must not be substituted for full-test PPL. Eight validation windows and 16 greedy positions also pass numerical agreement checks against the PyTorch factors, with matching checked token choices.\par\smallskip
\begin{table}[t]
\centering
\caption{Separate Gemma 3-270M FP32 GPU execution study. Lower is better. Latencies are host-wall medians in milliseconds, including wrappers, allocation, prompt transfer and cache handling. ``P+D'' includes prompt 128 plus 32 cached steps. Dense PPL is remeasured, but dense timings are omitted because the rotating-order timing cohort did not include it. Each PPL uses all 292,281 test targets. New execution methods share GPU cuBLAS prefill.}
\label{tab:gpu270-execution}
{\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrr@{}}\toprule
Execution & PPL & \shortstack{Prefill\\128} & \shortstack{Prefill\\256} & P+D \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}}
\end{table}
None of the tested custom or graph paths establishes a latency advantage over PyTorch. Graph timings include copying new inputs and cloning outputs to prevent replay from overwriting caller-held results. The prompt/decode PyTorch p95 is 2.489\,s; new-path p95 values span 2.437--2.899\,s, so small median differences are descriptive, not statistically established effects. The saved dense and whitening checkpoints occupy 1,072,489,021 and 1,058,261,499 bytes, respectively (1.327\% reduction); the execution variants reuse the same factor parameters, while graph buffers add runtime state. Retained reference/graph buffers prevent interpreting allocator readings as isolated deployment-memory savings. These results concern the tested implementations and device, not custom GPU kernels in general.\par\smallskip
"""
    text = before.replace(r"\section{Results}", protocol + r"\section{Results}", 1)
    text = text.replace(r"\section{Discussion}", results + r"\section{Discussion}", 1)
    text = text.replace("A practical compression claim therefore needs separate evidence",
        "A separate 270M-parameter GPU execution study preserves the compressed model's likelihood but establishes no acceleration with the tested kernels or per-projection graphs. A practical compression claim therefore needs separate evidence", 1)
    marker = r"\section{Limitations and reproducibility}"
    text = text.replace(marker,
        r"The 270M execution cohort distinguishes agreement with a compressed reference from agreement with the original teacher. Executing the same factors accurately preserves their approximation error; fewer launches or smaller intermediate storage cannot restore discarded model information. Its unchanged perplexity and absent measured speed advantage reinforce the need to validate quality and execution cost separately.\par\smallskip" + "\n" + marker, 1)
    old = "This revision uses the saved measurements without changing the fitting budget, model or dataset."
    new = (r"The separate GPU follow-up is pinned to model revision \texttt{9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1} and the recorded WikiText-2 revision/token hashes. It reuses one fitted checkpoint, not independent calibration repeats. Frozen records in \texttt{results/gpu-lowrank-v2/20261016-mvb5e5qd} and \texttt{results/gpu-perplexity/20261016-mvb6ajxf} retain contracts, source/checkpoint hashes, raw timing samples, complete target coverage and audits. Its 600-second execution study and 3,600-second PPL evaluation completed within their separate caps; startup failure evidence from the earlier kernel attempt is preserved. This editorial revision adds those saved records without rerunning evaluation or changing any earlier fit. Its one-device timing results and short decode diagnostic cannot establish general acceleration or free-running equivalence.")
    assert old in text
    text = text.replace(old, new, 1)
    text = text.replace("The separate BF16 practical cohort saves 0.894\\% of the complete checkpoint size and has mixed timing results.",
        "The separate BF16 practical cohort saves 0.894\\% of the complete checkpoint size and has mixed timing results. The additional 270M FP32 GPU cohort preserves the low-rank reference's measured likelihood but establishes no execution speedup.", 1)
    text = text.replace("and with preparing formatting, table-assembly and plotting scripts.",
        "and with preparing formatting, table-assembly and plotting scripts and prototyping GPU execution code.", 1)
    references = r"""\bibitem{nvidia2022nvrtc} NVIDIA. NVRTC (Runtime Compilation), CUDA Toolkit 11.8 documentation, 2022. \href{https://docs.nvidia.com/cuda/archive/11.8.0/nvrtc/index.html}{Official documentation}.
\bibitem{pytorch27graphs} PyTorch. CUDA semantics: CUDA Graphs, PyTorch 2.7 documentation. Accessed October 16, 2026. \href{https://docs.pytorch.org/docs/2.7/notes/cuda.html\#cuda-graphs}{Official documentation}.
"""
    text = text.replace(r"\end{thebibliography}", references + r"\end{thebibliography}", 1)
    assert text.count(r"\label{tab:gpu270-execution}") == 1
    assert before.count(r"\begin{equation}") == text.count(r"\begin{equation}")
    assert before.count(r"\begin{figure}") == text.count(r"\begin{figure}")
    assert before.count(r"\begin{figure*}") == text.count(r"\begin{figure*}")
    SOURCE.write_text(text, encoding="utf-8")
    (EDITION / "evidence-extraction.json").write_text(json.dumps({
        "before_sha256": digest(EDITION / "ieee-conference-before-gpu.tex"), "after_sha256": digest(SOURCE),
        "frozen_inputs": {k: {"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for k, p in paths.items()},
        "table_values": extracted, "dense_ppl": quality["dense"]["perplexity"],
        "new_experiments_run_during_edit": False, "historical_equations_and_figures_preserved": True,
        "note": "Primary IEEE source updated; reference-style manuscript/Markdown remain prior historical versions."
    }, indent=2) + "\n", encoding="utf-8")
    print("IEEE source updated; evidence extraction saved")


if __name__ == "__main__":
    main()
