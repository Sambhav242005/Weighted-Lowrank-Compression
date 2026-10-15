"""Apply explicit prose edits to the frozen editorial edition; preserve data.

Humanizer 3.1.0 principles: explain for the intended reader, use direct verbs,
remove formulaic framing, and retain every scientific limit. Numeric tables,
math expressions, citation keys and link targets are checked separately.
"""
import difflib
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'docs/paper-reference-style-20261015/manuscript.md'

REPLACEMENTS=[
("Replacing a dense projection with two smaller matrices offers a direct way to reduce stored weights. The difficult question is whether those replacements preserve the behavior of the complete model. A projection can reconstruct its teacher outputs on calibration inputs yet encounter different activations after earlier projections have also been replaced. Conversely, an altered model can improve held-out next-token likelihood while moving farther from the teacher's internal representations. These outcomes test different properties and should not be summarized by a single compression score.",
 "Language models use weight matrices to transform text into predictions. Low-rank compression replaces a large matrix with two smaller factors. The replacement saves weights, but its effect depends on the rest of the model. A factorized layer may closely match the original on calibration inputs and still receive different inputs after earlier layers have been compressed. A compressed model may also predict text more accurately while its internal representations move farther from those of the original model, which we call the teacher. We measure these outcomes separately."),
("We ask: **at a fixed factor budget, does adding a weight-space anchor to activation-weighted low-rank fitting consistently improve both held-out likelihood and residual-state fidelity, and what storage and execution benefit remains?** The anchor changes the fitting objective without adding an inference branch. This makes it a controlled comparison against unanchored covariance weighting, rather than a comparison between differently sized representations.",
 "We test whether a weight-space anchor improves prediction quality and internal similarity when the number of stored factor weights is fixed. Activation weighting gives more importance to input directions observed during calibration. The anchor adds a penalty for changing the original weights. It changes how the factors are fitted but adds no branch at inference. We compare it with covariance weighting without the anchor, using the same factor budget, and measure storage and execution cost."),
("Our contribution is a bounded empirical analysis with three parts. First, we distinguish activation weighting, anchoring and extra ridge shrinkage mathematically and compare their recorded controls. Second, we examine complete attention-output stacks and consecutive groups, keeping likelihood and propagated residual drift separate. Third, we distinguish targeted parameter savings from serialized whole-model size and measured runtime. The results expose useful failures: average improvements need not reproduce across seeds or groups, and better likelihood can accompany substantially greater drift.",
 "The study separates the effects of activation weighting, anchoring and extra ridge shrinkage. We test attention-output replacements across the full stack and in consecutive groups, measuring prediction quality alongside the changes that propagate through the residual stream. We also measure complete checkpoint size and runtime. The results show where an average joint improvement fails to hold across calibration seeds or layer groups, and where better prediction quality accompanies greater internal drift."),
("This is an exploratory study, rather than a claim of a new whitening algorithm or a solved global-preservation problem. The strict numerical-readiness checks were not uniformly satisfied. The Qwen primary comparison is available, but its validation frontier is incomplete. The later BF16 practical cohort has only two calibration seeds and adds no BF16 drift measurement. We retain those boundaries throughout the analysis.",
 "The results remain exploratory. Some strict numerical accuracy checks failed, and Qwen's primary comparison is available without a complete validation rank frontier. The later practical cohort uses bfloat16 (BF16), has only two calibration seeds, and adds no BF16 drift measurements. We do not claim a new whitening algorithm or reliable preservation of the complete model's computation."),
("Our fixed-rank diagonal-scaling and covariance-weighted controls omit rank search and recovery fine-tuning. They are component comparisons, not reproductions of either complete pipeline.",
 "Our controls use fixed ranks and omit rank search and recovery fine-tuning. They isolate the diagonal-scaling and covariance-weighting components of those methods."),
("These approaches motivate future matched-budget studies; none was executed as a comparator here.",
 "These methods remain candidates for future comparisons at matched budgets; we did not run them here."),
("We cite these methods as context, not as measured baselines or evidence of an LLM speedup. Centered kernel alignment and temperature scaling supply diagnostic and calibration tools, respectively; neither certifies teacher preservation",
 "These methods provide context for the practical measurements; we did not benchmark them or establish faster language-model inference. Centered kernel alignment and temperature scaling provide representation and calibration diagnostics, respectively; neither establishes teacher preservation"),
("Factors contain $r(o+i)$ weights rather than $oi$; a parameter reduction therefore requires $r<oi/(o+i)$. A smaller rank alone is not a sufficient storage claim.",
 "The factors contain $r(o+i)$ weights, compared with $oi$ in the dense matrix. Parameter savings therefore require $r<oi/(o+i)$. Rank reduction by itself does not specify the storage saving."),
("Anchoring changes fitting geometry, not the factor budget or inference graph.",
 "The anchor changes the fitting geometry while keeping the factor budget and inference graph fixed."),
("Covariance fitting and weighted decomposition use FP64 computation, then FP32 factors before the declared storage cast; plain SVD uses FP32.",
 "Covariance fitting and weighted decomposition use 64-bit floating-point computation (FP64), then 32-bit floating-point factors (FP32) before the declared storage cast; plain SVD uses FP32."),
("Thus $D$ is an average of block-level energy ratios, not a last-layer error or an average of per-token ratios.",
 "The metric $D$ averages the energy ratio for each block. It differs from both a last-layer error and an average of per-token ratios."),
("The original strict FP16 factorization-readiness gate failed.",
 "The original strict numerical check for 16-bit floating-point (FP16) factorization failed."),
("These controls make the following results exploratory observations under recorded implementations, rather than a numerically certified comparison of ideal objectives.",
 "Because these checks failed, the results describe the recorded implementations. They cannot isolate the ideal mathematical objectives from numerical error."),
("Its inherited observations are reused records, not independent replications.",
 "The continuation reuses observations from an interrupted attempt; they do not count as independent replications."),
("This counterexample rules out an interpretation that the modest full-stack average gain implies uniform group-level benefit. It does not prove that drift must grow monotonically with group size: layer position, downstream sensitivity and the inclusion of unchanged blocks all affect the aggregate.",
 "The full-stack average gain therefore does not hold for every group. Drift also need not grow monotonically with group size: layer position, downstream sensitivity and the unchanged blocks included in the average all affect the result."),
("LFM exhibits the central metric disagreement:","LFM shows the clearest disagreement between the metrics:"),
("The results distinguish three questions. A representation can be smaller than its dense source, fit a data distribution better than a comparator, and still diverge from the teacher internally. None of these outcomes logically supplies the others. In particular, the LFM observation is evidence against using likelihood improvement alone as a preservation certificate. It is not evidence that residual similarity is always the correct optimization target: $D$ measures similarity in fixed coordinates, while different internal states can implement similar functions. A future claim about global preservation needs converging evidence from final logits, tasks, drift and generation rather than privileging one proxy.",
 "LFM predicts text better under the anchored fit while its residual representations differ more from the teacher's. This result limits what likelihood alone can tell us about preservation. Residual similarity also has limits: $D$ compares states in fixed coordinates, and different states can implement similar functions. A stronger preservation claim needs agreement across final logits, task performance, drift measurements and generation tests. Smaller storage or a better score on one measure does not supply that evidence."),
("The strongest comparison for anchoring is matched covariance weighting. Large gains over plain SVD primarily establish the importance of data-aware fitting within this implementation; they do not establish the incremental value or novelty of the anchor. Gemma's small average advantage, Qwen's joint worsening and LFM's tradeoff do not support a model-independent recommendation to add the same anchor. Because model weights, tokenizers, mixer structures and historical inference precision differ, this study cannot attribute the different outcomes solely to architecture.",
 "Matched covariance weighting is the relevant baseline for the anchor. Gains over plain SVD show the value of data-aware fitting in this implementation, but cannot establish what the anchor adds or whether it is novel. Gemma's small average advantage, Qwen's joint worsening and LFM's tradeoff provide no consistent basis for adding the same anchor across models. The models also differ in pretrained weights, tokenizers, mixers and historical inference precision, so we cannot attribute those outcomes solely to architecture."),
("Retaining these failures is useful: it identifies a prerequisite for stronger future comparisons instead of allowing a successful identity split to validate a different decomposition path.",
 "Future comparisons need to resolve these failures. An exact identity split does not validate a different decomposition path."),
("Finally, model-level storage and speed depend on coverage and implementation.",
 "Model-level storage and speed depend on which layers are compressed and how the factors execute."),
("MLPs, hybrid non-attention mixers, larger models and other rank allocations require separate conclusions.",
 "Multilayer perceptrons (MLPs), hybrid non-attention mixers, larger models and other rank allocations require separate conclusions."),
("The editorial revision launches no research experiment and changes no saved metric, fitting budget, model or dataset.",
 "This revision uses the saved measurements without changing the fitting budget, model or dataset."),
("At the measured attention-output budgets, activation-aware low-rank fitting improves on plain SVD, but anchoring does not consistently improve both likelihood and propagated residual fidelity over matched whitening. Gemma's small full-stack average change does not transfer uniformly to layer groups; Qwen worsens on both metrics; LFM improves likelihood while increasing drift. The separate BF16 practical cohort achieves a measured 0.894% whole-checkpoint saving with mixed timing. These findings support reporting likelihood, teacher-state fidelity and practical cost as distinct outcomes. Reliable preservation of global computation remains unresolved.",
 "Activation-aware low-rank fitting improves on plain SVD at the tested attention-output budgets. Adding the anchor provides no consistent joint improvement in likelihood and residual fidelity over matched whitening. Gemma's small average gain does not hold across all layer groups. Qwen worsens on both metrics, while LFM improves likelihood and increases drift. The separate BF16 practical cohort saves 0.894% of the complete checkpoint size and has mixed timing results. Compression remains unproven as a way to preserve the original computation across the tested models.")]


def main():
    destination=HERE/'manuscript.source.md'
    if destination.exists():raise FileExistsError(destination)
    old=SOURCE.read_text(encoding='utf-8')
    text=old
    before_abstract=re.search(r'(?<=## Abstract\n\n).*?(?=\n\nKeywords:)',text,re.S)[0]
    abstract=(HERE/'abstract.md').read_text(encoding='utf-8').strip()
    assert 150<=len(abstract.split())<=250
    assert not re.search(r'\$|\[@|\n|\b(?:NLL|PPL|BF16|FP16|SVD)\b',abstract)
    text=text.replace(before_abstract,abstract,1)
    edits=[]
    for before,after in REPLACEMENTS:
        assert text.count(before)==1,before[:100]
        text=text.replace(before,after,1)
        edits.append({'before':before,'after':after})
    # This acknowledgment records observed assistance; it does not attest that
    # the author has already completed the final scientific review.
    acknowledgment=('## Acknowledgment\n\nOpenAI Codex [@openaicodex2026] assisted with drafting and editing the Abstract, Introduction, Methods, Discussion and Conclusion, and with preparing formatting, table-assembly and plotting scripts. Numerical results in this revision come from the saved experiment records; this editorial work did not run a new compression experiment.\n\n')
    text=text.replace('## Appendix A.',acknowledgment+'## Appendix A.',1)
    tables=lambda value:[line for line in value.splitlines() if line.startswith('|')]
    math=lambda value:re.findall(r'\$[^$]+\$',value)
    links=lambda value:re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',value)
    assert tables(text)==tables(old)
    assert math(text)==math(old)
    assert links(text)==links(old)
    assert set(re.findall(r'\[@([^\]]+)\]',text))==set(re.findall(r'\[@([^\]]+)\]',old))|{'openaicodex2026'}
    destination.write_text(text,encoding='utf-8',newline='\n')
    (HERE/'editorial.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='previous-manuscript.md',tofile='manuscript.source.md')),encoding='utf-8')
    notes={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
           'abstract_words':len(abstract.split()),'humanizer_version':'3.1.0',
           'humanizer_source':'https://github.com/blader/humanizer/blob/main/SKILL.md',
           'tables_unchanged':True,'mathematics_unchanged':True,'existing_link_targets_unchanged':True,
           'original_22_citation_keys_retained':True,'additional_citation':'OpenAI Codex in assistance acknowledgment',
           'prose_edits':edits,'research_experiments_run':False}
    (HERE/'editorial-audit.json').write_text(json.dumps(notes,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'abstract_words':len(abstract.split()),'prose_edits':len(edits),'tables_and_math':'unchanged'},indent=2))


if __name__=='__main__':main()
