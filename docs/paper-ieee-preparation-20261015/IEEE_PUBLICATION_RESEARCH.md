# IEEE publication preparation - checked 15 October 2026

The paper is being prepared for the teacher's review. The user reports that it
must be submitted before **17 October 2026** or a **₹1,000 charge** will apply,
but does not know the conference or journal name. The venue, submission system,
exact cutoff time/time zone, and purpose of that charge remain unverified.
Searching those details did not identify a matching event. No venue was selected,
paper submitted, account created, message sent or fee paid.

## Immediate recommendation

Use the readable manuscript and provisional IEEE conference source for the
teacher's review. Obtain the actual submission notice before choosing a portal
or paying a fee. Ask the teacher to identify whether the deadline concerns an
initial manuscript, an accepted paper's final version, registration or an internal
college submission. These are different stages; a matching date does not identify
the event. The conference must then supply its paper size, page cap, anonymity
rules and any supplemental-material restrictions.

The date and rupee amount alone do not justify selecting an IEEE journal.
For example, IEEE Access currently lists **US$2,160 plus applicable taxes** as its
article processing charge. That is a publication charge and does not verify the
reported ₹1,000 penalty. [IEEE Access fee schedule](https://ieeeaccess.ieee.org/about/article-processing-charges/)

## Rules that apply to the current writing work

IEEE's journal and conference author guides recommend a single-paragraph,
self-contained abstract of no more than 250 words, without abbreviations,
citations, footnotes or equations. They recommend a small set of keywords.
The revised abstract explains the storage idea, comparison, observed outcome and
limitations without requiring the reader to know the fitting terminology.
[Journal article structure](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-the-text-of-your-article/structure-your-article/),
[conference paper structure](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/structure-your-paper/)

IEEE provides conference templates through its Author Center, and its Template
Selector supports specific publications. The new `ieee-conference.tex` uses
`IEEEtran` in conference mode as a provisional starting point. This is not a
claim that it satisfies an unidentified conference. Keep the class-controlled
layout; do not force a page limit by shrinking fonts or margins.
[Conference authoring tools](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/),
[IEEE author tools and Template Selector](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/)

IEEE requires acknowledgment of generated content, identification of the system
and the affected sections, and an explanation of its use. Its policy distinguishes
ordinary grammar editing. This project's assistance included drafting, restructuring
and formatting scripts, so the draft identifies OpenAI Codex and describes those
uses. Humanizer is editing guidance; it does not remove the need for an accurate
disclosure. The declaration does not assert that the author or teacher has already
completed a final scientific review.
[IEEE conference submission policy](https://conferences.ieeeauthorcenter.ieee.org/author-ethics/guidelines-and-policies/submission-policies/)

## Possible journal routes if the teacher intended a journal

| Publication | Evidence of topic fit | Assessment for this paper |
|---|---|---|
| IEEE Access | Broad engineering/computing scope; expressly recognizes rigorously conducted negative-result articles | A reasonable option to discuss for a bounded empirical/negative study after resolving its scientific weaknesses. It is not chosen as the teacher's intended venue. |
| IEEE Transactions on Artificial Intelligence | Official call includes artificial-intelligence methodology, applications and natural-language processing | Topic overlap exists. The current paper needs a clearer validated contribution; a scope match does not predict acceptance. |

The route assessments are editorial judgments. IEEE Access requires adequate
technical methods and conclusions supported by data; a negative result still
needs rigorous evidence. [Access acceptance requirements](https://ieeeaccess.ieee.org/authors/preparing-your-article/),
[Access article types](https://ieeeaccess.ieee.org/authors/submission-guidelines/),
[official Transactions on Artificial Intelligence scope](https://cis.ieee.org/images/files/Publications/TAI/Call_for_Papers.pdf)

If Access is selected, its checklist requires its own template, matching source
and PDF, populated public ORCID for the submitting author, author biographies,
and 3-10 submission keywords. Its recommended length is below 20 pages; longer
manuscripts can require an inquiry. Affiliation, contact details, ORCID and a
biography have not been invented. [Access submission checklist](https://ieeeaccess.ieee.org/authors/submission-guidelines/)

The TAI author-information page could not be retrieved in this session; detailed
TAI fees, page limits and portal instructions are therefore not represented as
verified. We did not reuse deadlines from old special-issue announcements.

## Scientific issues to resolve before an external submission

The current manuscript supports an exploratory study of metric disagreement.
Its preserved controls and coverage impose specific limits:

1. **Numerical checks:** strict factorization-readiness checks failed. Before
   attributing every downstream difference to the fitting objective, explain or
   resolve the numerical discrepancy under a separately authorized protocol.
   An exact identity split does not validate SVD-based factor preparation.
2. **Contribution:** describe the observed likelihood/drift disagreement and
   matched-budget controls clearly. Do not claim a new whitening algorithm,
   universal preservation, denoising, lossless quantization or general speedup.
3. **Coverage:** Qwen has its primary test comparison but only 10/18 validation
   frontier records. The practical cohort has two seeds and no added BF16 drift
   measurement. Neither missing data nor reused measurements should be presented
   as independent confirmation.
4. **Generalization:** the corpus has already influenced the study. Fresh-corpus,
   downstream-task and generated-text confirmation are unmeasured. Broader claims
   require separately planned evidence; this edit authorizes no new experiment.
5. **Release review:** the author and teacher still need to inspect the equations,
   references, possible retractions, claim scope and source-to-PDF fidelity. No
   external similarity, reference-validation or IEEE PDF-compliance service was
   run, and no acceptance or submission-readiness certification is claimed.

These are assessments of the repository's saved evidence, not extra requirements
invented as universal IEEE rules. A near deadline does not supply the missing
evidence. The author may instead discuss a narrower preliminary-study submission
with the teacher, if the actual venue accepts that type of work.

## Files to review now

- `abstract.md`: revised accessible abstract.
- `manuscript.pdf`: readable two-column review rendering with current results.
- `ieee-conference.tex`: provisional official-class conference source.
- `manuscript.tex`: independently rendered reference-style source.
- `editorial-audit.json`: explicit prose edits and preservation checks.

The review PDF is generated independently of the TeX. It is not presented as a
camera-ready PDF compiled from `ieee-conference.tex`; the exact IEEE source must
be compiled/exported and checked under the chosen venue's rules before upload.
Do not treat the successful local compiler diagnostic as IEEE PDF validation.

Humanizer version 3.1.0 was read from the user-requested
[blader/humanizer skill](https://github.com/blader/humanizer/blob/main/SKILL.md).
It was applied to clarify prose and remove repetitive framing. Tables, equations,
existing research references and uncertainty statements were preserved. No text
detector was used and no claim about evading one is made.
