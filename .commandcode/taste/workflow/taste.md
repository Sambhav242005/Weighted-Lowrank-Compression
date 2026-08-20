# Workflow Preferences

- Values reproducibility in research code: pin model/dataset revisions, use fixed seeds, write new runs to unique per-run directories, and preserve legacy scripts/artifacts. Confidence: 0.7
- Prefers test-driven, offline-safe, incremental development: small changes, pytest configuration, deterministic tests, and never overwriting results/. Confidence: 0.7
- Prefers to read code and run tests first to understand a project before proposing an improvement plan. Confidence: 0.6
- Wants agent-readable Markdown files for the project plan and testing so agents can track experiments and keep improving automatically. Confidence: 0.7
- Prefers preserving existing drafts/artifacts and narrowing claims in place rather than deleting or overwriting prior work. Confidence: 0.6
- When producing a document from a template, wants the output written to a new file and the original template left untouched. Confidence: 0.8
- Prefers keeping the repository tidy and organized: consolidates scattered root files into a docs/ folder and removes empty cache/junk directories during cleanup. Confidence: 0.6
- Before destructive or irreversible operations (deleting/moving files), confirms scope with the user via explicit options and flags conflicts with project rules (e.g., AGENTS.md) rather than proceeding silently. Confidence: 0.7
- Expects the agent to re-verify its own edits (re-read the file, grep for leftover markers, confirm final structure/line count) and report the resulting cleanup rather than assuming the task is complete after a first pass. Confidence: 0.55
