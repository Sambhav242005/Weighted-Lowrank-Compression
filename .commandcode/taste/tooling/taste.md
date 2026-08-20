# Tooling Preferences

- Uses the agent-skills ecosystem (`npx skills find` / `npx skills add`) to discover and install specialized skills for a task (e.g., the academic-paper skill). Confidence: 0.7
- Prefers standalone, self-contained HTML reports with charts, graphs, and tables for presenting experiment results. Confidence: 0.6
- Uses git for version control and expects completed work to be committed and pushed (explicitly asks for commit + push when work is done). Confidence: 0.7
- Prefers generating Microsoft Word (.docx) documents programmatically via python-docx (PowerShell + Python scripts) rather than manual editing. Confidence: 0.7
- Prefers matching a source document's exact styles, section layout, and table borders when generating output, and cleaning up throwaway scripts after the task. Confidence: 0.7
- Before deleting, moving, or making destructive file changes, checks what is git-tracked, sizes, and cross-references (grep) so nothing important is lost or broken. Confidence: 0.7
- Prefers matplotlib for generating figures/visuals (e.g., charts for research papers). Confidence: 0.7
