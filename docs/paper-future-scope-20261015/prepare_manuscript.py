"""Remove paper appendices and add proposed work; retain all measured results."""
import difflib
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PREVIOUS=ROOT/'docs/paper-math-readable-20261015'

FUTURE_SCOPE='''## 8. Future Scope

### 8.1 Compression with a small neural network

A possible extension is to replace a selected dense projection with a small bottleneck neural network. Its input would pass through a narrower hidden layer before returning to the original output dimension, allowing a compact mapping to be learned from the original model's activations. Transformer knowledge distillation provides a related precedent for training smaller students from larger teachers [@jiao2020tinybert]. The proposed projection-level replacement would still require evaluation in the models and compression setting studied here.

Another option is to keep the low-rank factors and train a small network to correct their output error. The factor rank would need to leave enough room for this correction network within the matched storage budget. Both options should count all network weights and biases, together with any retained factors. A reduction in local error would need to improve held-out predictions and control residual drift when the replacements are composed across layers. Neither option has been implemented or tested in this study.

### 8.2 Numerical checks and broader evaluation

Further work should first resolve the failed numerical controls and complete the missing Qwen validation measurements. Controlled comparisons could then test whether matching the output of a group of layers reduces the downstream drift seen after composition. Fresh evaluation data, downstream tasks and generated-text tests would help assess whether any improvement extends beyond the text already used in this project. The model, data splits, fitting budget and decision criteria should be fixed before each comparison, with replication across calibration seeds.

### 8.3 Storage and execution cost

The small-network proposals also need measurements of complete checkpoint size, peak memory and warmed-up inference latency. Fused factor execution and rank allocation informed by runtime cost are further candidates for study. Each must be compared with matched low-rank baselines under the same workloads, since fewer stored weights alone do not establish a practical speed improvement.

'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=HERE/'manuscript.source.md'
    if output.exists():
        raise FileExistsError('Preserve the existing authored edition')
    original=(PREVIOUS/'manuscript.md').read_text(encoding='utf-8')
    assert original.count('## Appendix A.')==1
    before=original.split('## Appendix A.',1)[0].rstrip()+'\n'
    revised=before.replace('Appendix uncertainty therefore describes',
        'These intervals therefore describe')
    old='Every main table is reconstructed from the frozen evidence, with its source run identified in Appendix A and exact extraction fields retained in the revision manifest.'
    new="Every main table is reconstructed from the frozen evidence. The [frozen evidence report](../../results/paper-v2/revised-report-20261014-151432-07c08d61/report.md) identifies the source runs, and the revision manifest records the exact extraction fields. Selected paired intervals remain in the repository's [evidence file](report.json)."
    assert revised.count(old)==1
    revised=revised.replace(old,new)
    assert revised.count('## 8. Conclusion')==1
    revised=revised.replace('## 8. Conclusion',FUTURE_SCOPE+'## 9. Conclusion')
    revised=revised.replace('Methods, Discussion and Conclusion,',
        'Methods, Discussion, Future Scope and Conclusion,')
    assert not re.search('appendi',revised,re.I)
    # All measured main-table values and raw mathematical expressions survive.
    tables=lambda s:[line for line in s.splitlines() if line.startswith('|')]
    math=lambda s:re.findall(r'\$([^$]+)\$',s)
    assert tables(before)==tables(revised)
    assert math(before)==math(revised)
    original_keys=set(re.findall(r'\[@([^\]]+)\]',before))
    revised_keys=set(re.findall(r'\[@([^\]]+)\]',revised))
    assert revised_keys-original_keys=={'jiao2020tinybert'}
    assert not original_keys-revised_keys
    output.write_text(revised,encoding='utf-8',newline='\n')
    patch=''.join(difflib.unified_diff(original.splitlines(keepends=True),
        revised.splitlines(keepends=True),fromfile='previous/manuscript.md',
        tofile='current/manuscript.source.md'))
    (HERE/'editorial.patch').write_text(patch,encoding='utf-8',newline='\n')
    audit={'source_sha256':sha(PREVIOUS/'manuscript.md'),
        'authored_sha256':sha(output),'appendix_a_removed':True,
        'appendix_b_renderer_removal_required':True,'appendix_cross_references_removed':True,
        'future_scope_section':'8','conclusion_section':'9',
        'measured_main_table_lines_unchanged':True,'raw_math_expressions_unchanged':True,
        'existing_citations_preserved':sorted(original_keys),
        'new_citation':'jiao2020tinybert','new_network_idea_status':'Proposed; not implemented or evaluated',
        'research_experiments_run':False}
    (HERE/'editorial-audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
