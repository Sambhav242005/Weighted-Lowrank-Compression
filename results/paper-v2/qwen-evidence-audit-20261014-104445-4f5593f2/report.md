# Qwen immutable evidence audit

State: passed

18 records: eight native test and ten validation. Ten source rows preserved exactly.

Cumulative charged seconds: 7200.173392762; cooperative cap overshoot 0.173392762 seconds.

Missing scopes: [('validation', 3, 'baseline', 37), ('validation', 3, 'solver', 37), ('validation', 4, 'baseline', 11), ('validation', 4, 'solver', 11), ('validation', 4, 'baseline', 23), ('validation', 4, 'solver', 23), ('validation', 4, 'baseline', 37), ('validation', 4, 'solver', 37)]

- Qwen validation cohort incomplete; no restart or budget borrowing.
- Retained old running status is immutable reboot evidence; sidecar records interruption.
- Reused rows are not an independent replication.
- CPU file audit does not deserialize tensors or re-evaluate predictions.

Errors: []
