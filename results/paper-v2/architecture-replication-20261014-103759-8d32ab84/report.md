# Architecture confirmation

Eight full native-test and eighteen validation-frontier records completed. Attention-only targets; all decoder blocks contribute drift. BF16 confirmation transfers Gemma-selected settings without retuning. Compare within-model teacher-relative quality; do not rank raw PPL across tokenizers. Validation ranks use divisors 2/3/4 and do not retune frozen test settings. Each teacher-relative value uses the corresponding split's teacher. Readiness is validation-probe-only, with no practical hardware gate, so results remain exploratory.

| Split | Divisor | Rank | Method | Seed | NLL | PPL / same-split teacher PPL | Teacher KL | Mean block drift | Target parameter savings | Whole-model parameter savings | Factor file bytes |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test | None | None | teacher | 11 | 3.998735561 | 1.000000 | 0 | 0 | 0.000000 | 0.000000 | None |
| test | None | 725 | svd | 11 | 3.852431979 | 0.863895 | 0.9750141 | 0.237778731 | 0.291992 | 0.006279 | 35639807 |
| test | None | 725 | baseline | 11 | 4.078895427 | 1.083460 | 0.186381077 | 0.0367447259 | 0.291992 | 0.006279 | 35639871 |
| test | None | 725 | solver | 11 | 3.846287095 | 0.858603 | 0.628231489 | 0.138573675 | 0.291992 | 0.006279 | 35639871 |
| test | None | 725 | baseline | 23 | 4.033142051 | 1.035005 | 0.178240255 | 0.0356589047 | 0.291992 | 0.006279 | 35639871 |
| test | None | 725 | solver | 23 | 3.825618212 | 0.841039 | 0.619261516 | 0.139073598 | 0.291992 | 0.006279 | 35639871 |
| test | None | 725 | baseline | 37 | 4.057176407 | 1.060182 | 0.179688534 | 0.0352375519 | 0.291992 | 0.006279 | 35639871 |
| test | None | 725 | solver | 37 | 3.835843044 | 0.849683 | 0.622567556 | 0.137390458 | 0.291992 | 0.006279 | 35639871 |
| validation | 2 | 1024 | baseline | 11 | 4.127814249 | 1.021306 | 0.0655320426 | 0.0133690081 | 0.000000 | 0.000000 | 50336319 |
| validation | 2 | 1024 | solver | 11 | 4.001031176 | 0.899694 | 0.23811293 | 0.0503504637 | 0.000000 | 0.000000 | 50336319 |
| validation | 2 | 1024 | baseline | 23 | 4.107909154 | 1.001178 | 0.0645385624 | 0.013126532 | 0.000000 | 0.000000 | 50336319 |
| validation | 2 | 1024 | solver | 23 | 3.995769545 | 0.894973 | 0.236249319 | 0.0501067209 | 0.000000 | 0.000000 | 50336319 |
| validation | 2 | 1024 | baseline | 37 | 4.116178031 | 1.009491 | 0.0648618651 | 0.0130808604 | 0.000000 | 0.000000 | 50336319 |
| validation | 2 | 1024 | solver | 37 | 3.994494799 | 0.893832 | 0.234875536 | 0.0496809884 | 0.000000 | 0.000000 | 50336319 |
| validation | 3 | 682 | baseline | 11 | 4.171730591 | 1.067158 | 0.205704682 | 0.0392252303 | 0.333984 | 0.007182 | 33526335 |
| validation | 3 | 682 | solver | 11 | 3.893256609 | 0.807772 | 0.718161806 | 0.156594786 | 0.333984 | 0.007182 | 33526335 |
| validation | 3 | 682 | baseline | 23 | 4.136230301 | 1.029938 | 0.201702124 | 0.0385109493 | 0.333984 | 0.007182 | 33526335 |
| validation | 3 | 682 | solver | 23 | 3.875179910 | 0.793302 | 0.709453283 | 0.156456457 | 0.333984 | 0.007182 | 33526335 |
| validation | 3 | 682 | baseline | 37 | 4.167174700 | 1.062307 | 0.20504727 | 0.0387219816 | 0.333984 | 0.007182 | 33526335 |
| validation | 3 | 682 | solver | 37 | 3.887604595 | 0.803220 | 0.708162197 | 0.154461257 | 0.333984 | 0.007182 | 33526335 |
| validation | 4 | 512 | baseline | 11 | 4.248412612 | 1.152209 | 0.354139313 | 0.0642826896 | 0.500000 | 0.010751 | 25170495 |
| validation | 4 | 512 | solver | 11 | 3.900268779 | 0.813456 | 1.09088517 | 0.247557372 | 0.500000 | 0.010751 | 25170495 |
| validation | 4 | 512 | baseline | 23 | 4.187747866 | 1.084388 | 0.342901834 | 0.0627777659 | 0.500000 | 0.010751 | 25170495 |
| validation | 4 | 512 | solver | 23 | 3.888028041 | 0.803560 | 1.08357656 | 0.248432135 | 0.500000 | 0.010751 | 25170495 |
| validation | 4 | 512 | baseline | 37 | 4.231007553 | 1.132328 | 0.351726126 | 0.0633803278 | 0.500000 | 0.010751 | 25170495 |
| validation | 4 | 512 | solver | 37 | 3.890327787 | 0.805410 | 1.08418296 | 0.245933346 | 0.500000 | 0.010751 | 25170495 |
