# Compressibility Profile Report

**Model:** GPT-2 Small (12 layers, 768 dim)
**Parameters per layer:** Attention=2,359,296, MLP=4,718,592

## Spectral Analysis

| Matrix | Shape | Rank | 99% Rank | 95% Rank | Decay | Rate | Eff Rank | Compression |
|--------|-------|------|----------|----------|-------|------|----------|-------------|
| layer0.attn.W_Q | 768x768 | 768 | 526 | 390 | exponential | 0.005 | 549 | 1.46x |
| layer0.attn.W_K | 768x768 | 768 | 504 | 357 | exponential | 0.006 | 517 | 1.52x |
| layer0.attn.W_V | 768x768 | 768 | 543 | 409 | exponential | 0.005 | 560 | 1.41x |
| layer0.attn.W_O | 768x768 | 768 | 287 | 202 | exponential | 0.008 | 354 | 2.68x |
| layer0.mlp.W_up | 768x3072 | 768 | 704 | 589 | exponential | 0.002 | 700 | 1.09x |
| layer0.mlp.W_down | 3072x768 | 768 | 713 | 600 | exponential | 0.002 | 702 | 1.08x |
| layer1.attn.W_Q | 768x768 | 768 | 422 | 302 | exponential | 0.007 | 465 | 1.82x |
| layer1.attn.W_K | 768x768 | 768 | 456 | 319 | exponential | 0.006 | 482 | 1.68x |
| layer1.attn.W_V | 768x768 | 768 | 540 | 403 | exponential | 0.005 | 560 | 1.42x |
| layer1.attn.W_O | 768x768 | 768 | 481 | 327 | exponential | 0.006 | 491 | 1.60x |
| layer1.mlp.W_up | 768x3072 | 768 | 722 | 621 | exponential | 0.002 | 707 | 1.06x |
| layer1.mlp.W_down | 3072x768 | 768 | 717 | 626 | exponential | 0.002 | 710 | 1.07x |
| layer2.attn.W_Q | 768x768 | 768 | 487 | 353 | exponential | 0.006 | 511 | 1.58x |
| layer2.attn.W_K | 768x768 | 768 | 508 | 369 | exponential | 0.006 | 525 | 1.51x |
| layer2.attn.W_V | 768x768 | 768 | 552 | 416 | exponential | 0.005 | 571 | 1.39x |
| layer2.attn.W_O | 768x768 | 768 | 518 | 380 | exponential | 0.005 | 540 | 1.48x |
| layer2.mlp.W_up | 768x3072 | 768 | 720 | 610 | exponential | 0.002 | 702 | 1.07x |
| layer2.mlp.W_down | 3072x768 | 768 | 715 | 607 | exponential | 0.002 | 702 | 1.07x |
| layer3.attn.W_Q | 768x768 | 768 | 543 | 408 | exponential | 0.005 | 558 | 1.41x |
| layer3.attn.W_K | 768x768 | 768 | 550 | 416 | exponential | 0.005 | 566 | 1.40x |
| layer3.attn.W_V | 768x768 | 768 | 540 | 399 | exponential | 0.005 | 556 | 1.42x |
| layer3.attn.W_O | 768x768 | 768 | 526 | 390 | exponential | 0.005 | 549 | 1.46x |
| layer3.mlp.W_up | 768x3072 | 768 | 714 | 603 | exponential | 0.002 | 702 | 1.08x |
| layer3.mlp.W_down | 3072x768 | 768 | 711 | 596 | exponential | 0.002 | 695 | 1.08x |
| layer4.attn.W_Q | 768x768 | 768 | 523 | 378 | exponential | 0.006 | 533 | 1.47x |
| layer4.attn.W_K | 768x768 | 768 | 545 | 408 | exponential | 0.005 | 558 | 1.41x |
| layer4.attn.W_V | 768x768 | 768 | 540 | 402 | exponential | 0.005 | 560 | 1.42x |
| layer4.attn.W_O | 768x768 | 768 | 513 | 379 | exponential | 0.006 | 536 | 1.50x |
| layer4.mlp.W_up | 768x3072 | 768 | 715 | 599 | exponential | 0.002 | 698 | 1.07x |
| layer4.mlp.W_down | 3072x768 | 768 | 720 | 606 | exponential | 0.002 | 700 | 1.07x |
| layer5.attn.W_Q | 768x768 | 768 | 521 | 378 | exponential | 0.006 | 536 | 1.47x |
| layer5.attn.W_K | 768x768 | 768 | 508 | 371 | exponential | 0.006 | 527 | 1.51x |
| layer5.attn.W_V | 768x768 | 768 | 534 | 392 | exponential | 0.005 | 550 | 1.44x |
| layer5.attn.W_O | 768x768 | 768 | 538 | 406 | exponential | 0.005 | 561 | 1.43x |
| layer5.mlp.W_up | 768x3072 | 768 | 715 | 599 | exponential | 0.002 | 699 | 1.07x |
| layer5.mlp.W_down | 3072x768 | 768 | 718 | 607 | exponential | 0.002 | 702 | 1.07x |
| layer6.attn.W_Q | 768x768 | 768 | 518 | 377 | exponential | 0.006 | 536 | 1.48x |
| layer6.attn.W_K | 768x768 | 768 | 535 | 399 | exponential | 0.006 | 556 | 1.44x |
| layer6.attn.W_V | 768x768 | 768 | 530 | 386 | exponential | 0.005 | 545 | 1.45x |
| layer6.attn.W_O | 768x768 | 768 | 517 | 378 | exponential | 0.005 | 538 | 1.49x |
| layer6.mlp.W_up | 768x3072 | 768 | 717 | 602 | exponential | 0.002 | 700 | 1.07x |
| layer6.mlp.W_down | 3072x768 | 768 | 718 | 608 | exponential | 0.002 | 703 | 1.07x |
| layer7.attn.W_Q | 768x768 | 768 | 514 | 371 | exponential | 0.006 | 530 | 1.49x |
| layer7.attn.W_K | 768x768 | 768 | 504 | 362 | exponential | 0.006 | 522 | 1.52x |
| layer7.attn.W_V | 768x768 | 768 | 559 | 425 | exponential | 0.005 | 579 | 1.37x |
| layer7.attn.W_O | 768x768 | 768 | 533 | 397 | exponential | 0.005 | 555 | 1.44x |
| layer7.mlp.W_up | 768x3072 | 768 | 712 | 601 | exponential | 0.002 | 699 | 1.08x |
| layer7.mlp.W_down | 3072x768 | 768 | 718 | 611 | exponential | 0.002 | 705 | 1.07x |
| layer8.attn.W_Q | 768x768 | 768 | 521 | 381 | exponential | 0.006 | 538 | 1.47x |
| layer8.attn.W_K | 768x768 | 768 | 537 | 401 | exponential | 0.005 | 556 | 1.43x |
| layer8.attn.W_V | 768x768 | 768 | 563 | 432 | exponential | 0.005 | 586 | 1.36x |
| layer8.attn.W_O | 768x768 | 768 | 524 | 386 | exponential | 0.005 | 547 | 1.47x |
| layer8.mlp.W_up | 768x3072 | 768 | 714 | 605 | exponential | 0.002 | 703 | 1.08x |
| layer8.mlp.W_down | 3072x768 | 768 | 716 | 605 | exponential | 0.002 | 703 | 1.07x |
| layer9.attn.W_Q | 768x768 | 768 | 527 | 387 | exponential | 0.006 | 543 | 1.46x |
| layer9.attn.W_K | 768x768 | 768 | 534 | 397 | exponential | 0.005 | 551 | 1.44x |
| layer9.attn.W_V | 768x768 | 768 | 561 | 430 | exponential | 0.005 | 586 | 1.37x |
| layer9.attn.W_O | 768x768 | 768 | 553 | 425 | exponential | 0.005 | 581 | 1.39x |
| layer9.mlp.W_up | 768x3072 | 768 | 711 | 601 | exponential | 0.002 | 701 | 1.08x |
| layer9.mlp.W_down | 3072x768 | 768 | 715 | 605 | exponential | 0.002 | 705 | 1.07x |
| layer10.attn.W_Q | 768x768 | 768 | 518 | 375 | exponential | 0.006 | 531 | 1.48x |
| layer10.attn.W_K | 768x768 | 768 | 517 | 373 | exponential | 0.005 | 529 | 1.49x |
| layer10.attn.W_V | 768x768 | 768 | 541 | 405 | exponential | 0.005 | 563 | 1.42x |
| layer10.attn.W_O | 768x768 | 768 | 578 | 459 | exponential | 0.004 | 607 | 1.33x |
| layer10.mlp.W_up | 768x3072 | 768 | 714 | 603 | exponential | 0.002 | 701 | 1.08x |
| layer10.mlp.W_down | 3072x768 | 768 | 718 | 613 | exponential | 0.002 | 710 | 1.07x |
| layer11.attn.W_Q | 768x768 | 768 | 519 | 380 | exponential | 0.005 | 535 | 1.48x |
| layer11.attn.W_K | 768x768 | 768 | 518 | 375 | exponential | 0.005 | 532 | 1.48x |
| layer11.attn.W_V | 768x768 | 768 | 565 | 429 | exponential | 0.005 | 581 | 1.36x |
| layer11.attn.W_O | 768x768 | 768 | 550 | 416 | exponential | 0.005 | 564 | 1.40x |
| layer11.mlp.W_up | 768x3072 | 768 | 715 | 601 | exponential | 0.002 | 698 | 1.07x |
| layer11.mlp.W_down | 3072x768 | 768 | 725 | 620 | exponential | 0.002 | 713 | 1.06x |

## Statistical Properties

| Matrix | Mean | Std | Skew | Kurtosis | Entropy (bits) | Sparsity <1e-3 |
|--------|------|-----|------|----------|----------------|----------------|
| layer0.attn.W_Q | 0.0002 | 0.2387 | -0.003 | 1.168 | 4.90 | 0.004 |
| layer0.attn.W_K | 0.0000 | 0.2433 | -0.006 | 3.389 | 4.10 | 0.004 |
| layer0.attn.W_V | -0.0000 | 0.0581 | -0.016 | 1.072 | 4.44 | 0.016 |
| layer0.attn.W_O | -0.0002 | 0.1475 | -0.090 | 25.794 | 2.57 | 0.026 |
| layer0.mlp.W_up | -0.0007 | 0.1412 | 0.040 | 3.038 | 3.06 | 0.008 |
| layer0.mlp.W_down | 0.0000 | 0.0880 | -0.022 | 130.624 | 1.60 | 0.011 |
| layer1.attn.W_Q | -0.0002 | 0.1510 | 0.021 | 2.397 | 4.65 | 0.009 |
| layer1.attn.W_K | 0.0002 | 0.1590 | -0.002 | 1.816 | 4.85 | 0.007 |
| layer1.attn.W_V | 0.0000 | 0.1037 | 0.001 | 0.574 | 4.63 | 0.009 |
| layer1.attn.W_O | -0.0001 | 0.1019 | -0.645 | 150.321 | 2.09 | 0.013 |
| layer1.mlp.W_up | 0.0006 | 0.1307 | 0.001 | 1.973 | 3.68 | 0.007 |
| layer1.mlp.W_down | 0.0001 | 0.0872 | 4.526 | 705.483 | 1.08 | 0.015 |
| layer2.attn.W_Q | 0.0001 | 0.1893 | -0.004 | 1.361 | 4.81 | 0.006 |
| layer2.attn.W_K | -0.0000 | 0.1518 | 0.000 | 2.512 | 4.22 | 0.007 |
| layer2.attn.W_V | 0.0002 | 0.1050 | 0.002 | 0.563 | 4.96 | 0.010 |
| layer2.attn.W_O | -0.0000 | 0.0810 | 0.035 | 13.803 | 2.86 | 0.012 |
| layer2.mlp.W_up | -0.0051 | 0.1335 | 0.149 | 28.739 | 1.81 | 0.006 |
| layer2.mlp.W_down | 0.0002 | 0.0931 | 5.399 | 756.551 | 1.28 | 0.011 |
| layer3.attn.W_Q | -0.0002 | 0.1647 | 0.001 | 0.201 | 5.02 | 0.005 |
| layer3.attn.W_K | -0.0001 | 0.1538 | 0.012 | 1.349 | 4.13 | 0.005 |
| layer3.attn.W_V | 0.0002 | 0.0977 | 0.013 | 0.697 | 4.93 | 0.010 |
| layer3.attn.W_O | 0.0000 | 0.0841 | -0.083 | 5.684 | 3.36 | 0.012 |
| layer3.mlp.W_up | -0.0060 | 0.1295 | -0.018 | 0.539 | 3.45 | 0.006 |
| layer3.mlp.W_down | 0.0002 | 0.0918 | 5.252 | 790.536 | 0.98 | 0.011 |
| layer4.attn.W_Q | 0.0003 | 0.1704 | 0.001 | 1.510 | 4.77 | 0.005 |
| layer4.attn.W_K | 0.0001 | 0.1575 | -0.017 | 4.840 | 3.32 | 0.006 |
| layer4.attn.W_V | 0.0001 | 0.1023 | -0.001 | 0.541 | 5.12 | 0.009 |
| layer4.attn.W_O | -0.0000 | 0.0930 | 0.036 | 5.424 | 3.36 | 0.011 |
| layer4.mlp.W_up | -0.0033 | 0.1297 | 0.000 | 0.593 | 3.68 | 0.006 |
| layer4.mlp.W_down | 0.0002 | 0.0910 | 0.512 | 23.536 | 2.16 | 0.011 |
| layer5.attn.W_Q | -0.0003 | 0.1413 | -0.006 | 0.449 | 4.91 | 0.006 |
| layer5.attn.W_K | 0.0000 | 0.1361 | 0.017 | 0.694 | 4.36 | 0.006 |
| layer5.attn.W_V | -0.0001 | 0.1033 | -0.003 | 0.366 | 5.20 | 0.009 |
| layer5.attn.W_O | 0.0000 | 0.0938 | 0.022 | 3.242 | 3.39 | 0.010 |
| layer5.mlp.W_up | -0.0042 | 0.1267 | -0.009 | 0.451 | 3.81 | 0.007 |
| layer5.mlp.W_down | 0.0001 | 0.0974 | 0.190 | 5.925 | 2.85 | 0.010 |
| layer6.attn.W_Q | 0.0002 | 0.1340 | -0.004 | 0.468 | 4.66 | 0.007 |
| layer6.attn.W_K | -0.0001 | 0.1276 | 0.012 | 0.916 | 4.12 | 0.007 |
| layer6.attn.W_V | 0.0002 | 0.1185 | -0.000 | 0.303 | 5.28 | 0.008 |
| layer6.attn.W_O | 0.0000 | 0.1137 | 0.023 | 2.657 | 3.70 | 0.008 |
| layer6.mlp.W_up | -0.0028 | 0.1264 | -0.003 | 0.410 | 3.96 | 0.007 |
| layer6.mlp.W_down | 0.0001 | 0.1073 | 0.112 | 4.719 | 3.14 | 0.009 |
| layer7.attn.W_Q | -0.0005 | 0.1365 | 0.004 | 0.350 | 4.47 | 0.006 |
| layer7.attn.W_K | 0.0001 | 0.1305 | 0.010 | 0.740 | 3.95 | 0.007 |
| layer7.attn.W_V | 0.0001 | 0.1195 | -0.002 | 0.288 | 5.24 | 0.008 |
| layer7.attn.W_O | 0.0000 | 0.1139 | -0.040 | 3.357 | 3.49 | 0.008 |
| layer7.mlp.W_up | -0.0035 | 0.1264 | -0.008 | 0.207 | 4.65 | 0.006 |
| layer7.mlp.W_down | 0.0001 | 0.1187 | 0.057 | 4.598 | 2.49 | 0.008 |
| layer8.attn.W_Q | -0.0003 | 0.1301 | 0.000 | 0.257 | 4.46 | 0.006 |
| layer8.attn.W_K | 0.0002 | 0.1244 | 0.009 | 0.757 | 3.81 | 0.007 |
| layer8.attn.W_V | -0.0004 | 0.1263 | 0.005 | 0.204 | 5.37 | 0.007 |
| layer8.attn.W_O | 0.0000 | 0.1224 | 0.066 | 4.827 | 3.29 | 0.008 |
| layer8.mlp.W_up | -0.0021 | 0.1273 | -0.002 | 0.201 | 4.30 | 0.006 |
| layer8.mlp.W_down | 0.0000 | 0.1354 | 0.062 | 4.660 | 2.65 | 0.007 |
| layer9.attn.W_Q | 0.0004 | 0.1230 | -0.004 | 0.293 | 4.19 | 0.007 |
| layer9.attn.W_K | -0.0003 | 0.1189 | 0.002 | 0.892 | 3.70 | 0.007 |
| layer9.attn.W_V | -0.0003 | 0.1361 | 0.000 | 0.256 | 5.13 | 0.007 |
| layer9.attn.W_O | -0.0000 | 0.1368 | -0.003 | 2.643 | 3.94 | 0.007 |
| layer9.mlp.W_up | -0.0027 | 0.1276 | -0.005 | 0.347 | 3.44 | 0.006 |
| layer9.mlp.W_down | 0.0000 | 0.1559 | 0.018 | 3.247 | 2.65 | 0.006 |
| layer10.attn.W_Q | 0.0003 | 0.1187 | 0.006 | 0.338 | 4.20 | 0.007 |
| layer10.attn.W_K | -0.0001 | 0.1147 | 0.020 | 1.199 | 3.68 | 0.007 |
| layer10.attn.W_V | 0.0001 | 0.1446 | -0.002 | 0.244 | 5.30 | 0.006 |
| layer10.attn.W_O | -0.0000 | 0.1466 | 0.012 | 12.502 | 2.84 | 0.006 |
| layer10.mlp.W_up | -0.0032 | 0.1276 | -0.010 | 0.401 | 3.50 | 0.006 |
| layer10.mlp.W_down | 0.0000 | 0.1781 | 0.016 | 13.939 | 1.91 | 0.005 |
| layer11.attn.W_Q | -0.0000 | 0.1098 | -0.008 | 0.282 | 4.47 | 0.007 |
| layer11.attn.W_K | -0.0000 | 0.1055 | 0.004 | 0.684 | 4.19 | 0.008 |
| layer11.attn.W_V | 0.0002 | 0.1623 | -0.009 | 0.950 | 3.87 | 0.005 |
| layer11.attn.W_O | -0.0001 | 0.1819 | -0.447 | 190.101 | 1.93 | 0.006 |
| layer11.mlp.W_up | -0.0018 | 0.1300 | -0.008 | 1.011 | 3.88 | 0.006 |
| layer11.mlp.W_down | -0.0004 | 0.1982 | 0.009 | 26.870 | 2.17 | 0.005 |

## Key Findings

### Attention Output Projections (W_O)
- Average 99% rank: 510/768 (66.4% of full rank)
- Average effective rank: 535/768
- Average decay rate: 0.0054
- **Most compressible block type**

### Attention Q/K/V Projections
- Average 99% rank: 526/768 (68.4% of full rank)
- Average effective rank: 544/768

### MLP Blocks
- W_up 99% rank: 714/768 (93.0%)
- W_down 99% rank: 717/768 (93.4%)
- **Least compressible block type** — near full rank

### Compression Potential by Layer

| Layer | Best Compressible | Worst Compressible |
|-------|-------------------|--------------------|
| 0 | W_O (2.68x) | W_down (1.08x) |
| 1 | W_Q (1.82x) | W_up (1.06x) |
| 2 | W_Q (1.58x) | W_up (1.07x) |
| 3 | W_O (1.46x) | W_up (1.08x) |
| 4 | W_O (1.50x) | W_down (1.07x) |
| 5 | W_K (1.51x) | W_down (1.07x) |
| 6 | W_O (1.49x) | W_down (1.07x) |
| 7 | W_K (1.52x) | W_down (1.07x) |
| 8 | W_Q (1.47x) | W_down (1.07x) |
| 9 | W_Q (1.46x) | W_down (1.07x) |
| 10 | W_K (1.49x) | W_down (1.07x) |
| 11 | W_K (1.48x) | W_down (1.06x) |

### Conclusion

All layers show **exponential spectral decay**, confirming the Structured-Weight hypothesis for at least the spectral dimension.

**Key insight:** Attention O-projections are consistently the most compressible (~2.7x at 99% variance), while MLP blocks are near full rank. This suggests the attention output mixing is the primary bottleneck for compression.

**Next steps:**
1. Test functional substitution on W_O layers (highest compression potential)
2. Compare SVD baseline vs. Fourier/INR representations
3. Measure functional preservation (perplexity) under compression
