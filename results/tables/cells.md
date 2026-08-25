| cell | split | classes | cap | n_train | as-reported macro-F1 (own test) | gold defect-F1 (common) | gold 9-class macro-F1 | lot_share | dup_rate |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| A1-B1-C1 | original | 9 classes | no cap | 43,555 | 0.440 ± 0.002 | 0.775 ± 0.004 | 0.659 ± 0.005 | 0.00 | 0.022 |
| A1-B1-C2 | original | 9 classes | cap 5000/class | 12,952 | 0.546 ± 0.006 | 0.769 ± 0.001 | 0.599 ± 0.011 | 0.00 | 0.006 |
| A1-B1-C3 | original | 9 classes | balanced (min class) | 603 | 0.443 ± 0.029 | 0.585 ± 0.016 | 0.412 ± 0.021 | 0.00 | 0.006 |
| A1-B2-C1 | original | 8 defect classes | no cap | 14,111 | 0.575 ± 0.008 | 0.773 ± 0.002 | — | 0.00 | 0.010 |
| A1-B2-C2 | original | 8 defect classes | cap 5000/class | 11,698 | 0.581 ± 0.007 | 0.772 ± 0.006 | — | 0.00 | 0.010 |
| A1-B2-C3 | original | 8 defect classes | balanced (min class) | 573 | 0.481 ± 0.010 | 0.581 ± 0.004 | — | 0.00 | 0.007 |
| A2-B1-C1 | random | 9 classes | no cap | 110,668 | 0.843 ± 0.004 | 0.827 ± 0.004 | 0.835 ± 0.003 | 0.98 | 0.024 |
| A2-B1-C2 | random | 9 classes | cap 5000/class | 18,138 | 0.831 ± 0.005 | 0.831 ± 0.001 | 0.740 ± 0.002 | 0.82 | 0.004 |
| A2-B1-C3 | random | 9 classes | balanced (min class) | 856 | 0.670 ± 0.017 | 0.600 ± 0.005 | 0.477 ± 0.005 | 0.20 | 0.006 |
| A2-B2-C1 | random | 8 defect classes | no cap | 16,333 | 0.845 ± 0.007 | 0.829 ± 0.004 | — | 0.80 | 0.005 |
| A2-B2-C2 | random | 8 defect classes | cap 5000/class | 14,138 | 0.827 ± 0.003 | 0.823 ± 0.008 | — | 0.77 | 0.005 |
| A2-B2-C3 | random | 8 defect classes | balanced (min class) | 761 | 0.719 ± 0.009 | 0.610 ± 0.007 | — | 0.21 | 0.003 |
| A3-B1-C1 | lot-group | 9 classes | no cap | 110,665 | 0.836 ± 0.004 | 0.830 ± 0.006 | 0.839 ± 0.005 | 0.00 | 0.024 |
| A3-B1-C2 | lot-group | 9 classes | cap 5000/class | 18,138 | 0.835 ± 0.007 | 0.829 ± 0.003 | 0.738 ± 0.004 | 0.00 | 0.005 |
| A3-B1-C3 | lot-group | 9 classes | balanced (min class) | 857 | 0.650 ± 0.021 | 0.605 ± 0.006 | 0.477 ± 0.011 | 0.00 | 0.009 |
| A3-B2-C1 | lot-group | 8 defect classes | no cap | 16,333 | 0.823 ± 0.003 | 0.829 ± 0.008 | — | 0.00 | 0.006 |
| A3-B2-C2 | lot-group | 8 defect classes | cap 5000/class | 14,138 | 0.829 ± 0.005 | 0.827 ± 0.003 | — | 0.00 | 0.005 |
| A3-B2-C3 | lot-group | 8 defect classes | balanced (min class) | 761 | 0.723 ± 0.038 | 0.616 ± 0.006 | — | 0.00 | 0.012 |
| A4-B1-C1 | A4 | 9 classes | no cap | 110,665 | 0.571 ± 0.026 | 0.826 ± 0.003 | 0.830 ± 0.002 | 0.00 | 0.000 |
