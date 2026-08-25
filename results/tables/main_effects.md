| metric | axis | level means | range (max−min) |
|---|---|---|---:|
| own_macro_f1 | split | original=0.511, random=0.789, lot-group=0.783 | 0.278 |
| own_macro_f1 | classes | 9 classes=0.677, 8 defect classes=0.711 | 0.034 |
| own_macro_f1 | cap | no cap=0.727, cap 5000/class=0.741, balanced (min class)=0.614 | 0.127 |
| gold_defect_f1 | split | original=0.709, random=0.753, lot-group=0.756 | 0.047 |
| gold_defect_f1 | classes | 9 classes=0.739, 8 defect classes=0.740 | 0.001 |
| gold_defect_f1 | cap | no cap=0.811, cap 5000/class=0.808, balanced (min class)=0.600 | 0.211 |

Whole-model additive-fit residual (interaction), one value per metric — not per axis:

| metric | interaction RMS | interaction max |
|---|---:|---:|
| own_macro_f1 | 0.030 | 0.086 |
| gold_defect_f1 | 0.008 | 0.016 |
