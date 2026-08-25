# WM-811K labeled set — EDA

- labeled wafers: 172,950
- lots: 10,762 (singleton lot ids: 0)
- labeled wafers per lot, median: 23

## Class counts

| class | count | share |
|---|---:|---:|
| Center | 4,294 | 2.48% |
| Donut | 555 | 0.32% |
| Edge-Loc | 5,189 | 3.00% |
| Edge-Ring | 9,680 | 5.60% |
| Loc | 3,593 | 2.08% |
| Random | 866 | 0.50% |
| Scratch | 1,193 | 0.69% |
| Near-full | 149 | 0.09% |
| none | 147,431 | 85.24% |

## Original Training/Test labels

- counts: {'Test': 118595, 'Training': 54355}
- lots containing both Training and Test wafers: **0**
- Test wafers whose lot also has Training wafers: 0.000
- lot-number range: {'Training': {'min': 1.0, 'max': 46729.0, 'median': 36921.0}, 'Test': {'min': 40328.0, 'max': 47542.0, 'median': 44422.0}}
- Training/Test runs along lot order: 16

| split | Center | Donut | Edge-Loc | Edge-Ring | Loc | Random | Scratch | Near-full | none |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Test | 832 | 146 | 2,772 | 1,126 | 1,973 | 257 | 693 | 95 | 110,701 |
| Training | 3,462 | 409 | 2,417 | 8,554 | 1,620 | 609 | 500 | 54 | 36,730 |

## Within-lot structure

- lots with >=2 defect wafers: 3,669; single-class among them: 0.515
- defect wafers living in such lots: 21,141 / 25,519

## Exact duplicates (raw maps)

- rows in duplicate groups: 6,502 (3.76%), groups: 3,250
- groups spanning >1 lot: 0.996; groups spanning Training&Test: 3,158
- rows by class: {'none': 6304, 'Edge-Loc': 67, 'Loc': 39, 'Random': 28, 'Scratch': 22, 'Near-full': 15, 'Edge-Ring': 14, 'Center': 13}

## Exact duplicates (after 64x64 resize)

- rows in duplicate groups: 6,508 (3.76%), groups: 3,252
- groups spanning >1 lot: 0.996; groups spanning Training&Test: 3,158
- rows by class: {'none': 6304, 'Edge-Loc': 67, 'Loc': 39, 'Random': 28, 'Scratch': 22, 'Edge-Ring': 20, 'Near-full': 15, 'Center': 13}

## Map shapes

- unique shapes: 346; H quantiles {'0': 15.0, '10': 25.0, '50': 33.0, '90': 44.0, '100': 212.0}; W quantiles {'0': 3.0, '10': 26.0, '50': 33.0, '90': 44.0, '100': 204.0}
- maps with H>64 or W>64 (downsampled): 2.01%
- top shapes: 25x27 (18,781), 26x26 (14,366), 30x34 (12,400), 29x26 (11,751), 27x25 (10,682), 39x37 (10,173), 33x29 (8,125), 42x44 (6,724), 44x41 (6,134), 41x33 (5,091)
