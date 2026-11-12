# ragbench evaluation

| chunking | config | k | recall@k | MRR@k | citation P | citation R | answer F1 | abstain | grounded |
|---|---|---|---|---|---|---|---|---|---|
| fixed | overlap=60, size=250 | 3 | 1.000 | 1.000 | 0.808 | 1.000 | 0.367 | 1.000 | 0.862 |
| fixed | overlap=60, size=250 | 5 | 1.000 | 1.000 | 0.808 | 1.000 | 0.367 | 1.000 | 0.862 |
| paragraph | max_chars=600 | 3 | 1.000 | 1.000 | 0.782 | 1.000 | 0.364 | 1.000 | 0.858 |
| paragraph | max_chars=600 | 5 | 1.000 | 1.000 | 0.782 | 1.000 | 0.364 | 1.000 | 0.858 |
| fixed | overlap=80, size=400 | 3 | 1.000 | 1.000 | 0.769 | 1.000 | 0.368 | 1.000 | 0.856 |
| fixed | overlap=80, size=400 | 5 | 1.000 | 1.000 | 0.769 | 1.000 | 0.368 | 1.000 | 0.856 |
