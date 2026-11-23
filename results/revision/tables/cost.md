| component | model | threads | n_ref | us_per_flow | recall_at5 |
|---|---|---|---|---|---|
| detector | LGBM | 1 |  | 13.65 |  |
| detector | RF | 1 |  | 17.45 |  |
| detector | MLP | 1 |  | 2.27 |  |
| detector | LR | 1 |  | 1.09 |  |
| knn | brute | 1 | 1000.00 | 5.44 |  |
| knn | kd_tree | 1 | 1000.00 | 59.91 |  |
| knn | ball_tree | 1 | 1000.00 | 36.85 |  |
| knn | hnsw | 1 | 1000.00 | 13.19 | 1.00 |
| knn | brute | 1 | 5000.00 | 25.61 |  |
| knn | kd_tree | 1 | 5000.00 | 178.28 |  |
| knn | ball_tree | 1 | 5000.00 | 211.95 |  |
| knn | hnsw | 1 | 5000.00 | 17.20 | 0.99 |
| knn | brute | 1 | 20000.00 | 87.95 |  |
| knn | kd_tree | 1 | 20000.00 | 857.15 |  |
| knn | ball_tree | 1 | 20000.00 | 990.41 |  |
| knn | hnsw | 1 | 20000.00 | 27.95 | 0.98 |
| knn | brute | 1 | 50000.00 | 235.01 |  |
| knn | kd_tree | 1 | 50000.00 | 1919.12 |  |
| knn | ball_tree | 1 | 50000.00 | 3036.09 |  |
| knn | hnsw | 1 | 50000.00 | 34.85 | 0.87 |
| knn | brute | 1 | 100000.00 | 466.74 |  |
| knn | kd_tree | 1 | 100000.00 | 3511.74 |  |
| knn | ball_tree | 1 | 100000.00 | 7015.40 |  |
| knn | hnsw | 1 | 100000.00 | 56.36 | 0.96 |
| conformal+fisher |  | 1 |  | 0.31 |  |
| detector | LGBM | 4 |  | 4.75 |  |
| detector | RF | 4 |  | 5.98 |  |
| detector | MLP | 4 |  | 1.93 |  |
| detector | LR | 4 |  | 1.05 |  |
| knn | brute | 4 | 1000.00 | 1.53 |  |
| knn | kd_tree | 4 | 1000.00 | 59.22 |  |
| knn | ball_tree | 4 | 1000.00 | 54.21 |  |
| knn | hnsw | 4 | 1000.00 | 3.39 | 1.00 |
| knn | brute | 4 | 5000.00 | 6.62 |  |
| knn | kd_tree | 4 | 5000.00 | 118.81 |  |
| knn | ball_tree | 4 | 5000.00 | 253.69 |  |
| knn | hnsw | 4 | 5000.00 | 5.26 | 0.99 |
| knn | brute | 4 | 20000.00 | 23.34 |  |
| knn | kd_tree | 4 | 20000.00 | 587.19 |  |
| knn | ball_tree | 4 | 20000.00 | 993.98 |  |
| knn | hnsw | 4 | 20000.00 | 7.26 | 0.98 |
| knn | brute | 4 | 50000.00 | 57.29 |  |
| knn | kd_tree | 4 | 50000.00 | 939.37 |  |
| knn | ball_tree | 4 | 50000.00 | 1988.54 |  |
| knn | hnsw | 4 | 50000.00 | 8.80 | 0.87 |
| knn | brute | 4 | 100000.00 | 128.15 |  |
| knn | kd_tree | 4 | 100000.00 | 1839.15 |  |
| knn | ball_tree | 4 | 100000.00 | 3797.83 |  |
| knn | hnsw | 4 | 100000.00 | 9.12 | 0.96 |
| conformal+fisher |  | 4 |  | 0.29 |  |
