# Data quality report - creditcard_transactions

- **Batch**: `creditcard-latest-20k`
- **Rows**: 20,000  |  **Columns**: 31
- **Generated**: 2026-09-18T17:23:23+00:00
- **Profile**: curated expectation suite

## Quality score

**99.9 / 100 (grade A)**

Weighted geometric mean across dimensions. The arithmetic mean would read 99.9; the gap is the penalty for uneven quality across dimensions.

| Dimension | Score | Weight | Rules | Failed |
| --- | ---: | ---: | ---: | ---: |
| completeness | 100.0 | 0.30 | 31 | 0 |
| validity | 100.0 | 0.30 | 62 | 2 |
| consistency | 100.0 | 0.00 | 0 | 0 |
| uniqueness | 99.4 | 0.15 | 1 | 1 |
| plausibility | 100.0 | 0.15 | 1 | 0 |

## Anomaly detection

- Model: **autoencoder**
- Records flagged: **51** (0.26% of the batch)
- Threshold: 2.63331 (99.50th percentile of unlabelled calibration scores)
- Known positives in batch: 12, of which 8 were flagged

## Issues

52 open issues: 0 critical, 5 high, 47 medium, 0 low.

### Top 25 by priority

| # | Severity | Dimension | Rule | Records | Reason | Recommended action |
| ---: | --- | --- | --- | ---: | --- | --- |
| 1 | high | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 138.45 (100.0000% percentile, 425.3x the batch median). Largest deviations: V27 (observed 81.91 vs expected 46.31), log_amount (observed 4.24 vs expected 30.96), V28 (observed -50.36 vs expected -25.17). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 2 | high | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 37.265 (99.9950% percentile, 114.5x the batch median). Largest deviations: V27 (observed -18.21 vs expected 0.89), V8 (observed -17.49 vs expected -1.71), V21 (observed -9.08 vs expected 6.60). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 3 | high | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 36.293 (99.9900% percentile, 111.5x the batch median). Largest deviations: V27 (observed -16.83 vs expected 1.44), V8 (observed -17.80 vs expected -1.93), V21 (observed -9.25 vs expected 6.45). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 4 | high | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 20.615 (99.9850% percentile, 63.3x the batch median). Largest deviations: V27 (observed -11.52 vs expected 1.63), V28 (observed -10.66 vs expected 0.28), V21 (observed -4.87 vs expected 5.54). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 5 | high | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 20.483 (99.9800% percentile, 62.9x the batch median). Largest deviations: V27 (observed -11.46 vs expected 1.50), V28 (observed -10.66 vs expected 0.20), V21 (observed -4.89 vs expected 5.51). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 6 | medium | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 20.163 (99.9750% percentile, 61.9x the batch median). Largest deviations: V27 (observed -11.14 vs expected 0.73), V8 (observed -13.58 vs expected -2.81), V21 (observed -4.85 vs expected 5.53). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 7 | medium | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 11.631 (99.9700% percentile, 35.7x the batch median). Largest deviations: V23 (observed -17.04 vs expected -0.66), V19 (observed -2.75 vs expected 1.20), V22 (observed -4.69 vs expected -0.95). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 8 | medium | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 10.292 (99.9650% percentile, 31.6x the batch median). Largest deviations: log_amount (observed 3.68 vs expected 14.98), V20 (observed 3.35 vs expected -1.92), V28 (observed -8.86 vs expected -4.23). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 9 | medium | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 10.241 (99.9600% percentile, 31.5x the batch median). Largest deviations: V17 (observed -9.65 vs expected -0.15), V16 (observed -6.64 vs expected 0.56), V18 (observed -5.96 vs expected 0.92). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 10 | medium | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 9.2255 (99.9550% percentile, 28.3x the batch median). Largest deviations: V23 (observed -17.03 vs expected -2.36), V28 (observed 0.52 vs expected -2.47), V19 (observed -0.33 vs expected 2.62). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 11 | medium | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 7.4655 (99.9500% percentile, 22.9x the batch median). Largest deviations: V23 (observed -13.58 vs expected -0.27), V19 (observed -2.44 vs expected 0.42), V28 (observed 0.99 vs expected -1.80). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 12 | medium | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 6.6121 (99.9450% percentile, 20.3x the batch median). Largest deviations: V17 (observed -6.13 vs expected 0.71), V14 (observed -7.27 vs expected -0.64), V12 (observed -4.61 vs expected 0.17). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 13 | medium | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 6.3358 (99.9400% percentile, 19.5x the batch median). Largest deviations: V17 (observed -6.08 vs expected 0.93), V14 (observed -7.54 vs expected -1.76), V12 (observed -4.97 vs expected 0.03). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 14 | medium | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 5.7033 (99.9350% percentile, 17.5x the batch median). Largest deviations: V17 (observed -7.66 vs expected -0.04), V7 (observed -4.09 vs expected -0.36), V3 (observed -4.17 vs expected -0.44). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 15 | medium | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 5.6586 (99.9300% percentile, 17.4x the batch median). Largest deviations: V17 (observed -7.13 vs expected 0.58), V3 (observed -5.46 vs expected -1.52), V10 (observed -2.89 vs expected 0.98). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 16 | medium | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 5.6567 (99.9250% percentile, 17.4x the batch median). Largest deviations: V17 (observed -7.34 vs expected 0.39), V3 (observed -5.09 vs expected -1.09), V10 (observed -2.98 vs expected 0.80). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 17 | medium | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 5.5727 (99.9200% percentile, 17.1x the batch median). Largest deviations: V23 (observed -15.06 vs expected -5.00), V19 (observed -2.53 vs expected 1.76), V28 (observed 1.45 vs expected -2.00). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 18 | medium | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 5.1111 (99.9150% percentile, 15.7x the batch median). Largest deviations: V27 (observed 18.10 vs expected 12.78), V3 (observed -2.71 vs expected -6.32), V11 (observed -0.80 vs expected 2.80). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 19 | medium | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 4.6211 (99.9100% percentile, 14.2x the batch median). Largest deviations: V7 (observed -7.50 vs expected -1.66), V5 (observed 6.73 vs expected 1.65), V27 (observed 1.66 vs expected 5.57). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 20 | medium | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 4.5006 (99.9050% percentile, 13.8x the batch median). Largest deviations: V27 (observed -12.72 vs expected -4.71), V19 (observed -1.82 vs expected 1.61), V24 (observed -4.08 vs expected -0.92). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 21 | medium | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 4.4018 (99.9000% percentile, 13.5x the batch median). Largest deviations: V27 (observed 17.04 vs expected 12.74), V23 (observed 38.15 vs expected 33.87), V17 (observed -2.03 vs expected 2.02). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 22 | medium | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 4.2304 (99.8950% percentile, 13.0x the batch median). Largest deviations: V28 (observed 3.36 vs expected -1.92), V8 (observed -2.67 vs expected -6.34), V25 (observed 3.88 vs expected 0.46). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 23 | medium | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 3.9975 (99.8900% percentile, 12.3x the batch median). Largest deviations: log_amount (observed 2.88 vs expected 7.83), V4 (observed 6.65 vs expected 2.93), V27 (observed -3.25 vs expected 0.22). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 24 | medium | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 3.8023 (99.8850% percentile, 11.7x the batch median). Largest deviations: V27 (observed -12.44 vs expected -6.34), V19 (observed -2.78 vs expected 1.01), V28 (observed -1.34 vs expected -4.78). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |
| 25 | medium | anomaly | `anomaly.autoencoder` | 1 | Anomaly score 3.6721 (99.8800% percentile, 11.3x the batch median). Largest deviations: V25 (observed -8.26 vs expected -3.62), V28 (observed 2.58 vs expected -1.85), V7 (observed -7.11 vs expected -2.84). | Route to manual review. The record is unlike the transactions the model was trained on; confirm it before it is processed automatically. |

## All rules evaluated

| Rule | Dimension | Checked | Failed | Score |
| --- | --- | ---: | ---: | ---: |
| `expect_column_values_to_not_be_null.Amount` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.Class` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.Time` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V1` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V10` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V11` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V12` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V13` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V14` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V15` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V16` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V17` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V18` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V19` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V2` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V20` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V21` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V22` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V23` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V24` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V25` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V26` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V27` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V28` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V3` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V4` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V5` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V6` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V7` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V8` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.V9` | completeness | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_plausible.Amount` | plausibility | 20,000 | 0 | 1.0000 |
| `expect_table_rows_to_be_unique` | uniqueness | 20,000 | 111 | 0.9944 |
| `expect_column_values_to_be_between.Amount` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.Class` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.Time` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V1` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V10` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V11` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V12` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V13` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V14` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V15` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V16` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V17` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V18` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V19` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V2` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V20` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V21` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V22` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V23` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V24` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V25` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V26` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V27` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V28` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V3` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V4` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V5` | validity | 20,000 | 1 | 1.0000 |
| `expect_column_values_to_be_between.V6` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V7` | validity | 20,000 | 1 | 1.0000 |
| `expect_column_values_to_be_between.V8` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_between.V9` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.Amount` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.Class` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.Time` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V1` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V10` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V11` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V12` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V13` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V14` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V15` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V16` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V17` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V18` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V19` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V2` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V20` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V21` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V22` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V23` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V24` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V25` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V26` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V27` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V28` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V3` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V4` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V5` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V6` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V7` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V8` | validity | 20,000 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.V9` | validity | 20,000 | 0 | 1.0000 |
