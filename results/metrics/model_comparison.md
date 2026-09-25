# Baseline vs autoencoder

Held-out temporal test window, identical features, identical 0.5% alert budget.

| model                                    |   pr_auc |   pr_auc_lift |   roc_auc |   precision |   recall |     f1 |   alerts |   true_positives |   false_positives |   false_negatives |   precision_at_100 |   review_hours |   review_hours_per_fraud_found |
|:-----------------------------------------|---------:|--------------:|----------:|------------:|---------:|-------:|---------:|-----------------:|------------------:|------------------:|-------------------:|---------------:|-------------------------------:|
| Logistic Regression (supervised ceiling) |   0.778  |      590.85   |    0.9814 |      0.2278 |   0.8533 | 0.3596 |      281 |               64 |               217 |                11 |               0.59 |          7.025 |                         0.1098 |
| Autoencoder (trained on normals)         |   0.2873 |      218.235  |    0.9204 |      0.2523 |   0.72   | 0.3737 |      214 |               54 |               160 |                21 |               0.34 |          5.35  |                         0.0991 |
| Isolation Forest (trained on normals)    |   0.0898 |       68.1664 |    0.9492 |      0.1496 |   0.4667 | 0.2265 |      234 |               35 |               199 |                40 |               0.11 |          5.85  |                         0.1671 |
| Autoencoder (fully unsupervised)         |   0.0852 |       64.7436 |    0.9437 |      0.1628 |   0.28   | 0.2059 |      129 |               21 |               108 |                54 |               0.19 |          3.225 |                         0.1536 |
| Isolation Forest (fully unsupervised)    |   0.0759 |       57.6661 |    0.949  |      0.1231 |   0.32   | 0.1778 |      195 |               24 |               171 |                51 |               0.1  |          4.875 |                         0.2031 |
| Random (floor)                           |   0.0012 |        0.9487 |    0.4618 |      0      |   0      | 0      |      291 |                0 |               291 |                75 |               0    |          7.275 |                       inf      |
