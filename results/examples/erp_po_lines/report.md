# Data quality report - erp_po_lines

- **Batch**: `erp-po-lines-2024Q4`
- **Rows**: 89,649  |  **Columns**: 12
- **Generated**: 2026-09-18T17:23:23+00:00
- **Profile**: curated expectation suite

## Quality score

**99.8 / 100 (grade A)**

Weighted geometric mean across dimensions. The arithmetic mean would read 99.8; the gap is the penalty for uneven quality across dimensions.

| Dimension | Score | Weight | Rules | Failed |
| --- | ---: | ---: | ---: | ---: |
| completeness | 99.9 | 0.25 | 12 | 1 |
| validity | 99.9 | 0.25 | 19 | 3 |
| consistency | 99.4 | 0.20 | 6 | 4 |
| uniqueness | 100.0 | 0.15 | 3 | 0 |
| plausibility | 99.7 | 0.15 | 3 | 3 |

## Issues

9 open issues: 0 critical, 6 high, 2 medium, 1 low.

### Top 9 by priority

| # | Severity | Dimension | Rule | Records | Reason | Recommended action |
| ---: | --- | --- | --- | ---: | --- | --- |
| 1 | high | completeness | `expect_column_values_to_not_be_null.unit_price` | 1,076 | 'unit_price' is 1.20% null (tolerance 0.00%). 1,076 of 89,649 rows checked failed (1.20%). | Trace the null rows back to the source system and make the field mandatory at entry; quarantine affected records until the value is supplied. |
| 2 | high | validity | `expect_column_values_to_be_between.unit_price` | 538 | 'unit_price' must fall within [0.01, None]. 538 of 88,573 rows checked failed (0.61%). | Reject the out-of-range values at ingestion and add a bounds check to the source form; review whether a sign or unit error is involved. |
| 3 | high | validity | `expect_column_values_to_be_between.quantity` | 359 | 'quantity' must fall within [0.01, None]. 359 of 89,201 rows checked failed (0.40%). | Reject the out-of-range values at ingestion and add a bounds check to the source form; review whether a sign or unit error is involved. |
| 4 | high | validity | `expect_column_values_to_be_of_type.quantity` | 448 | 'quantity' must parse as float. 448 of 89,649 rows checked failed (0.50%). | Fix the export locale or column typing at the source; values arriving as text will not aggregate and will silently drop out of downstream sums. |
| 5 | high | consistency | `consistency.material_id_exists_in_material_master` | 359 | Every line must reference a material that exists in the master. 359 of 89,649 rows checked failed (0.40%). | Create the missing material master records or correct the references; these lines cannot be received or invoiced as they stand. |
| 6 | high | consistency | `consistency.net_value_matches_quantity_times_price` | 1,669 | Net value must equal ordered quantity times unit price. 1,669 of 88,129 rows checked failed (1.89%). | Recompute net value from quantity times unit price, or correct whichever of the three fields was mis-keyed. Invoice matching will fail on these lines. |
| 7 | medium | consistency | `consistency.uom_matches_material_master` | 715 | Line unit of measure must match the material master base unit. 715 of 89,290 rows checked failed (0.80%). | Align the line unit of measure with the material master, or convert the quantity. A unit mismatch silently corrupts every quantity aggregate. |
| 8 | medium | consistency | `consistency.receipt_not_greater_than_order` | 359 | Goods receipt cannot exceed the ordered quantity. 359 of 89,201 rows checked failed (0.40%). | Investigate over-delivery: either the receipt was posted against the wrong line or the order quantity was reduced after the fact. |
| 9 | low | plausibility | `expect_column_values_to_be_plausible.quantity` | 905 | 'quantity' values beyond 4.0 robust standard deviations of the median. 905 of 89,201 rows checked failed (1.01%). | Review the flagged values manually. These are statistical outliers, not confirmed errors, and some will be legitimate extremes. |

## All rules evaluated

| Rule | Dimension | Checked | Failed | Score |
| --- | --- | ---: | ---: | ---: |
| `expect_column_values_to_not_be_null.currency` | completeness | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.delivery_date` | completeness | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.goods_receipt_qty` | completeness | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.invoice_qty` | completeness | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.material_id` | completeness | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.net_value` | completeness | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.po_id` | completeness | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.po_line_id` | completeness | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.position` | completeness | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.quantity` | completeness | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_not_be_null.unit_price` | completeness | 89,649 | 1,076 | 0.9880 |
| `expect_column_values_to_not_be_null.uom` | completeness | 89,649 | 0 | 1.0000 |
| `consistency.invoice_not_greater_than_receipt` | consistency | 89,649 | 0 | 1.0000 |
| `consistency.material_id_exists_in_material_master` | consistency | 89,649 | 359 | 0.9960 |
| `consistency.net_value_matches_quantity_times_price` | consistency | 88,129 | 1,669 | 0.9811 |
| `consistency.po_id_exists_in_purchase_orders` | consistency | 89,649 | 0 | 1.0000 |
| `consistency.receipt_not_greater_than_order` | consistency | 89,201 | 359 | 0.9960 |
| `consistency.uom_matches_material_master` | consistency | 89,290 | 715 | 0.9920 |
| `expect_column_values_to_be_plausible.net_value` | plausibility | 89,649 | 2 | 1.0000 |
| `expect_column_values_to_be_plausible.quantity` | plausibility | 89,201 | 905 | 0.9899 |
| `expect_column_values_to_be_plausible.unit_price` | plausibility | 88,573 | 13 | 0.9999 |
| `expect_column_values_to_be_unique.po_line_id` | uniqueness | 89,649 | 0 | 1.0000 |
| `expect_primary_key_to_be_unique` | uniqueness | 89,649 | 0 | 1.0000 |
| `expect_table_rows_to_be_unique` | uniqueness | 89,649 | 0 | 1.0000 |
| `expect_column_dates_to_be_between.delivery_date` | validity | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_be_between.goods_receipt_qty` | validity | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_be_between.invoice_qty` | validity | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_be_between.net_value` | validity | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_be_between.position` | validity | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_be_between.quantity` | validity | 89,201 | 359 | 0.9960 |
| `expect_column_values_to_be_between.unit_price` | validity | 88,573 | 538 | 0.9939 |
| `expect_column_values_to_be_in_set.currency` | validity | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_be_in_set.uom` | validity | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.delivery_date` | validity | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.goods_receipt_qty` | validity | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.invoice_qty` | validity | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.net_value` | validity | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.position` | validity | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_be_of_type.quantity` | validity | 89,649 | 448 | 0.9950 |
| `expect_column_values_to_be_of_type.unit_price` | validity | 88,573 | 0 | 1.0000 |
| `expect_column_values_to_match_regex.material_id` | validity | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_match_regex.po_id` | validity | 89,649 | 0 | 1.0000 |
| `expect_column_values_to_match_regex.po_line_id` | validity | 89,649 | 0 | 1.0000 |
