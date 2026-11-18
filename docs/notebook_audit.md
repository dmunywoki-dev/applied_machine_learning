# Review of the supplied `113719_.ipynb`

The supplied notebook is a substantial exploratory and modelling attempt and documents the original concept as a Temporal Graph-Enhanced Ensemble (TGEE). It reports 162,432 releases over five OCDS JSON packages and explores procurement red flags, a buyer–supplier graph, centrality, Node2Vec, multiple classifiers, SMOTE-Tomek and SHAP. The notebook contains useful descriptive work, but its model metrics cannot be presented as evidence of procurement-fraud detection in the current form.

## Material validity issues identified

1. **Circular proxy target / target leakage.** Cell 28 defines `is_high_risk` from `flag_flash_tender AND flag_single_bidder`, `pair_recurrence_count`, `pair_value_concentration`, `flag_high_val_direct`, and `supplier_betweenness`. The modelling feature list in cell 26 includes those same indicators and graph features. The classifier therefore learns a label that is directly calculated from its inputs. Its near-perfect ROC-AUC/F1/MCC table measures how well the model reproduces the rule, not whether a tender is associated with an independently verified integrity outcome.
2. **Future information in features.** Supplier aggregates in cell 20, the award graph and centrality in cell 21, community/embedding features in cells 22–25, and buyer–supplier pair recurrence/value concentration in cell 24 are computed over the full dataframe before the random split. An earlier tender can consequently receive statistics derived from later awards and from records assigned to the test set.
3. **Record-level random split.** Cell 31 uses stratified `train_test_split` after the global features have been computed. This does not test forecasting into a later period or generalisation to buyers/suppliers not represented in training. It also does not address repeated OCDS releases for the same OCID as independent rows.
4. **Bidder/supplier role conflation.** Cell 6 counts parties with either `supplier` or `tenderer` role in `num_bidders`. These roles are not interchangeable: an award supplier is not the complete list of firms that competed. The replacement pipeline counts only explicit `tenderer` roles.
5. **Outcome timing.** Several notebook predictors (`award_value`, `num_awards`, supplier graph features, contract execution fields) are only known after tender publication. They cannot be used for a tender-time score unless the prediction task is explicitly moved to a later decision point.
6. **Interpretation overreach.** The notebook's “key findings” describe red-flag patterns and a proxy score as evidence of threshold manipulation or cartel structure. Those interpretations require independent validation and contextual review; a statistical indicator alone is not proof.
7. **Feature explanation labels.** The SHAP cells replace actual feature names with generic labels such as `Feature 0`, reducing traceability and making individual explanations difficult to audit.
8. **Denominator and data coverage checks.** The notebook reports a 2.3% flash-tender rate in one exploration output and later states 91.9% of parseable non-zero-duration tenders are under seven days. These are not necessarily the same denominator, but the divergence requires a documented denominator audit before either statistic is reported as a finding.

## How this repository changes the design

- The observation is one OCID. Raw release histories are scored at the first tender-stage publication; compiled-only snapshots are treated as latest-state review records, not first-tender forecasts.
- Post-award values and supplier identities are used only to update later historical summaries, never to score their own earlier tender.
- Every feature derived from an event history is calculated from events with publication timestamps strictly earlier than the scored tender.
- Processes with equal prediction timestamps are featurised as a batch so they cannot become one another's prior history.
- The initial modelling task is anomaly prioritisation, not supervised fraud detection, because no independently verified label file accompanied the supplied assets.
- A conditional bidder-group feature path is present, but it requires actual disclosure of at least two tenderers and is not inferred from supplier records.
- Evaluation is chronological; no supervised fraud metric is emitted in the absence of independently verified outcome labels.

This review does not invalidate the exploratory work. The notebook can still be used to inspect missingness, publication coverage and candidate feature definitions, provided the denominators and temporal availability of each field are checked.
