# Model Card — MTHL prototype

## Intended use

Prioritise publicly disclosed procurement processes for human review by identifying records that are unusual relative to a historical reference distribution. The system is an exploratory research prototype for academic use.

## Out-of-scope use

Do not use the score as a fraud probability, legal conclusion, automatic supplier debarment criterion, procurement eligibility decision, or sole basis for an investigation. Do not make allegations against named suppliers or buyers from the model score alone.

## Model family

Default: scikit-learn Isolation Forest with a fitted preprocessing pipeline. Structured numerical fields use median imputation with missingness indicators and scaling; categorical fields use most-frequent imputation and one-hot encoding; tender title/description text uses TF-IDF word unigrams and bigrams. The raw anomaly score is converted to an empirical percentile against the training reference distribution. The API risk bands are operational review queues, not probability thresholds.

Optional: a custom non-negative PU logistic estimator may be selected if verified positive findings, publication timestamps and a defensible positive-class prior exist. Its output remains a ranking score unless calibration is independently demonstrated.

## Training and evaluation

Raw release histories are scored at the first tender-stage publication; compiled-only snapshots are scored at their latest compiled process-state date and cannot substantiate an early-warning-at-tender claim. The evaluation fit uses chronological 70/15/15 partitions based on unique prediction timestamps. The final deployment artifact is refitted on the complete frozen snapshot only after the holdout scoring step. At present, the bundled artifact was fitted only on a synthetic software-test fixture. No real PPIP empirical model has been trained in this archive.

## Data and known limitations

The target population is Kenyan public procurement as observed in the PPIP/OCDS publication. The public registry notes incomplete data coverage and no reported tenderer objects for the current Kenya snapshot. Buyer/supplier identifiers and lifecycle-stage records require auditing. Verified positive findings are incomplete and selected; unlabelled records are not confirmed negatives.

## Fairness and impact

Procurement procedures vary legitimately by market, buyer, category, urgency and location. A high anomaly score can reflect unusual but lawful activity, a small market or publication gaps. Monitor performance and data coverage by buyer type, county, category and procurement method when legitimate labels or expert review are available. Avoid treating missingness as misconduct. Require human review and provide an appeal/correction route before any operational follow-up.

## Monitoring

The included KS/PSI monitor detects selected input-distribution changes. It does not detect concept drift or prove that model performance has degraded. Performance and subgroup monitoring require valid later outcomes and an appropriately approved evaluation process.

## Limitations

- The packaged model and reports are synthetic demonstration assets only.
- No actual PPIP export or independent audit/sanction label file was provided with the capstone attachments.
- Tender text is represented by TF-IDF; a fine-tuned Transformer/Sentence-BERT text encoder was not selected because data quality, labels and compute do not yet justify it.
- A learned heterogeneous GNN, state-space sequence model and neural hypergraph encoder are not claimed in the current implementation. Temporal and relational signals are represented with point-in-time historical features.
