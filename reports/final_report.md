# DSA 8401 Capstone Technical Report

## Explainable Temporal and Relational Anomaly Prioritisation of Kenyan Public Procurement Processes

### Abstract

This capstone implements a reproducible machine-learning screening prototype for ranking Kenyan public-procurement processes for human review. OCDS data are ingested and reduced to one observation per contracting process. When raw release histories are supplied, the process is scored at its first tender-stage release; when only compiled releases are available, it is scored at the latest compiled process state and is not treated as a first-tender forecast. Historical buyer and supplier-concentration features are constructed only from source events published strictly before the applicable prediction time; tender text is represented with TF-IDF; and a fitted Isolation Forest provides an unsupervised anomaly-ranking baseline. The model is wrapped in a FastAPI service and packaged for Docker deployment. Because no actual PPIP export or independently verified outcome-label file was supplied with the capstone attachments, the included end-to-end run uses a synthetic OCDS-like fixture for software verification only. No real-world fraud-detection performance claim is made.

## 1. Problem and task definition

The task is procurement-process risk prioritisation: given the information public at a tender's first publication timestamp, produce a ranked screening signal that directs limited human review capacity. The output is not a determination of wrongdoing. This framing is consistent with the research proposal's distinction between documented integrity concerns, incomplete labels and review-support scores.

The unit of analysis is one OCID. The score time depends on the OCDS representation. For raw release histories it is the first tender-stage release date. For compiled snapshots it is the compiled release date, which represents the latest process state. OCDS distinguishes these: a release is an immutable update, while a compiled release is the latest merge of fields; the record's `releases` array or a versioned release preserves change history. Therefore, a compiled-snapshot-only export cannot reconstruct the exact first-tender information set. The pipeline detects this and marks compiled-snapshot observations rather than labelling them first-tender early warnings. Features from award events are used only for later observations whose prediction timestamps follow the award publication date. The target of the default model is not a hand-written “fraud” label: it is statistical unusualness relative to the training reference distribution.

## 2. Data and quality

The primary planned source is PPRA's PPIP publication in OCDS form. The OCP Registry profile describes the Kenya dataset as covering July 2018 to October 2026 and notes missing identifiers and lifecycle objects; notably, it reports no tenderer objects for the snapshot profile. The registry states that each JSONL line represents one compiled release per contracting process. Under the OCDS standard, a compiled release contains the latest field values, while raw releases or a versioned release are needed to reconstruct changes over time. These conditions make data-quality assessment part of the modelling problem, not a one-off cleaning step.

The user's original notebook states that it loaded 162,432 releases from five JSON packages. However, those raw JSON packages were not present among the supplied files in this work session, so this reported volume could not be independently reproduced or audited here. The full PPIP export was also not successfully retrieved from the public download endpoint in this environment. Consequently, the real-data ingestion, schema audit, feature availability and actual eligible OCID count remain to be run by the student after placing the source export in `data/raw/`.

For a repeatable integration test, a synthetic fixture was generated with 180 fictional processes and 360 releases. Its audit reports 180 unique OCIDs, 180 process observations and 113 synthetic processes with at least two disclosed tenderer identities. These numbers describe only the synthetic fixture and must not be cited as facts about Kenyan procurement.

## 3. Pipeline and modelling choices

The pipeline has four principal stages:

1. **Ingestion and data quality:** accepts OCDS JSON release packages or JSON Lines / compressed JSONL; retains source file hashes; validates OCID and publication timestamps; records invalid rows and field completeness; detects raw release histories versus compiled snapshots.
2. **Point-in-time feature engineering:** creates one process observation per OCID. Raw histories are scored at the first tender-stage release; compiled-only records are scored at their latest compiled state. The pipeline derives tender duration, estimated value, method/category, text and calendar features; builds buyer history and supplier-concentration summaries from events published before the prediction timestamp; and preserves missingness indicators. Equal-timestamp records are processed as a batch to avoid using one another as historical evidence.
3. **Model and evaluation:** compares the default unsupervised anomaly-ranking path under a chronological 70/15/15 split. Imputation, scaling, categorical encoding and TF-IDF vocabulary are fitted only on the training period for the evaluation fit. The deployment artifact is refitted on the frozen snapshot after the out-of-time scoring step.
4. **Serving and monitoring:** stores preprocessing and estimator together; exposes `/health` and `/score` endpoints; attaches an evidence trace; supports KS/PSI input-drift monitoring.

Isolation Forest was selected as the primary empirical baseline because independently verified negative cases were not supplied and unlabelled procurement records cannot be treated as confirmed legitimate. TF-IDF was selected as a transparent, lightweight representation for tender descriptions. A large fine-tuned Transformer, HGT/GNN and DG-Mamba-style state-space learner were not implemented: the available evidence does not yet establish reliable tenderer identities, linked outcome labels or a demonstrated benefit that justifies their additional complexity. The conditional bidder-group recurrence feature is disabled when the source does not contain sufficiently disclosed tenderer sets; it is not filled in from award-supplier records.

An optional nnPU logistic model is implemented for a later stage where verified positive findings and a defensible class prior exist. The label file must contain an OCID and the timestamp at which the finding was published. Findings not public by the training cut-off are excluded from that training fit. This is an option, not a model result reported in this run.

## 4. Evaluation and software verification

The synthetic data workflow was executed end-to-end. The generated data were ingested, quality-audited, used to fit an Isolation Forest and saved as a serialized artifact. The out-of-time portion contains 55 synthetic records (28 validation and 27 test rows) scored by the train-only evaluation fit. The report records score distribution summaries only. No AUPRC, ROC-AUC, F1, precision or recall for fraud is reported because no independent outcome labels exist.

Automated checks cover: one observation per OCID; exclusion of a process's own award from its tender-time features; strict prior-event logic; same-timestamp batching; supplier/tenderer role separation; JSONL.GZ ingestion; bounded anomaly-priority percentiles; nnPU learner numerical execution; chronological split integrity; and API health/scoring contracts. The local test suite completed with **12 passing tests**.

Dockerfile and Docker Compose definitions are included. Docker was not installed in the execution environment, so the image build and container run could not be tested here. The API code and model scoring path were exercised through Python tests.

## 5. Audit of the original notebook

The supplied `113719_.ipynb` is valuable as an exploration notebook, with red-flag construction, descriptive analysis, a buyer–supplier network and a stacked ensemble. However, its near-perfect reported classifier metrics are invalid as independent fraud-detection evidence. The `is_high_risk` target in cell 28 is directly derived from the same red-flag and network features included in the cell 26 feature list. In addition, cells 20–25 create supplier aggregates, graph centrality, community, Node2Vec and buyer-supplier recurrence features over the complete dataset before cell 31 performs a random stratified split. These steps allow future and test-period information into predictors. Cell 6's `num_bidders` includes both `supplier` and `tenderer` roles, although an awarded supplier is not the same thing as a disclosed competing tenderer. The detailed audit is in `docs/notebook_audit.md`.

The replacement pipeline corrects these issues at the design level. It does not present the earlier notebook's AUC or risk prevalence as empirical findings.

## 6. Ethics, fairness and operational safeguards

A high anomaly score can arise from a legitimate emergency, a specialised market, a small supplier pool, inconsistent reporting or missing publication fields. Missingness is reported as a data limitation rather than misconduct. No protected attribute is inferred. The system is human-in-the-loop, and any consequential follow-up requires independent documentary evidence, due process and authorised review. The project must follow Strathmore's research-ethics requirements, data terms and applicable Kenyan law, including the Data Protection Act, 2019.

Monitoring currently detects selected covariate/input drift. Concept drift and fairness performance cannot be assessed without later, valid outcome or reviewer labels. If those labels become available, the next evaluation should include NDCG@K and observed-positive retrieval, calibration where meaningful, buyer- and supplier-disjoint tests, subgroup coverage and error analysis, confidence intervals, and sensitivity to the PU class prior.

## 7. Limitations and next empirical steps

The main limitation is that the real PPIP raw export and verified audit/sanction outcomes were not supplied and could not be retrieved from the registry download endpoint during this build. Also, the registry's compiled JSONL snapshot does not by itself preserve the original release history; an early-tender prediction claim requires embedded/raw releases or versioned records. The present archive is therefore a tested software capstone foundation and synthetic deployment demonstration, not a completed empirical study on the real Kenyan dataset. Before presenting this project as empirically validated, the student must:

1. obtain the actual PPIP/OCDS export and freeze the raw snapshot with download date, source, file hash and permitted-use notes;
2. execute ingestion and report actual schema/field coverage, invalid rows, unique OCIDs, tenderer disclosure and temporal changes around e-GPS adoption;
3. link a documented set of official positive findings only where linkage quality and publication time are defensible;
4. run the chronological evaluation and, when data permit, buyer- and supplier-disjoint evaluation and model ablations;
5. compare with transparent red-flag, logistic/RF/boosted-tree and Isolation Forest baselines using the same permitted information and splits;
6. run the container build and API smoke test on Docker, then replace the synthetic model artifact with the real-data model;
7. update all numerical results, interpretation and limitations from the actual runs.

## 8. Conclusion

The capstone repository now provides the core end-to-end engineering requirements: scripted raw-data ingestion, leakage-aware feature construction, a reproducible training command, saved model/preprocessing artifact, FastAPI scoring service, Docker packaging, automated tests, monitoring scaffolding and project documentation. Its strongest contribution at this stage is a more defensible data and evaluation design than the original notebook. Real-world predictive performance remains unestablished until the PPIP snapshot and independent outcome evidence are available.
