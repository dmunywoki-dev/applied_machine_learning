# Mapping to DSA 8401 course content

| Course area | How it is applied in this capstone | Deliberate choice / exclusion |
|---|---|---|
| Data preparation and feature engineering | OCDS JSON/JSONL ingestion with raw-history versus compiled-snapshot detection, invalid-date audit, missingness indicators, custom point-in-time buyer history, TF-IDF and `ColumnTransformer` preprocessing | No global imputation or whole-dataset aggregation before the split; no post-award fields for the tender they belong to |
| Leakage and evaluation discipline | One OCID observation, same-timestamp batching, own-award exclusion, timestamp-grouped chronological split | No random record-level stratified split for a forecasting task |
| Supervised learning | Optional non-negative PU logistic learner if verified positive findings and a defensible prevalence prior become available | Unlabelled does not mean negative; no synthetic fraud target constructed from the feature columns |
| Unsupervised learning | Isolation Forest anomaly ranking as the deployable default; optional input-distribution profiling | No claim that statistical unusualness equals fraud; no clustering used as a proxy label |
| Dimensionality reduction | Not required for the initial serving model | PCA/t-SNE are useful for exploratory visualisation but do not directly answer the review-ranking task or improve the scoring contract yet |
| Deep learning | Text remains TF-IDF; structured features use classical preprocessing | MLP/TabM, CNN and transfer-learning image methods do not fit the present tabular + tender-text data and available labels |
| Sequence models and time series | Strict historical aggregation and a forward-only evaluation split | No GRU/TFT: the current unit is a tender process with irregular event history, and there is no demonstrated need for a full sequence forecaster |
| Transformers and GenAI | Tender text is represented as TF-IDF for the baseline | No fine-tuned LLM or RAG: the task is structured risk ranking, not generation or policy QA; a transformer text encoder is a later ablation if evidence justifies it |
| Graph and relational modelling | Buyer histories, supplier concentration and conditional disclosed-bidder-group recurrence | No learned GNN/HGT or node2vec in the current release: entity identifiers and tenderer disclosure require validation; graph centrality computed over the full dataset would leak future events |
| Deployment | FastAPI typed input/output contracts, health endpoint, serialized preprocessing+model, non-root Dockerfile and Compose | Docker build was not locally executed because Docker was unavailable in the build environment |
| MLOps and experiment tracking | Optional MLflow run logging, seed and snapshot metadata, CI tests/container build | Automated registry promotion, champion/challenger gates and approval workflow are identified as extensions, not claimed as complete |
| Monitoring, drift and fairness | KS and PSI checks for selected numeric/categorical features; missingness and data-confidence trace | Input drift is not concept drift; outcome and subgroup fairness monitoring require validated labels and a governance process |
