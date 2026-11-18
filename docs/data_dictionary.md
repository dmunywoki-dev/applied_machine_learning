# Data dictionary (processed OCID-level table)

`prediction_time` and the prediction task depend on `source_representation`: raw-release mode uses the first tender-stage release date; compiled-snapshot mode uses the latest compiled-release date. See the README before interpreting any model result as an early warning.

| Field | Meaning | Available at scoring time? |
|---|---|---|
| `ocid` | OCDS contracting-process identifier | Identifier only; not a model feature |
| `prediction_time` | First tender-release date in raw mode; latest compiled-release date in snapshot mode | Yes; defines the scoring cut-off for the declared mode |
| `source_representation`, `prediction_stage` | Identifies raw release history vs latest compiled snapshot | Audit metadata; not model predictors |
| `buyer_id`, `buyer_name`, `county` | Buyer reference and available location fields | Yes; retained for grouping/audit, not sent into the model |
| `procurement_method`, `procurement_category`, `tender_status` | Tender method/category/status | Yes, from the selected tender release |
| `title`, `description`, `text` | Published tender wording | Yes, from the selected release |
| `estimated_value`, `estimated_value_log1p` | Published estimated value and log transform | Yes, where published |
| `duration_days` | Tender-period end minus start in days | Yes, where both tender dates were published in that release |
| `num_tenderers` | Count of parties explicitly marked `tenderer` | Only where disclosed; zero does not mean one bidder or no competition |
| `tenderer_data_available` | Whether an explicit tenderer identity was present | Yes; disclosure coverage indicator |
| `buyer_prior_*` | Buyer history using award events published before the score timestamp | Yes, strictly historical |
| `days_since_buyer_previous_tender` | Days since a prior tender for the buyer | Yes, when history exists |
| `pub_month_*`, `pub_weekday` | Calendar encodings of publication time | Yes |
| `flag_*` | Interpretable process/history indicators | Yes, only from point-in-time fields |
| `hyperedge_*` | Prior bidder-group recurrence from explicit tenderer sets | Conditional; not reliable in the current PPIP snapshot |
| `is_synthetic` | Demo-fixture marker | Not a model predictor; use to block real-world use |

Award values and awarded-supplier identities are not features of the tender they belong to. They enter only the historical state for a later tender after their publication time. No post-award contract execution variables are included in the tender-time model.
