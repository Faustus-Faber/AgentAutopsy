# Kaggle Dataset Publication Guide

This guide details how to publish the **33,000 Trajectory Behavioral Audit Corpus** and the **Full Trajectory Execution Corpus** to Kaggle Datasets.

---

## 1. Overview of Artifacts

| Dataset | Package Directory / File | Size | Formats | Target Kaggle Slug |
| :--- | :--- | :--- | :--- | :--- |
| **Behavioral Audit Corpus** (33,000 trajectories) | `kaggle/kaggle_audit_corpus_package/` | ~6.7 MB (Parquet)<br>~4.0 MB (JSONL.gz) | `.parquet`<br>`.jsonl.gz`<br>`data_dictionary.csv` | `agent-prompt-injection-behavioral-audit-33k` |
| **Trajectory Execution Archives** (146,000+ files) | `kaggle/trajectory_archives/` | ~850 MB total (compressed tar.gz) | Segmented `.tar.gz` per benchmark & intervention | `agent-prompt-injection-trajectories` |

---

## 2. Publishing the 33k Behavioral Audit Corpus

The package is already generated and verified in `kaggle/kaggle_audit_corpus_package/` with:
- `audit_corpus_33k.parquet` (6.69 MB)
- `audit_corpus_33k.jsonl.gz` (4.04 MB)
- `data_dictionary.csv`
- `dataset-metadata.json`

### Option A: Using the Kaggle CLI (Recommended)
1. Install and authenticate the Kaggle CLI if not already configured:
   ```bash
   pip install kaggle
   # Ensure ~/.kaggle/kaggle.json contains your API credentials
   ```
2. Navigate to the repository root:
   ```bash
   cd "E:\Thesis Writing\AgentAutopsy"
   ```
3. Initialize and publish the dataset:
   ```bash
   kaggle datasets create -p kaggle/kaggle_audit_corpus_package
   ```
4. To update the dataset in the future:
   ```bash
   kaggle datasets version -p kaggle/kaggle_audit_corpus_package -m "Updated audit annotations and metadata"
   ```

### Option B: Using the Kaggle Web Interface
1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) and click **+ New Dataset**.
2. Set the Title to:
   ```text
   Tool-Using Agent Prompt-Injection Behavioral Audit (33k Trajectories)
   ```
3. Drag and drop the following files from `kaggle/kaggle_audit_corpus_package/`:
   - `audit_corpus_33k.parquet`
   - `audit_corpus_33k.jsonl.gz`
   - `data_dictionary.csv`
4. Copy the markdown content from `kaggle/AUDIT_DATASET_CARD.md` into the Kaggle description tab.
5. Add relevant tags: `Security`, `Artificial Intelligence`, `NLP`, `Cybersecurity`, `Benchmarking`.
6. Set license to **CC-BY-4.0** or **Apache 2.0** and click **Create**.

---

## 3. Publishing the Full Trajectory Execution Corpus

To publish the complete set of 146,000+ raw episode trajectories without choking the Kaggle uploader:

1. Run the trajectory packaging routine:
   ```bash
   python kaggle/package_kaggle_datasets.py --package-trajectories
   ```
   This generates segmented archives in `kaggle/trajectory_archives/`:
   - `injecagent_baselines.tar.gz` (11 models × 3 precisions, 34,782 files)
   - `agentdojo_baselines.tar.gz` (11 models × 3 precisions, 36,042 files)
   - `boundary_guardrails_intervention.tar.gz` (9 models, 9,496 files)
   - `administrative_authority_interventions.tar.gz` (11 models, 33,046 files)
   - `dose_response_and_reasoning_ablation.tar.gz` (continuous ladder & think-ablation, 23,255 files)
   - `context_budget_and_precision_sweeps.tar.gz` (macro/micro clamping, 9,798 files)

2. Create the dataset via Kaggle CLI:
   ```bash
   kaggle datasets create -p kaggle/trajectory_archives
   ```

---

## 4. Kaggle Starter Notebook Walkthrough

A starter notebook is provided at `kaggle/kaggle_starter_walkthrough.ipynb`. When researchers visit your Kaggle dataset page, they can click **New Notebook** and upload this notebook to reproduce all key findings with one click:
- Loading the 33,000 Parquet records in < 1 second.
- Generating the 6-bucket taxonomy breakdown across all 11 models.
- Computing the Perception-Compliance Dissociation (PCD) gap.
- Calculating the Attack-Task Outcome Matrix (ATOM) on AgentDojo.
