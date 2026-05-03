# Mutation-Aware BRCA Variant Classifier

A deep learning model that classifies genetic variants as **Pathogenic** or **Benign** using mutation-aware dual DNA sequence encoding (REF vs ALT), a multi-kernel CNN + Transformer architecture, and explainability analysis (Integrated Gradients, Absolute Gradients, In-Silico Mutagenesis).

---

## Project Structure

```
/root/project/
├── train.py                        # Main training script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── variant_summary.txt.gz          # Downloaded automatically (ClinVar)
├── variant_summary.txt             # Extracted automatically
├── hg38.fa.gz                      # Downloaded automatically (~900 MB)
├── hg38.fa                         # Extracted automatically (~15 GB)
├── dataset_cache.pkl               # Built automatically (re-used on next run)
│
├── best_brca_mutation_model.pt     # Saved best model checkpoint
├── guided_absolutegrad_sample.csv  # Attribution scores per position
├── llm_explanation_sample.txt      # Gemini LLM explanation output
│
├── absgrad_ref.png                 # REF attribution plot
├── absgrad_alt.png                 # ALT attribution plot
├── absgrad_delta.png               # Delta attribution plot (ALT - REF)
├── absgrad_ref_diff.png            # REF plot zoomed to diff region
└── absgrad_alt_diff.png            # ALT plot zoomed to diff region
```

---

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended: RTX Pro 4000 24GB or better)
- ~20 GB free disk space (for hg38 genome)
- Gemini API key (for LLM explanation step)

---

## Setup

### 1. Clone / upload files to your VM

```bash
mkdir -p /root/project
cd /root/project
# Upload train.py and requirements.txt here via scp or JupyterLab
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Gemini API key

```bash
export GEMINI_API_KEY="your-key-here"
```

To make it permanent across sessions, add it to your shell profile:

```bash
echo 'export GEMINI_API_KEY="your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

---

## Running

```bash
cd /root/project
python train.py
```

The script will automatically:
1. Download ClinVar variant data
2. Download the hg38 reference genome (~900 MB compressed, ~15 GB uncompressed)
3. Filter and build the mutation-aware dataset
4. Cache the dataset to `dataset_cache.pkl` (re-used on future runs)
5. Train the model for 8 epochs
6. Save the best model checkpoint
7. Run explainability analysis and save plots
8. Send results to Gemini API and save the explanation

> **Note:** The first run takes the longest due to the genome download and dataset building. Subsequent runs skip both steps using the cache.

---

## Viewing Plots

Since the VM has no display, plots are saved as PNG files. You have three options:

### Option A — Download PNGs to your PC
```bash
# Run this on your local machine
scp root@<your-vm-ip>:/root/project/*.png ./
```

### Option B — JupyterLab in the browser (recommended)
```bash
pip install jupyterlab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```
Then open `http://<your-vm-ip>:8888` in your browser. You can upload the `.ipynb` notebook and run it interactively with inline plots.

### Option C — VS Code Remote SSH
Install the **Remote - SSH** extension in VS Code, connect to your VM, and open `/root/project/`. Plots render in a side panel.

---

## Model Architecture

```
Input: REF sequence + ALT sequence (one-hot encoded, 1024 bp × 5 channels)
         ↓                    ↓
   MultiKernelCNN        MultiKernelCNN       ← kernels: 3, 7, 15
   (3 parallel Conv1D branches)
         ↓                    ↓
   Linear projection     Linear projection    ← to transformer_dim=128
         ↓                    ↓
   PositionalEncoding    PositionalEncoding
         ↓                    ↓
   TransformerEncoder    TransformerEncoder   ← 2 layers, 4 heads
         ↓                    ↓
         └──── Δ = ALT - REF ─────────────┘
                      ↓
          MutationAwarePooling               ← attention biased to mutation site
                      ↓
         [ref_g | alt_g | delta_g]           ← concat pooled vectors
                      ↓
              MLP Classifier                 ← 384 → 256 → 64 → 1
                      ↓
              Pathogenic probability
```

---

## Explainability Methods

| Method | Description | Output |
|---|---|---|
| Guided Absolute Gradients | Backprop from logit, sum abs gradients per position | `absgrad_ref.png`, `absgrad_alt.png` |
| Delta AbsoluteGrad | ALT scores minus REF scores | `absgrad_delta.png` |
| Integrated Gradients | Path integral from zero baseline to input | Via `captum` library |
| In-Silico Mutagenesis | Mutate each position, measure probability change | Heatmap plot |
| LLM Explanation | Gemini summarizes findings in plain language | `llm_explanation_sample.txt` |

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Window size | 1024 bp |
| Batch size | 64 |
| Epochs | 8 |
| Learning rate | 2e-4 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Loss | BCEWithLogitsLoss (class-weighted) |
| Mixed precision | Yes (torch.amp) |

---

## Data Sources

- **ClinVar** — variant_summary.txt from NCBI FTP (GRCh38 assembly only)
- **hg38** — Reference genome from UCSC Golden Path

Variants are filtered to: `single nucleotide variant`, `deletion`, `insertion`, `indel` with clear Pathogenic or Benign labels (conflicting interpretations excluded).

---

## Troubleshooting

**CUDA out of memory**
Reduce batch size in `train.py`: change `BATCH_SIZE = 64` to `32` or `16`.

**Disk space error during hg38 download**
You need ~20 GB free. Check with `df -h`. If tight, delete the `.gz` after extraction:
```bash
rm /root/project/hg38.fa.gz
```

**Dataset build is slow**
Normal — it processes hundreds of thousands of variants against the genome. It only runs once; after that the cache is reused automatically.

**Gemini API error**
Make sure your key is set: `echo $GEMINI_API_KEY`. If empty, re-run the export command.

**Port 8888 not accessible for JupyterLab**
Check your VM's firewall rules in the GPU Mart dashboard and open port 8888 for inbound traffic.
