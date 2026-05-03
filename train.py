#!/usr/bin/env python3
"""
Mutation-Aware BRCA Variant Classifier
Converted from Jupyter Notebook → VM-ready Python script
Run: python train.py
"""

# ============================================================
# SETUP: Set your Gemini API key before running
#   export GEMINI_API_KEY="your-key-here"
# ============================================================

import os
import sys
import gzip
import math
import random
import shutil
import pickle
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no display needed on a server
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial
from multiprocessing import Pool

from pyfaidx import Fasta

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score

# ── Working directory (replaces /kaggle/working/) ────────────
WORK_DIR = Path("/root/project")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# ── Paths ─────────────────────────────────────────────────────
CLINVAR_GZ   = WORK_DIR / "variant_summary.txt.gz"
CLINVAR_TXT  = WORK_DIR / "variant_summary.txt"
HG38_GZ      = WORK_DIR / "hg38.fa.gz"
HG38_FA      = WORK_DIR / "hg38.fa"
CACHE_PATH   = WORK_DIR / "dataset_cache.pkl"
BEST_MODEL   = WORK_DIR / "best_brca_mutation_model.pt"
ABSGRAD_CSV  = WORK_DIR / "guided_absolutegrad_sample.csv"
LLM_TXT      = WORK_DIR / "llm_explanation_sample.txt"

# ── Hyperparameters ───────────────────────────────────────────
SEED        = 42
WINDOW_SIZE = 1024
EPOCHS      = 8
BATCH_SIZE  = 64
LR          = 2e-4
WEIGHT_DECAY= 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)


# ============================================================
# 1. Download ClinVar variant_summary
# ============================================================
CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"

if not CLINVAR_GZ.exists():
    print("Downloading ClinVar …")
    subprocess.run(["wget", "-O", str(CLINVAR_GZ), CLINVAR_URL], check=True)

if not CLINVAR_TXT.exists():
    print("Extracting ClinVar …")
    with gzip.open(CLINVAR_GZ, "rb") as f_in, open(CLINVAR_TXT, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

print("ClinVar file ready:", CLINVAR_TXT)


# ============================================================
# 2. Load + filter ClinVar
# ============================================================
df = pd.read_csv(CLINVAR_TXT, sep="\t", low_memory=False)
print("Raw shape:", df.shape)

df = df[df["Assembly"] == "GRCh38"].copy()

df = df[
    df["ReferenceAlleleVCF"].notna() &
    df["AlternateAlleleVCF"].notna() &
    (df["ReferenceAlleleVCF"] != "-") &
    (df["AlternateAlleleVCF"] != "-")
].copy()

bad_tokens = {"na", "NA", ".", ""}
df = df[
    ~df["ReferenceAlleleVCF"].astype(str).isin(bad_tokens) &
    ~df["AlternateAlleleVCF"].astype(str).isin(bad_tokens)
].copy()


def normalize_clinsig(x: str):
    x = str(x).strip().lower()
    if "conflicting" in x:
        return None
    if "pathogenic" in x:
        return 1
    if "benign" in x:
        return 0
    return None


df["label"] = df["ClinicalSignificance"].apply(normalize_clinsig)
df = df[df["label"].notna()].copy()
df["label"] = df["label"].astype(int)

allowed_types = {"single nucleotide variant", "deletion", "insertion", "indel"}
df = df[df["Type"].str.lower().isin(allowed_types)].copy()
df = df.drop_duplicates(subset=["VariationID", "Assembly"])

print("Filtered shape:", df.shape)
print(df["label"].value_counts())


# ============================================================
# 3. Download hg38 reference genome
# ============================================================
HG38_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"

if not HG38_GZ.exists():
    print("Downloading hg38 (~900 MB compressed) … this may take a while")
    subprocess.run(["wget", "-O", str(HG38_GZ), HG38_URL], check=True)

if not HG38_FA.exists():
    print("Decompressing hg38 … (~15 GB uncompressed)")
    subprocess.run(
        f"gunzip -c {HG38_GZ} > {HG38_FA}",
        shell=True, check=True
    )

print("hg38 FASTA ready:", HG38_FA)


# ============================================================
# 4. Load hg38
# ============================================================
genome = Fasta(str(HG38_FA), as_raw=True, sequence_always_upper=True)
print("Chromosomes loaded:", len(genome.keys()))
print(list(genome.keys())[:5])


# ============================================================
# 5. Sequence utilities
# ============================================================
BASE2IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
IDX2BASE = {v: k for k, v in BASE2IDX.items()}
BASES    = ["A", "C", "G", "T", "N"]


def clean_seq(seq: str) -> str:
    return "".join(ch if ch in "ACGT" else "N" for ch in str(seq).upper())


def clean_seq_loose(seq: str) -> str:
    return "".join(ch if ch in "ACGTN" else "N" for ch in str(seq).upper())


def one_hot_encode(seq: str) -> np.ndarray:
    seq = clean_seq(seq)
    arr = np.zeros((len(seq), 5), dtype=np.float32)
    for i, ch in enumerate(seq):
        arr[i, BASE2IDX.get(ch, 4)] = 1.0
    return arr


def decode_onehot(x) -> str:
    return "".join(BASES[i] for i in x.argmax(axis=-1))


def anchor_crop_or_pad(seq: str, target_len: int, anchor: int, pad_char="N") -> str:
    seq  = clean_seq(seq)
    half = target_len // 2
    left = anchor - half
    right= left + target_len
    if left < 0:
        seq   = pad_char * (-left) + seq
        left  = 0
        right = target_len
    chunk = seq[left:right]
    if len(chunk) < target_len:
        chunk = chunk + pad_char * (target_len - len(chunk))
    return chunk


def get_chr_name(chrom: str) -> str:
    chrom = str(chrom)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def build_ref_alt_window(chrom, pos_1based, ref_allele, alt_allele,
                         genome, window_size=1024):
    chrom         = get_chr_name(chrom)
    ref_allele_raw= str(ref_allele).upper()
    alt_allele_raw= str(alt_allele).upper()
    ref_allele    = clean_seq(ref_allele_raw)
    alt_allele    = clean_seq(alt_allele_raw)
    half          = window_size // 2

    v_start0  = int(pos_1based) - 1
    v_end0    = v_start0 + len(ref_allele)
    fetch_extra = max(len(alt_allele), len(ref_allele), half) + half
    left_start0 = max(0, v_start0 - fetch_extra)
    right_end0  = v_end0 + fetch_extra

    ref_full        = str(genome[chrom][left_start0:right_end0])
    local_var_start = v_start0 - left_start0
    local_var_end   = local_var_start + len(ref_allele)

    observed_ref = clean_seq_loose(ref_full[local_var_start:local_var_end])
    expected_ref = clean_seq_loose(ref_allele_raw)

    if len(observed_ref) != len(expected_ref):
        return None, None, None
    for ob, ex in zip(observed_ref, expected_ref):
        if ob != ex and ob != "N" and ex != "N":
            return None, None, None

    ref_full = clean_seq(ref_full)
    alt_full = ref_full[:local_var_start] + alt_allele + ref_full[local_var_end:]

    ref_seq = anchor_crop_or_pad(ref_full, window_size, anchor=local_var_start)
    alt_seq = anchor_crop_or_pad(alt_full, window_size, anchor=local_var_start)

    return ref_seq, alt_seq, window_size // 2


# ============================================================
# 6. Build dataset (parallel, cached)
# ============================================================
def worker_init(fa_path):
    global _genome
    _genome = Fasta(fa_path, as_raw=True, sequence_always_upper=True)


def build_one_record(row_dict, window_size):
    try:
        ref_seq, alt_seq, var_start = build_ref_alt_window(
            chrom       = row_dict["Chromosome"],
            pos_1based  = int(row_dict["PositionVCF"]),
            ref_allele  = str(row_dict["ReferenceAlleleVCF"]),
            alt_allele  = str(row_dict["AlternateAlleleVCF"]),
            genome      = _genome,
            window_size = window_size
        )
        if ref_seq is None:
            return None
        ref = str(row_dict["ReferenceAlleleVCF"])
        return {
            "VariationID":          row_dict["VariationID"],
            "GeneSymbol":           row_dict["GeneSymbol"],
            "ClinicalSignificance": row_dict["ClinicalSignificance"],
            "PhenotypeList":        row_dict["PhenotypeList"],
            "Type":                 row_dict["Type"],
            "Chromosome":           row_dict["Chromosome"],
            "PositionVCF":          int(row_dict["PositionVCF"]),
            "ref_allele":           ref,
            "alt_allele":           str(row_dict["AlternateAlleleVCF"]),
            "ref_seq":              ref_seq,
            "alt_seq":              alt_seq,
            "var_start":            var_start,
            "var_end":              var_start + len(ref),
            "variant_type":         str(row_dict["Type"]).lower(),
            "label":                int(row_dict["label"]),
        }
    except Exception:
        return None


if CACHE_PATH.exists():
    print("Loading cached dataset …")
    with open(CACHE_PATH, "rb") as f:
        records = pickle.load(f)
else:
    print("Building dataset in parallel …")
    rows = df.to_dict("records")
    fn   = partial(build_one_record, window_size=WINDOW_SIZE)
    with Pool(processes=4, initializer=worker_init,
              initargs=(str(HG38_FA),)) as pool:
        results = pool.map(fn, rows, chunksize=500)
    records = [r for r in results if r is not None]
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(records, f)
    print(f"Saved cache → {CACHE_PATH}")

data_df = pd.DataFrame(records)
print("Usable samples:", data_df.shape)
print(data_df["label"].value_counts())


# ============================================================
# 7. Train / val split
# ============================================================
train_df, val_df = train_test_split(
    data_df, test_size=0.2, random_state=SEED, stratify=data_df["label"]
)
train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
print("Train:", train_df.shape, " | Val:", val_df.shape)


# ============================================================
# 8. PyTorch Dataset
# ============================================================
class BRCAMutationDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        ref_x= torch.tensor(one_hot_encode(row["ref_seq"]), dtype=torch.float32)
        alt_x= torch.tensor(one_hot_encode(row["alt_seq"]), dtype=torch.float32)
        y    = torch.tensor(row["label"], dtype=torch.float32)
        return {
            "ref":          ref_x,
            "alt":          alt_x,
            "label":        y,
            "var_start":    int(row["var_start"]),
            "var_end":      int(row["var_end"]),
            "variant_type": str(row["variant_type"]),
        }


train_ds = BRCAMutationDataset(train_df)
val_ds   = BRCAMutationDataset(val_df)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)


# ============================================================
# 9. Model definition
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiKernelCNN(nn.Module):
    def __init__(self, in_ch=5, branch_ch=32, kernels=(3, 7, 15)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, branch_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(branch_ch),
                nn.GELU()
            )
            for k in kernels
        ])
        self.out_ch = branch_ch * len(kernels)

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], dim=1)


class MutationAwarePooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ref_gate   = nn.Linear(d_model, 1)
        self.alt_gate   = nn.Linear(d_model, 1)
        self.delta_gate = nn.Linear(d_model, 1)

    def forward(self, ref_h, alt_h, delta_h):
        delta_mag   = delta_h.abs().mean(dim=-1)
        ref_attn    = torch.softmax(self.ref_gate(ref_h).squeeze(-1)   + delta_mag, dim=-1)
        alt_attn    = torch.softmax(self.alt_gate(alt_h).squeeze(-1)   + delta_mag, dim=-1)
        delta_attn  = torch.softmax(self.delta_gate(delta_h).squeeze(-1)+delta_mag, dim=-1)
        ref_g   = (ref_h   * ref_attn.unsqueeze(-1)).sum(1)
        alt_g   = (alt_h   * alt_attn.unsqueeze(-1)).sum(1)
        delta_g = (delta_h * delta_attn.unsqueeze(-1)).sum(1)
        return ref_g, alt_g, delta_g, ref_attn, alt_attn, delta_attn


class MutationAwareBRCAClassifierV2(nn.Module):
    def __init__(self, in_ch=5, cnn_branch_ch=32, kernels=(3, 7, 15),
                 transformer_dim=128, nhead=4, num_layers=2,
                 ff_dim=256, dropout=0.1):
        super().__init__()
        self.ref_cnn = MultiKernelCNN(in_ch, cnn_branch_ch, kernels)
        self.alt_cnn = MultiKernelCNN(in_ch, cnn_branch_ch, kernels)
        cnn_out      = self.ref_cnn.out_ch
        self.ref_proj= nn.Conv1d(cnn_out, transformer_dim, 1)
        self.alt_proj= nn.Conv1d(cnn_out, transformer_dim, 1)
        self.pos_enc = PositionalEncoding(transformer_dim)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pooler      = MutationAwarePooling(transformer_dim)
        self.classifier  = nn.Sequential(
            nn.Linear(transformer_dim * 3, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64),                  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def encode_branch(self, x, cnn, proj):
        x = cnn(x.transpose(1, 2))
        x = proj(x).transpose(1, 2)
        return self.transformer(self.pos_enc(x))

    def forward(self, ref_x, alt_x, return_aux=False):
        ref_h   = self.encode_branch(ref_x, self.ref_cnn, self.ref_proj)
        alt_h   = self.encode_branch(alt_x, self.alt_cnn, self.alt_proj)
        delta_h = alt_h - ref_h
        ref_g, alt_g, delta_g, ra, aa, da = self.pooler(ref_h, alt_h, delta_h)
        logits  = self.classifier(torch.cat([ref_g, alt_g, delta_g], -1)).squeeze(-1)
        if return_aux:
            return logits, {"ref_h": ref_h, "alt_h": alt_h, "delta_h": delta_h,
                            "ref_attn": ra, "alt_attn": aa, "delta_attn": da}
        return logits


# ============================================================
# 10. Initialize model, loss, optimizer
# ============================================================
model = MutationAwareBRCAClassifierV2(
    in_ch=5, cnn_branch_ch=32, kernels=(3, 7, 15),
    transformer_dim=128, nhead=4, num_layers=2,
    ff_dim=256, dropout=0.1
).to(DEVICE)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

n_pos      = train_df["label"].sum()
n_neg      = len(train_df) - n_pos
pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)

criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print(model)


# ============================================================
# 11. Training loop (mixed precision)
# ============================================================
scaler = torch.amp.GradScaler("cuda")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, preds_all, labels_all = 0.0, [], []
    for batch in loader:
        ref_x = batch["ref"].to(device, non_blocking=True)
        alt_x = batch["alt"].to(device, non_blocking=True)
        y     = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(ref_x, alt_x)
            loss   = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * y.size(0)
        preds_all.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        labels_all.extend(y.detach().cpu().numpy().tolist())
    return total_loss / len(loader.dataset), np.array(preds_all), np.array(labels_all)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, labels_all = 0.0, [], []
    for batch in loader:
        ref_x = batch["ref"].to(device, non_blocking=True)
        alt_x = batch["alt"].to(device, non_blocking=True)
        y     = batch["label"].to(device, non_blocking=True)
        logits= model(ref_x, alt_x)
        loss  = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        preds_all.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        labels_all.extend(y.cpu().numpy().tolist())
    return total_loss / len(loader.dataset), np.array(preds_all), np.array(labels_all)


# ============================================================
# 12. Train
# ============================================================
best_auc = -1

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_probs, tr_y = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    va_loss, va_probs, va_y = evaluate(model, val_loader, criterion, DEVICE)

    tr_auc = roc_auc_score(tr_y, tr_probs) if len(np.unique(tr_y)) > 1 else float("nan")
    va_auc = roc_auc_score(va_y, va_probs) if len(np.unique(va_y)) > 1 else float("nan")

    print(f"Epoch {epoch:02d}  Train loss {tr_loss:.4f} | AUC {tr_auc:.4f}  "
          f"||  Val loss {va_loss:.4f} | AUC {va_auc:.4f}")

    if va_auc > best_auc:
        best_auc = va_auc
        torch.save(model.state_dict(), str(BEST_MODEL))
        print("  ✅ Saved best model")

print(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")


# ============================================================
# 13. Final evaluation
# ============================================================
model.load_state_dict(torch.load(str(BEST_MODEL), map_location=DEVICE))
_, val_probs, val_y = evaluate(model, val_loader, criterion, DEVICE)
val_pred = (val_probs >= 0.5).astype(int)

print("Best Val AUC:", roc_auc_score(val_y, val_probs))
print("Best Val F1:", f1_score(val_y, val_pred))
print(classification_report(val_y, val_pred, digits=4))


# ============================================================
# 14. Explainability helpers
# ============================================================
def absolute_grad_per_position(grad_tensor):
    return np.abs(grad_tensor.detach().cpu().numpy()[0]).sum(axis=-1)


def get_mut_region_from_sample(sample, flank=60):
    var_start = sample["var_start"]
    var_end   = sample["var_end"]
    center    = (var_start + var_end) // 2
    L         = sample["ref"].shape[0]
    return max(0, center - flank), min(L, center + flank), var_start, var_end


def find_diff_region(ref_seq, alt_seq):
    diffs = [i for i, (r, a) in enumerate(zip(ref_seq, alt_seq)) if r != a]
    return (min(diffs), max(diffs)) if diffs else None


def plot_and_save(xs, scores, title, path, var_start=None, var_end=None, color="red"):
    plt.figure(figsize=(16, 4))
    plt.bar(xs, scores)
    if var_start is not None:
        plt.axvspan(var_start, var_end, alpha=0.15, color=color, label="mutation site")
        plt.legend()
    plt.title(title)
    plt.xlabel("Position")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# 15. Guided Absolute Grad on first val sample
# ============================================================
class AbsoluteGradWrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.model = m
    def forward(self, ref_x, alt_x): return self.model(ref_x, alt_x)


def run_guided_absolute_grad(model, sample, device):
    model.eval()
    wrapper = AbsoluteGradWrapper(model).to(device)
    ref_x = sample["ref"].unsqueeze(0).to(device).clone().detach().requires_grad_(True)
    alt_x = sample["alt"].unsqueeze(0).to(device).clone().detach().requires_grad_(True)
    wrapper.zero_grad(set_to_none=True)
    logits = wrapper(ref_x, alt_x)
    pred_prob = torch.sigmoid(logits).item()
    logits.sum().backward()
    ref_scores = absolute_grad_per_position(ref_x.grad)
    alt_scores = absolute_grad_per_position(alt_x.grad)
    return ref_x, alt_x, ref_scores, alt_scores, pred_prob


sample = val_ds[0]
ref_x, alt_x, ref_scores, alt_scores, pred_prob = run_guided_absolute_grad(model, sample, DEVICE)

ref_seq = decode_onehot(ref_x[0].detach().cpu().numpy())
alt_seq = decode_onehot(alt_x[0].detach().cpu().numpy())
left, right, var_start, var_end = get_mut_region_from_sample(sample, flank=40)

print(f"\nPredicted pathogenic probability: {pred_prob:.4f}")
print(f"Variant type: {sample['variant_type']}")
print(f"Mutation region: [{var_start}, {var_end}]")

plot_and_save(np.arange(left, right), ref_scores[left:right],
              f"REF AbsoluteGrad near mutation ({sample['variant_type']})",
              WORK_DIR / "absgrad_ref.png", var_start, var_end, "red")

plot_and_save(np.arange(left, right), alt_scores[left:right],
              f"ALT AbsoluteGrad near mutation ({sample['variant_type']})",
              WORK_DIR / "absgrad_alt.png", var_start, var_end, "orange")

# Delta plot
delta_scores = alt_scores - ref_scores
plt.figure(figsize=(18, 4))
plt.bar(np.arange(len(delta_scores)), delta_scores)
plt.title("Delta AbsoluteGrad (ALT - REF)")
plt.xlabel("Position"); plt.ylabel("Difference")
plt.tight_layout()
plt.savefig(WORK_DIR / "absgrad_delta.png", dpi=150); plt.close()
print(f"Saved: {WORK_DIR / 'absgrad_delta.png'}")

# Save CSV
absgrad_df = pd.DataFrame({
    "position":    np.arange(len(ref_seq)),
    "ref_base":    list(ref_seq),
    "alt_base":    list(alt_seq),
    "ref_absgrad": ref_scores,
    "alt_absgrad": alt_scores,
})
absgrad_df.to_csv(ABSGRAD_CSV, index=False)
print(f"Saved: {ABSGRAD_CSV}")

# Diff region plot
diff_region = find_diff_region(ref_seq, alt_seq)
if diff_region:
    s, e = diff_region
    l = max(0, s - 40); r = min(len(ref_seq), e + 41)
    plot_and_save(np.arange(l, r), ref_scores[l:r],
                  "REF AbsoluteGrad near diff region", WORK_DIR / "absgrad_ref_diff.png")
    plot_and_save(np.arange(l, r), alt_scores[l:r],
                  "ALT AbsoluteGrad near diff region", WORK_DIR / "absgrad_alt_diff.png")


# ============================================================
# 16. LLM explanation (Gemini)
# ============================================================
import requests
import json


def build_explanation_prompt(sample, pred_prob, ref_scores, alt_scores, var_start, var_end):
    delta         = np.abs(alt_scores - ref_scores)
    top_positions = np.argsort(delta)[::-1][:5].tolist()
    top_scores    = delta[top_positions].tolist()
    ref_seq_local = decode_onehot(sample["ref"].numpy())[var_start-10:var_end+10]
    alt_seq_local = decode_onehot(sample["alt"].numpy())[var_start-10:var_end+10]
    label_str     = "Pathogenic" if int(sample["label"].item()) == 1 else "Benign"

    return f"""You are a genomics expert assistant. A deep learning model has analysed a genetic variant and produced the following results. Please explain these findings clearly for a biomedical researcher.

## Variant Details
- Gene: {val_df.iloc[0]['GeneSymbol']}
- Chromosome: {val_df.iloc[0]['Chromosome']}, Position: {val_df.iloc[0]['PositionVCF']}
- Variant type: {sample['variant_type']}
- Reference allele: {val_df.iloc[0]['ref_allele']}
- Alternate allele: {val_df.iloc[0]['alt_allele']}
- True clinical label: {label_str}

## Model Prediction
- Predicted pathogenicity probability: {pred_prob:.4f}
- Interpretation: {"HIGH risk of pathogenicity" if pred_prob > 0.5 else "LOW risk / likely benign"}

## Explainability Findings
- Mutation site in analysis window: positions {var_start} to {var_end}
- Reference sequence around mutation: {ref_seq_local}
- Alternate sequence around mutation: {alt_seq_local}
- Top 5 most changed positions (by attribution delta):
{chr(10).join(f"  Position {p}: delta score = {s:.4f}" for p, s in zip(top_positions, top_scores))}

## Your Task
1. Briefly explain what this variant is and where it occurs.
2. Interpret the model's predicted probability in clinical context.
3. Explain what the attribution analysis suggests about which genomic positions most influenced the prediction.
4. Comment on whether the prediction aligns with the true label and what that means.
Keep the explanation concise (4-6 sentences per section) and accessible to a researcher without deep ML knowledge."""


def explain_with_llm(prompt, max_tokens=10000):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.\n"
                           "Run: export GEMINI_API_KEY='your-key-here'")

    url = (f"https://generativelanguage.googleapis.com/v1beta/"
           f"models/gemini-2.5-flash:generateContent?key={api_key}")

    response = requests.post(url,
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.3}
        }
    )
    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


sample    = val_ds[0]
pred_prob = float(torch.sigmoid(
    model(sample["ref"].unsqueeze(0).to(DEVICE),
          sample["alt"].unsqueeze(0).to(DEVICE))
).item())

_, _, var_start, var_end = get_mut_region_from_sample(sample, flank=40)
prompt = build_explanation_prompt(sample, pred_prob, ref_scores, alt_scores, var_start, var_end)

print("\nSending to Gemini for explanation …")
explanation = explain_with_llm(prompt)

print("=" * 60)
print("LLM EXPLANATION")
print("=" * 60)
print(explanation)

with open(LLM_TXT, "w") as f:
    f.write(f"Gene    : {val_df.iloc[0]['GeneSymbol']}\n")
    f.write(f"Variant : {val_df.iloc[0]['ref_allele']} → {val_df.iloc[0]['alt_allele']}\n")
    f.write(f"Type    : {val_df.iloc[0]['Type']}\n")
    f.write(f"Pred prob: {pred_prob:.4f}\n")
    f.write(f"True label: {'Pathogenic' if int(sample['label'].item()) == 1 else 'Benign'}\n")
    f.write("=" * 60 + "\n\n")
    f.write(explanation)

print(f"\nSaved: {LLM_TXT}")
print("\n✅ All done!")
