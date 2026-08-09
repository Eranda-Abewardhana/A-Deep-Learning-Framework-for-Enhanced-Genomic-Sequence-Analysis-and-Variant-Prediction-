"""
FastAPI backend for BRCA Mutation Pathogenicity Prediction Demo
BSc AI Thesis — Genomic Variant Classification with Δ-Learning
"""

import os
import math
import json
import traceback
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_CKPT = os.environ.get(
    "MODEL_CKPT",
    os.path.join(os.path.dirname(__file__), "best_brca_mutation_model.pt"),
)
SEQ_LEN = 1024          # model's fixed input window
IN_CHANNELS = 5         # A, C, G, T, mutation-marker
DEVICE = "cpu"

# ─── Model components ─────────────────────────────────────────────────────────

class MultiKernelCNN(nn.Module):
    """Three parallel Conv1d branches with different kernel sizes."""
    def __init__(self, in_ch: int = 5, out_ch: int = 32,
                 kernels: tuple = (3, 7, 15)):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernels:
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ))

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], dim=1)  # (B, 96, L)


class SinusoidalPosEnc(nn.Module):
    """Fixed sinusoidal positional encoding (max_len buffer)."""
    def __init__(self, d_model: int = 128, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, L, D)
        return x + self.pe[:, :x.size(1)]


class GatedAttentionPooler(nn.Module):
    """Attention-gated pooling with separate gates for ref, alt, delta."""
    def __init__(self, d: int = 128):
        super().__init__()
        self.ref_gate = nn.Linear(d, 1)
        self.alt_gate = nn.Linear(d, 1)
        self.delta_gate = nn.Linear(d, 1)

    def _pool(self, h, gate):
        # h: (B, L, D)
        w = torch.softmax(gate(h), dim=1)  # (B, L, 1)
        return (w * h).sum(dim=1)           # (B, D)

    def forward(self, ref_h, alt_h, delta_h):
        return torch.cat([
            self._pool(ref_h, self.ref_gate),
            self._pool(alt_h, self.alt_gate),
            self._pool(delta_h, self.delta_gate),
        ], dim=1)  # (B, 3D)


class MutationAwareBRCAClassifierV2(nn.Module):
    """
    Dual-branch CNN + Transformer with explicit Δ-learning module.
    Architecture:
      ref_cnn / alt_cnn : MultiKernelCNN (in=5, out=32×3=96)
      ref_proj / alt_proj: Conv1d(96→128, kernel=1)
      pos_enc           : SinusoidalPosEnc(d=128)
      transformer       : 2-layer TransformerEncoder(d=128, nhead=4, ff=256)
      pooler            : GatedAttentionPooler(d=128) → 384
      classifier        : 384 → 256 → 64 → 1 (with ReLU + Dropout)
    """

    def __init__(self, in_channels: int = 5, cnn_filters: int = 32,
                 cnn_kernels: tuple = (3, 7, 15), d_model: int = 128,
                 nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 256, dropout: float = 0.3):
        super().__init__()

        cnn_out = cnn_filters * len(cnn_kernels)  # 96

        # Dual-branch CNN
        self.ref_cnn = MultiKernelCNN(in_channels, cnn_filters, cnn_kernels)
        self.alt_cnn = MultiKernelCNN(in_channels, cnn_filters, cnn_kernels)

        # Project CNN output to transformer dim
        self.ref_proj = nn.Conv1d(cnn_out, d_model, 1)
        self.alt_proj = nn.Conv1d(cnn_out, d_model, 1)

        # Positional encoding
        self.pos_enc = SinusoidalPosEnc(d_model)

        # Shared Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)

        # Gated attention pooling
        self.pooler = GatedAttentionPooler(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _encode_branch(self, x, cnn, proj):
        """Run one branch: CNN → project → pos_enc → transformer."""
        h = cnn(x)                       # (B, 96, L)
        h = proj(h)                      # (B, 128, L)
        h = h.permute(0, 2, 1)           # (B, L, 128)
        h = self.pos_enc(h)              # add positional encoding
        h = self.transformer(h)          # (B, L, 128)
        return h

    def forward(self, ref_x, alt_x):
        ref_h = self._encode_branch(ref_x, self.ref_cnn, self.ref_proj)
        alt_h = self._encode_branch(alt_x, self.alt_cnn, self.alt_proj)

        # Δ-learning: element-wise difference
        delta_h = ref_h - alt_h  # (B, L, 128)

        # Pool and classify
        pooled = self.pooler(ref_h, alt_h, delta_h)  # (B, 384)
        logit = self.classifier(pooled)               # (B, 1)
        return logit


# ─── Sequence encoding utilities ──────────────────────────────────────────────

BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def seq_to_onehot(seq: str) -> np.ndarray:
    """Convert a DNA string to a (4, L) one-hot array."""
    seq = seq.upper().replace(" ", "").replace("\n", "")
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, ch in enumerate(seq):
        idx = BASE_MAP.get(ch)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr


def anchor_pad_or_crop(ref_oh: np.ndarray, alt_oh: np.ndarray,
                       seq_len: int = SEQ_LEN):
    """
    Centre both sequences so that the mutation site is at position seq_len//2.
    Returns 5-channel tensors: (5, seq_len) — channels 0-3 = ACGT one-hot,
    channel 4 = mutation-site indicator.
    """
    L_ref = ref_oh.shape[1]
    L_alt = alt_oh.shape[1]

    # Find the first position where ref and alt differ
    min_len = min(L_ref, L_alt)
    mut_pos = None
    for i in range(min_len):
        if not np.array_equal(ref_oh[:, i], alt_oh[:, i]):
            mut_pos = i
            break
    if mut_pos is None:
        # No difference found — check length differences (insertion/deletion)
        if L_ref != L_alt:
            mut_pos = min_len
        else:
            mut_pos = min_len // 2  # fallback: centre

    centre = seq_len // 2  # 512

    def _place(oh, length):
        out = np.zeros((5, seq_len), dtype=np.float32)
        start_src = max(0, mut_pos - centre)
        start_dst = max(0, centre - mut_pos)
        copy_len = min(length - start_src, seq_len - start_dst)
        if copy_len > 0:
            out[:4, start_dst:start_dst + copy_len] = \
                oh[:, start_src:start_src + copy_len]
        # Mutation marker on the centred position
        marker_pos = min(centre, mut_pos) if mut_pos < centre else centre
        if 0 <= marker_pos < seq_len:
            out[4, marker_pos] = 1.0
        return out

    return _place(ref_oh, L_ref), _place(alt_oh, L_alt), mut_pos


# ─── Load model ───────────────────────────────────────────────────────────────

app = FastAPI(title="BRCA Mutation Pathogenicity Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model: Optional[MutationAwareBRCAClassifierV2] = None
model_loaded = False

try:
    model = MutationAwareBRCAClassifierV2()
    ckpt = torch.load(MODEL_CKPT, map_location=DEVICE, weights_only=False)

    # Handle DataParallel prefix
    if any(k.startswith("module.") for k in ckpt.keys()):
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

    model.load_state_dict(ckpt, strict=True)
    model.eval()
    model_loaded = True
    print(f"[OK] Model loaded from {MODEL_CKPT}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    traceback.print_exc()
    print(f"\n[FAIL] Failed to load model: {e}")
    # Print key diff for debugging
    if model is not None:
        expected = set(model.state_dict().keys())
        got = set(ckpt.keys()) if 'ckpt' in dir() else set()
        missing = expected - got
        unexpected = got - expected
        if missing:
            print(f"  Missing keys ({len(missing)}): {sorted(missing)[:10]}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:10]}")


# ─── Sample variants (from thesis evaluation data) ───────────────────────────

SAMPLES = [
    {
        "id": "pathogenic_brca1_del",
        "label": "Pathogenic (BRCA1 Frameshift c.68_72del)",
        "gene": "BRCA1",
        "variant": "5-bp Frameshift Deletion",
        "ref": ("GATCTCCCAGCCCCAGTCGGGAAGGAGCTTTGTTCAGACTTTTGAAAAGC"
                "ACCAGGATCCTTTGGTTCAGCTACAGGATGGAAAGTCAGGGCTCAAACTG"
                "GATTCATTTCCAGGTTGGCTCTGAGATGGATGATACTGAAGCTGATCCTC"
                "TTTCAAGCTCAGCCAGACTGCTCTCTTCAGAAATATCACTTGCATGGCCT"
                "GGAAAGCCAGCTCTTCTACCATCCATGTCAGAGTCATGGAAACACTCTCTA"
                "ACCTCCTTTGTTTTACCTCTATCTTGTACTCCATAAACCTCTCAGTACAGA"
                "ACATACAACTGATCACCCTGAATGGATCCCAAAGAGCTGATAAATATAAAT"
                "TGTGCTTAATTAAGAAGTCTTCCTTCAAAGAAGCAATGGCCAATCTAGTG"
                "AAGAATCAGGCAGCTGAATTTATTCAGCCTTCATTGCAAAAGCGCTGATA"
                "TTTTTAGAGATCTCTTGAGAATCTGATCAGAAAATCTAATGGATGCAAAG"
                "CTATTTAAGCATTGTACC"),
        "alt": ("GATCTCCCAGCCCCAGTCGGGAAGGAGCTTTGTTCAGACTAAAGCACCAG"
                "GATCCTTTGGTTCAGCTACAGGATGGAAAGTCAGGGCTCAAACTGGATTC"
                "ATTTCCAGGTTGGCTCTGAGATGGATGATACTGAAGCTGATCCTCTTTCA"
                "AGCTCAGCCAGACTGCTCTCTTCAGAAATATCACTTGCATGGCCTGGAAA"
                "GCCAGCTCTTCTACCATCCATGTCAGAGTCATGGAAACACTCTCTAACCT"
                "CCTTTGTTTTACCTCTATCTTGTACTCCATAAACCTCTCAGTACAGAACA"
                "TACAACTGATCACCCTGAATGGATCCCAAAGAGCTGATAAATATAAATTG"
                "TGCTTAATTAAGAAGTCTTCCTTCAAAGAAGCAATGGCCAATCTAGTGAA"
                "GAATCAGGCAGCTGAATTTATTCAGCCTTCATTGCAAAAGCGCTGATATT"
                "TTTAGAGATCTCTTGAGAATCTGATCAGAAAATCTAATGGATGCAAAGCT"
                "ATTTAAGCATTGTACC"),
    },
    {
        "id": "benign_brca1_snv",
        "label": "Benign (BRCA1 Synonymous c.5137G>A)",
        "gene": "BRCA1",
        "variant": "G>A SNV (Low Risk)",
        "ref": ("GATCTCCCAGCCCCAGTCGGGAAGGAGCTTTGTTCAGACTTTTGAAAAGC"
                "ACCAGGATCCTTTGGTTCAGCTACAGGATGGAAAGTCAGGGCTCAAACTG"
                "GATTCATTTCCAGGTTGGCTCTGAGATGGATGATACTGAAGCTGATCCTC"
                "TTTCAAGCTCAGCCAGACTGCTCTCTTCAGAAATATCACTTGCATGGCCT"
                "GGAAAGCCAGCTCTTCTACCATCCATGTCAGAGTCATGGAAACACTCTCTA"
                "ACCTCCTTTGTTTTACCTCTATCTTGTACTCCATAAACCTCTCAGTACAGA"
                "ACATACAACTGATCACCCTGAATGGATCCCAAAGAGCTGATAAATATAAAT"
                "TGTGCTTAATTAAGAAGTCTTCCTTCAAAGAAGCAATGGCCAATCTAGTG"
                "AAGAATCAGGCAGCTGAATTTATTCAGCCTTCATTGCAAAAGCGCTGATA"
                "TTTTTAGAGATCTCTTGAGAATCTGATCAGAAAATCTAATGGATGCAAAG"
                "CTATTTAAGCATTGTACC"),
        "alt": ("GATCTCCCAGCCCCAGTCGGGAAGGAGCTTTGTTCAGACTTTTGAAAAGC"
                "ACCAGGATCCTTTGGTTCAGCTACAGGATGGAAAGTCAGGGCTCAAACTG"
                "GATTCATTTCCAGGTTGGCTCTGAGATGGATGATACTGAAGCTGATCCTC"
                "TTTCAAGCTCAGCCAGACTGCTCTCTTCAGAAATATCACTTGCATGGCCT"
                "GGAAAGCCAGCTCTTCTACCATCCATGTCAGAGTCATGGAAACACTCTCTA"
                "ACCTCCTTTGTTTTACCTCTATCTTGTACTCCATAAACCTCTCAGTACAGA"
                "ACATACAACTGATCACCCTGAATGGATCCCAAAGAGCTGATAAATATAAAT"
                "TGTGCTTAATTAAGAAGTCTTCCTTCAAAGAAGCAATGGCCAATCTAGTA"
                "AAGAATCAGGCAGCTGAATTTATTCAGCCTTCATTGCAAAAGCGCTGATA"
                "TTTTTAGAGATCTCTTGAGAATCTGATCAGAAAATCTAATGGATGCAAAG"
                "CTATTTAAGCATTGTACC"),
    },
    {
        "id": "benign_arhgap21",
        "label": "Benign (ARHGAP21 G>A)",
        "gene": "ARHGAP21",
        "variant": "G>A SNV",
        "ref": ("GGATCTCCGATTTCTCCCTCTGCTAAAGGTCAGAGGTACTGGTGCGTAG"
                "GCCGTTCCCTGGCCCAGCCAGTCTCGGCATTCACTTTCCTCTCCCGCTCT"
                "GCTCTCATTCTCCCTAGTGTATATCTGCGCCGGATGCTTTTCCTTTTTAG"
                "CAGTCTCGGCATTCATCCCGCAGGGGGCTCTGGGGTCTCAGTTTTAACCG"),
        "alt": ("GGATCTCCGATTTCTCCCTCTGCTAAAGGTCAGAGGTACTGGTGCGTAG"
                "GCCGTTCCCTGGCCCAGCCAGTCTCGGCATTCACTTTCCTCTCCCGCTCT"
                "GCTCTCATTCTCCCTAGTGTATATCTGCGCCGGATGCTTTTCCTTTTTAG"
                "CAGTCTCGGCATTCATCCCGCAGGGGGCTCTGGGGTCTCAGTTTTAACCG"),
    },
]


# ─── API schemas ──────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    ref_seq: str
    alt_seq: str

class AttrPoint(BaseModel):
    pos: int
    ref_base: str
    alt_base: str
    ref_attr: float
    alt_attr: float

class PredictResponse(BaseModel):
    probability: float
    label: str
    seq_len_used: int
    mutation_pos: int
    attribution: list[AttrPoint]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
@app.get("/demo.html")
async def get_demo_html():
    demo_path = os.path.join(os.path.dirname(__file__), "demo.html")
    if os.path.exists(demo_path):
        return FileResponse(demo_path)
    raise HTTPException(status_code=404, detail="demo.html not found")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "device": str(DEVICE),
        "ckpt": MODEL_CKPT,
    }


@app.get("/samples")
def get_samples():
    return SAMPLES


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not model_loaded or model is None:
        raise HTTPException(503, "Model not loaded")

    ref_seq = req.ref_seq.upper().strip().replace(" ", "").replace("\n", "")
    alt_seq = req.alt_seq.upper().strip().replace(" ", "").replace("\n", "")

    if len(ref_seq) < 10 or len(alt_seq) < 10:
        raise HTTPException(400, "Sequences must be at least 10 bp long")

    # One-hot encode
    ref_oh = seq_to_onehot(ref_seq)
    alt_oh = seq_to_onehot(alt_seq)

    # Centre around mutation site into 1024bp window
    ref_input, alt_input, mut_pos = anchor_pad_or_crop(ref_oh, alt_oh)

    # Convert to tensors with gradient tracking for attribution
    ref_t = torch.tensor(ref_input, dtype=torch.float32).unsqueeze(0)
    alt_t = torch.tensor(alt_input, dtype=torch.float32).unsqueeze(0)
    ref_t.requires_grad_(True)
    alt_t.requires_grad_(True)

    # Forward pass
    logit = model(ref_t, alt_t)
    prob = torch.sigmoid(logit).item()

    # ── Gradient-based attribution (absolute gradient × input) ────────
    model.zero_grad()
    if ref_t.grad is not None:
        ref_t.grad.zero_()
    if alt_t.grad is not None:
        alt_t.grad.zero_()

    # Backward pass from sigmoid output
    sig = torch.sigmoid(logit)
    sig.backward()

    ref_grad = ref_t.grad.detach().squeeze(0)  # (5, 1024)
    alt_grad = alt_t.grad.detach().squeeze(0)

    # Absolute-gradient × input, summed over channels (4 base channels only)
    ref_attr_map = (ref_grad[:4] * ref_t.detach().squeeze(0)[:4]).abs().sum(dim=0)
    alt_attr_map = (alt_grad[:4] * alt_t.detach().squeeze(0)[:4]).abs().sum(dim=0)

    # Build attribution list
    # Determine what's actually in the window
    ref_attr_np = ref_attr_map.numpy()
    alt_attr_np = alt_attr_map.numpy()

    centre = SEQ_LEN // 2
    # Offset in original sequence
    start_in_orig = max(0, mut_pos - centre)

    attr_list = []
    for i in range(SEQ_LEN):
        orig_idx = start_in_orig + i
        # Get the base from the one-hot
        ref_oh_col = ref_input[:4, i]
        alt_oh_col = alt_input[:4, i]
        bases = "ACGT"
        rb = bases[ref_oh_col.argmax()] if ref_oh_col.max() > 0 else "N"
        ab = bases[alt_oh_col.argmax()] if alt_oh_col.max() > 0 else "N"

        attr_list.append(AttrPoint(
            pos=i,
            ref_base=rb,
            alt_base=ab,
            ref_attr=float(ref_attr_np[i]),
            alt_attr=float(alt_attr_np[i]),
        ))

    label = "Pathogenic" if prob >= 0.5 else "Benign"

    return PredictResponse(
        probability=prob,
        label=label,
        seq_len_used=SEQ_LEN,
        mutation_pos=centre,
        attribution=attr_list,
    )


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
