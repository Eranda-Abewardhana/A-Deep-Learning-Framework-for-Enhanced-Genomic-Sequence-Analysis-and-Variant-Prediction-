import sys
import torch
import numpy as np

from backend_server import model, seq_to_onehot, anchor_pad_or_crop, SAMPLES

lines = []
lines.append("--- EVALUATION OF BACKEND MODEL SAMPLES ---")
for i, s in enumerate(SAMPLES):
    ref_oh = seq_to_onehot(s['ref'])
    alt_oh = seq_to_onehot(s['alt'])
    ref_input, alt_input, mut_pos = anchor_pad_or_crop(ref_oh, alt_oh)

    ref_t = torch.from_numpy(ref_input).unsqueeze(0)
    alt_t = torch.from_numpy(alt_input).unsqueeze(0)

    with torch.no_grad():
        logit = model(ref_t, alt_t)
        prob = torch.sigmoid(logit).item()

    lines.append(f"Sample [{i}]: {s['label']}")
    lines.append(f"  Mutation position: {mut_pos}")
    lines.append(f"  Probability: {prob:.6f}")
    lines.append(f"  Classification: {'Pathogenic' if prob >= 0.5 else 'Benign'}")
    lines.append("")

with open("c:\\Users\\Eranda\\Downloads\\thesis_outputs_20260717_201747\\eval_output.txt", "w") as f:
    f.write("\n".join(lines))

print("Evaluation complete.")
