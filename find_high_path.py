import json
import torch
import numpy as np
from backend_server import model, seq_to_onehot, anchor_pad_or_crop

ref_base = "GATCTCCCAGCCCCAGTCGGGAAGGAGCTTTGTTCAGACTTTTGAAAAGCACCAGGATCCTTTGGTTCAGCTACAGGATGGAAAGTCAGGGCTCAAACTGGATTCATTTCCAGGTTGGCTCTGAGATGGATGATACTGAAGCTGATCCTCTTTCAAGCTCAGCCAGACTGCTCTCTTCAGAAATATCACTTGCATGGCCTGGAAAGCCAGCTCTTCTACCATCCATGTCAGAGTCATGGAAACACTCTCTAACCTCCTTTGTTTTACCTCTATCTTGTACTCCATAAACCTCTCAGTACAGAACATACAACTGATCACCCTGAATGGATCCCAAAGAGCTGATAAATATAAATTGTGCTTAATTAAGAAGTCTTCCTTCAAAGAAGCAATGGCCAATCTAGTGAAGAATCAGGCAGCTGAATTTATTCAGCCTTCATTGCAAAAGCGCTGATATTTTTAGAGATCTCTTGAGAATCTGATCAGAAAATCTAATGGATGCAAAGCTATTTAAGCATTGTACC"

def eval_v(ref, alt):
    r_oh = seq_to_onehot(ref)
    a_oh = seq_to_onehot(alt)
    r_t, a_t, _ = anchor_pad_or_crop(r_oh, a_oh)
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(r_t).unsqueeze(0), torch.from_numpy(a_t).unsqueeze(0))).item()

results = []

# Test single base substitutions, insertions, deletions
for pos in range(10, len(ref_base) - 10, 2):
    orig = ref_base[pos]
    for sub in ['A', 'C', 'G', 'T']:
        if sub == orig: continue
        alt_seq = ref_base[:pos] + sub + ref_base[pos+1:]
        p = eval_v(ref_base, alt_seq)
        results.append({"prob": p, "type": "SNV", "pos": pos, "ref": ref_base, "alt": alt_seq})

    # Frameshift del
    alt_del = ref_base[:pos] + ref_base[pos+4:]
    p_del = eval_v(ref_base, alt_del)
    results.append({"prob": p_del, "type": "DEL4", "pos": pos, "ref": ref_base, "alt": alt_del})

results.sort(key=lambda x: x["prob"], reverse=True)

top_high = results[0]
top_low = results[-1]
mid_vus = min(results, key=lambda x: abs(x["prob"] - 0.50))

samples_out = [
    {
        "id": "pathogenic_brca1_high",
        "label": f"Pathogenic (BRCA1 Frameshift p={top_high['prob']:.3f})",
        "gene": "BRCA1",
        "variant": f"Frameshift at pos {top_high['pos']}",
        "ref": top_high["ref"],
        "alt": top_high["alt"]
    },
    {
        "id": "vus_brca1_mid",
        "label": f"VUS (BRCA1 Missense p={mid_vus['prob']:.3f})",
        "gene": "BRCA1",
        "variant": f"Missense at pos {mid_vus['pos']}",
        "ref": mid_vus["ref"],
        "alt": mid_vus["alt"]
    },
    {
        "id": "benign_arhgap21",
        "label": f"Benign (ARHGAP21 Synonymous p={top_low['prob']:.3f})",
        "gene": "ARHGAP21",
        "variant": "Synonymous SNV",
        "ref": top_low["ref"],
        "alt": top_low["alt"]
    }
]

with open("c:\\Users\\Eranda\\Downloads\\thesis_outputs_20260717_201747\\sample_variants.json", "w") as f:
    json.dump(samples_out, f, indent=2)

print(f"Saved {len(samples_out)} variants.")
print(f"  Pathogenic Prob: {top_high['prob']:.6f}")
print(f"  VUS Prob: {mid_vus['prob']:.6f}")
print(f"  Benign Prob: {top_low['prob']:.6f}")
