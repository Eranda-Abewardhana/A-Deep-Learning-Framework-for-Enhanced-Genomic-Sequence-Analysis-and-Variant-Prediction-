import torch
import numpy as np
from backend_server import model, seq_to_onehot, anchor_pad_or_crop

ref_brca1 = "GATCTCCCAGCCCCAGTCGGGAAGGAGCTTTGTTCAGACTTTTGAAAAGCACCAGGATCCTTTGGTTCAGCTACAGGATGGAAAGTCAGGGCTCAAACTGGATTCATTTCCAGGTTGGCTCTGAGATGGATGATACTGAAGCTGATCCTCTTTCAAGCTCAGCCAGACTGCTCTCTTCAGAAATATCACTTGCATGGCCTGGAAAGCCAGCTCTTCTACCATCCATGTCAGAGTCATGGAAACACTCTCTAACCTCCTTTGTTTTACCTCTATCTTGTACTCCATAAACCTCTCAGTACAGAACATACAACTGATCACCCTGAATGGATCCCAAAGAGCTGATAAATATAAATTGTGCTTAATTAAGAAGTCTTCCTTCAAAGAAGCAATGGCCAATCTAGTGAAGAATCAGGCAGCTGAATTTATTCAGCCTTCATTGCAAAAGCGCTGATATTTTTAGAGATCTCTTGAGAATCTGATCAGAAAATCTAATGGATGCAAAGCTATTTAAGCATTGTACC"

# Pathogenic sample: Frameshift Deletion (pos 40-45)
alt_pathogenic = ref_brca1[:40] + ref_brca1[45:]

# Benign sample: Single nucleotide polymorphism at pos 402
alt_benign = ref_brca1[:402] + 'A' + ref_brca1[403:]

def eval_v(ref, alt):
    r_oh = seq_to_onehot(ref)
    a_oh = seq_to_onehot(alt)
    r_t, a_t, m_pos = anchor_pad_or_crop(r_oh, a_oh)
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(r_t).unsqueeze(0), torch.from_numpy(a_t).unsqueeze(0))).item()
    return p, m_pos

p_path, m_path = eval_v(ref_brca1, alt_pathogenic)
p_ben, m_ben = eval_v(ref_brca1, alt_benign)

with open("c:\\Users\\Eranda\\Downloads\\thesis_outputs_20260717_201747\\sample_probs.txt", "w") as f:
    f.write(f"PATHOGENIC: prob={p_path:.6f}, mut_pos={m_path}\n")
    f.write(f"ALT_PATH: {alt_pathogenic}\n")
    f.write(f"BENIGN:     prob={p_ben:.6f}, mut_pos={m_ben}\n")
    f.write(f"ALT_BEN:  {alt_benign}\n")
