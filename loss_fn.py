import torch 

def info_nce_loss(z1, z2, labels, temperature=0.07):
    """
    z1, z2: (B, D) normalized embeddings
    labels: list of room IDs (same = positive)
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    label_vec = torch.cat([labels, labels], dim=0)  # (2B,)

    # Cosine similarity matrix
    sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)
    sim_exp = torch.exp(sim)

    # Mask self-similarities
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim_exp = sim_exp.masked_fill(mask, 0.0)

    # Positive mask: same label but not self
    pos_mask = label_vec.unsqueeze(0) == label_vec.unsqueeze(1)
    pos_mask = pos_mask & (~mask)

    # For each anchor, compute sum of exp(sim) for positives and all
    pos_sim = (sim_exp * pos_mask).sum(dim=1)
    all_sim = sim_exp.sum(dim=1)

    loss = -torch.log(pos_sim / all_sim + 1e-10).mean()
    return loss