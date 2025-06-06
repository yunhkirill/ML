import torch
import torch.nn.functional as F


def info_nce_loss(features, batch_size, temperature=0.5):
    """
    InfoNCE loss for SimCLR: simplified implementation
    """
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    logits = similarity_matrix / temperature

    loss = F.cross_entropy(logits, labels.argmax(dim=1))
    return loss