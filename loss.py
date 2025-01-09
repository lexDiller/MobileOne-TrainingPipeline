import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, features, labels):
        features = features.float()
        features = F.normalize(features, p=2, dim=1)

        dist_mat = torch.cdist(features, features)

        n = labels.size(0)
        mask_pos = labels.expand(n, n).eq(labels.expand(n, n).t())
        mask_neg = ~mask_pos

        dist_ap = (dist_mat * mask_pos.float()).max(dim=1)[0]
        dist_an = (dist_mat * mask_neg.float() + mask_pos.float() * 1e9).min(dim=1)[0]

        loss = torch.clamp(dist_ap - dist_an + self.margin, min=0.0)

        valid_mask = loss > 0
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=features.device)

        return loss


class CombinedLoss(nn.Module):
    def __init__(self, margin=0.3, triplet_weight=0.1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=margin)
        self.triplet_weight = triplet_weight

    def forward(self, features, logits, labels):
        ce_loss = self.ce_loss(logits, labels)
        trip_loss = self.triplet_loss(features, labels)
        return ce_loss + self.triplet_weight * trip_loss
    