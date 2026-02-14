import torch


def _flatten(y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y_pred = y_pred.reshape(y_pred.size(0), -1)
    y_true = y_true.reshape(y_true.size(0), -1)
    return y_pred, y_true


def dice_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    y_pred, y_true = _flatten(probs, target)
    intersection = (y_pred * y_true).sum(dim=1)
    union = y_pred.sum(dim=1) + y_true.sum(dim=1)
    return ((2 * intersection + eps) / (union + eps)).mean()


def iou_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    y_pred, y_true = _flatten(probs, target)
    intersection = (y_pred * y_true).sum(dim=1)
    union = y_pred.sum(dim=1) + y_true.sum(dim=1) - intersection
    return ((intersection + eps) / (union + eps)).mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return 1.0 - dice_score(logits, target, eps=eps)
