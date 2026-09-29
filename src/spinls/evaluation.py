"""Small neural-classifier evaluation helper retained for the CIFAR notebook."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support


@torch.inference_mode()
def evaluate(dataloader, model, threshold=0.5, multilabel=False):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    targets, predictions = [], []
    total_loss, total_examples = 0.0, 0
    try:
        for x, y in dataloader:
            output = model(x.to(device))
            loss = (F.binary_cross_entropy_with_logits(output, y.to(device)) if multilabel
                    else F.cross_entropy(output, y.to(device)))
            total_loss += loss.item() * len(x)
            total_examples += len(x)
            targets.append(y.cpu().numpy())
            predictions.append((output.sigmoid() > threshold).cpu().numpy() if multilabel
                               else output.argmax(dim=1).cpu().numpy())
    finally:
        model.train(was_training)
    if not total_examples:
        raise ValueError("Cannot evaluate an empty dataloader")
    precision, recall, f1, support = precision_recall_fscore_support(
        np.concatenate(targets), np.concatenate(predictions),
        labels=list(range(len(dataloader.dataset.classes))), zero_division=0,
    )
    print(f"{'class':<12} {'precision':>9} {'recall':>7} {'f1':>7} {'support':>8}")
    for label, p, r, f, n in zip(dataloader.dataset.classes, precision, recall, f1, support):
        print(f"{label:<12} {p:9.3f} {r:7.3f} {f:7.3f} {n:8d}")
    return total_loss / total_examples, float(f1.mean())
