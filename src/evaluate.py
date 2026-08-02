import json
import sys

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.config import ARTIFACTS, load_config, parse_args
from src.model import build_head
from src.train import load_embeddings


def plot_confusion(cm: np.ndarray, labels: list[str], path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    Xte, yte = load_embeddings("test")
    chkpt = torch.load(ARTIFACTS / "head.pt", map_location="cpu")
    with open(ARTIFACTS / "splits.json") as f:
        meta = json.load(f)
    labels = [meta["labels"][str(i)] for i in range(meta["n_classes"])]

    head = build_head(cfg, chkpt["in_dim"], chkpt["n_classes"])
    head.load_state_dict(chkpt["state_dict"])
    head.eval()

    with torch.no_grad():
        preds = head(Xte).argmax(dim=1).numpy()
    y_true = yte.numpy()

    report = classification_report(y_true, preds, target_names=labels, output_dict=True)
    metrics = {
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted")),
        "per_class_recall": {c: report[c]["recall"] for c in labels},
    }
    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion(confusion_matrix(y_true, preds), labels, ARTIFACTS / "confusion_matrix.png")
    print(json.dumps(metrics, indent=2))

    gate = cfg["quality_gate"]["min_macro_f1"]
    if metrics["macro_f1"] < gate:
        print(f"Quality gate failed! Macro-F1 {metrics['macro_f1']:.4f} < {gate}")
        sys.exit(1)
    print(f"Quality gate passed ({metrics['macro_f1']:.4f} >= {gate})")


if __name__ == "__main__":
    main()
