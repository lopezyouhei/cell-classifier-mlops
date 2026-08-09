import json

import mlflow
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from src.config import ARTIFACTS, load_config, parse_args, set_seed
from src.model import build_head


def load_embeddings(split: str):
    d = np.load(ARTIFACTS / "embeddings" / f"{split}.npz")
    return torch.from_numpy(d["X"]).float(), torch.from_numpy(d["y"]).long()


def macro_f1(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1).cpu().numpy()
    return float(f1_score(y.cpu().numpy(), preds, average="macro"))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    Xtr, ytr = load_embeddings("train")
    Xva, yva = load_embeddings("val")
    weights = torch.from_numpy(np.load(ARTIFACTS / "class_weights.npy")).float()

    with open(ARTIFACTS / "splits.json") as f:
        meta = json.load(f)
    n_classes = meta["n_classes"]

    head = build_head(cfg, in_dim=Xtr.shape[1], n_classes=n_classes)
    criterion = nn.CrossEntropyLoss(weight=weights if cfg["train"]["class_weighted_loss"] else None)
    opt = torch.optim.AdamW(
        head.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )

    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "backbone": cfg["backbone"]["name"],
                "backbone_frozen": cfg["backbone"]["frozen"],
                "hidden_dim": cfg["head"]["hidden_dim"],
                "lr": cfg["train"]["lr"],
                "epochs": cfg["train"]["epochs"],
                "class_weighted_loss": cfg["train"]["class_weighted_loss"],
                "seed": cfg["seed"],
                "imbalance_ratio": meta["imbalance_ratio"],
            }
        )

        batch_size = cfg["train"]["batch_size"]
        best_f1, best_state, patience = -1.0, None, 0

        for epoch in range(cfg["train"]["epochs"]):
            head.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), batch_size):
                idx = perm[i : i + batch_size]
                opt.zero_grad()
                loss = criterion(head(Xtr[idx]), ytr[idx])
                loss.backward()
                opt.step()

            head.eval()
            with torch.no_grad():
                val_f1 = macro_f1(head(Xva), yva)
            mlflow.log_metric("val_macro_f1", val_f1, epoch)

            if val_f1 > best_f1:
                best_f1, patience = val_f1, 0
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
            else:
                patience += 1
                if patience >= cfg["train"]["early_stopping_patience"]:
                    print(f"early stop at epoch {epoch}")
                    break

        assert best_state is not None
        head.load_state_dict(best_state)
        mlflow.log_metric("best_val_macro_f1", best_f1)
        torch.save(
            {"state_dict": best_state, "in_dim": Xtr.shape[1], "n_classes": n_classes},
            ARTIFACTS / "head.pt",
        )

        mlflow.pytorch.log_model(
            head,
            artifact_path="head",
            input_example=Xtr[:1].numpy(),
            registered_model_name=cfg["mlflow"]["registered_model_name"],
        )
        print(f"run {run.info.run_id} | best val macro-F1 {best_f1:.4f}")


if __name__ == "__main__":
    main()
