import json
from pathlib import Path

import numpy as np

from src.config import ARTIFACTS, CONFIGS, DATA, RAW_DATA, load_config, parse_args, set_seed


def load_splits(cfg: dict, raw_data_dir: Path | str):
    import medmnist
    from medmnist import INFO

    flag = cfg["data"]["dataset"]
    info = INFO[flag]
    DataClass = getattr(medmnist, info["python_class"])

    kwargs = dict(
        root=raw_data_dir,
        download=True,
        size=cfg["data"]["size"],
    )

    return (
        DataClass(split="train", **kwargs),
        DataClass(split="val", **kwargs),
        DataClass(split="test", **kwargs),
        info,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ARTIFACTS.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    RAW_DATA.mkdir(exist_ok=True)

    train, val, test, info = load_splits(cfg, RAW_DATA)

    y_train = np.asarray(train.labels).ravel()
    n_classes = len(info["label"])
    counts = np.bincount(y_train, minlength=n_classes)

    # inverse-frequency weights to stabilize the loss
    weights = counts.sum() / (n_classes * np.maximum(counts, 1))
    weights = weights / weights.mean()

    np.save(ARTIFACTS / "class_weights.npy", weights.astype(np.float32))

    summary = {
        "dataset": cfg["data"]["dataset"],
        "size": cfg["data"]["size"],
        "n_classes": n_classes,
        "labels": info["label"],
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "train_class_counts": counts.tolist(),
        "imbalance_ratio": float(counts.max() / max(counts.min(), 1)),
    }

    with open(ARTIFACTS / "splits.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # store splits in 'configs' as the docker image wont have access to 'artifacts'
    # necessary so that predictions can be displayed with corresponding label
    # instead of indeces
    with open(CONFIGS / "labels.json", "w") as f:
        json.dump(
            {
                "labels": info["label"],
                "n_classes": n_classes,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
