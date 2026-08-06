import numpy as np
import torch

from src.config import ARTIFACTS, RAW_DATA, load_config, parse_args, set_seed
from src.data import load_splits


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_backbone(cfg: dict, device=torch.device):
    import timm

    model = timm.create_model(
        cfg["backbone"]["name"], pretrained=cfg["backbone"]["pretrained"], num_classes=0
    )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False  # just inferencing the model
    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    return model, transform


@torch.inference_mode()
def embed_split(
    model, transform, dataset, device, batch_size: int, num_workers: int
) -> tuple[np.ndarray, np.ndarray]:
    from torch.utils.data import DataLoader

    dataset.transform = transform
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    feats, labels = [], []
    for x, y in loader:
        feats.append(model(x.to(device)).cpu().numpy())
        labels.append(np.asarray(y).ravel())
    return np.concatenate(feats), np.concatenate(labels)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    out_dir = ARTIFACTS / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg["backbone"]["device"])
    model, transform = build_backbone(cfg, device)
    train, val, test, _ = load_splits(cfg, RAW_DATA)

    for name, ds in [("train", train), ("val", val), ("test", test)]:
        X, y = embed_split(
            model,
            transform,
            ds,
            device,
            cfg["backbone"]["batch_size"],
            cfg["backbone"]["num_workers"],
        )
        np.savez_compressed(out_dir / f"{name}.npz", X=X, y=y)
        print(f"{name}: {X.shape} embeddings cached")


if __name__ == "__main__":
    main()
