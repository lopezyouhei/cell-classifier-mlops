import argparse
import shutil
import subprocess
import sys
from typing import Any

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.config import load_config


def require(cfg: dict, *path: str) -> Any:
    """Fetch a required config key, naming the full path if it's missing"""
    node = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"missing config key: {'.'.join(path)}")
        node = node[key]
    return node


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("version", type=int)
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--deploy-config", default="configs/deploy.yaml")
    p.add_argument("--yes", action="store_true", help="updates the Container App")
    p.add_argument(
        "--skip-gate", action="store_true", help="promote even if the quality gate fails"
    )

    return p.parse_args()


def resolve_target(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve the deployment target."""
    cfg = load_config(args.deploy_config)

    app = cfg.get("container_app")
    resource_group = cfg.get("resource_group")

    return str(app), str(resource_group)


def resolve_az() -> str:
    """
    Locate the az executable.
    """

    az = shutil.which("az")
    if az is None:
        raise RuntimeError("az CLI not found on Path")
    return az


def fetch_version(client: MlflowClient, name: str, version: int):
    try:
        return client.get_model_version(name, str(version))
    except MlflowException as exc:
        raise RuntimeError(f"version {version} of '{name}' not found: {exc}") from exc


def fetch_gate_metric(client: MlflowClient, run_id: str, metric_key: str) -> float:
    if not run_id:
        raise RuntimeError("model version has no run_id, won't promote")
    try:
        run = client.get_run(run_id)
    except MlflowException as exc:
        raise RuntimeError(f"run {run_id} could not read: {exc}") from exc

    value = run.data.metrics.get(metric_key)
    if value is None:
        raise RuntimeError(
            f"run {run_id} has no '{metric_key}' metric."
            "the model was logged from a different run than the one that recorded metrics"
        )
    return float(value)


def clear_stale_tags(client: MlflowClient, name: str, tag_key: str, keep_version: int) -> list[str]:
    """
    Remove the champion tag from every version except the one being promoted.
    """
    cleared = []
    try:
        versions = client.search_model_versions(f"name='{name}'")
    except MlflowException:
        versions = []
        for v in range(1, keep_version + 1):
            try:
                versions.append(client.get_model_version(name, str(v)))
            except MlflowException:
                continue

    for mv in versions:
        if int(mv.version) == keep_version:
            continue
        if mv.tags.get(tag_key):
            client.delete_model_version_tag(name, mv.version, tag_key)
            cleared.append(mv.version)
    return cleared


def deploy(az: str, app: str, resource_group: str, version: int) -> str:
    subprocess.run(
        [
            az,
            "containerapp",
            "update",
            "-n",
            app,
            "-g",
            resource_group,
            "--set-env-vars",
            f"MODEL_VERSION={version}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = subprocess.run(
        [
            az,
            "containerapp",
            "revision",
            "list",
            "-n",
            app,
            "-g",
            resource_group,
            "--query",
            "[0].name",
            "-o",
            "tsv",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return revision.stdout.strip()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    try:
        name = require(cfg, "mlflow", "registered_model_name")
        metric_key = require(cfg, "quality_gate", "metric")
        gate = float(require(cfg, "quality_gate", "min_macro_f1"))
        app, resource_group = resolve_target(args)
    except (KeyError, RuntimeError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    tag_key = cfg["mlflow"].get("champion_tag", "stage")
    tag_value = cfg["mlflow"].get("champion_value", "champion")

    client = MlflowClient()

    try:
        mv = fetch_version(client, name, args.version)
        score = fetch_gate_metric(client, mv.run_id, metric_key)
    except RuntimeError as exc:
        print(f"refusing to promote: {exc}", file=sys.stderr)
        return 1

    print(f"{name} v{args.version}  run={mv.run_id}  {metric_key}={score:.4f}  gate={gate:.4f}")

    if score < gate:
        if not args.skip_gate:
            print("refusing to promote: quality gate not met", file=sys.stderr)
            return 1
        print("WARNING: quality gate not met, proceeding because --skip-gate was passed")

    if not args.yes:
        print(f"dry run - would set MODEL_VERSION={args.version} on {app} ({resource_group})")
        print("re-run with --yes to apply")
        return 0

    client.set_model_version_tag(name, str(args.version), tag_key, tag_value)
    cleared = clear_stale_tags(client, name, tag_key, args.version)
    if cleared:
        print(f"cleared {tag_key}={tag_value} from version(s): {', '.join(cleared)}")

    try:
        az = resolve_az()
        revision = deploy(az, app, resource_group, args.version)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        print(f"deployment failed: {exc}\n{stderr}", file=sys.stderr)
        print(
            f"NOTE: the registry tag was already moved to v{args.version} but the app was not "
            "updated - registry and deployment are now out of sync",
            file=sys.stderr,
        )
        return 1

    print(f"promoted {name} v{args.version} -> {app}")
    print(f"new revision: {revision}")
    print("rollback: python -m scripts.promote <previous-version> --yes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
