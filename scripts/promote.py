import argparse

from mlflow import MlflowClient

from src.config import load_config

p = argparse.ArgumentParser()
p.add_argument("version")
p.add_argument("--config", default="configs/train.yaml")
args = p.parse_args()

cfg = load_config(args.config)
name = cfg["mlflow"]["registered_model_name"]
alias = cfg["mlflow"]["champion_alias"]

client = MlflowClient()
client.set_registered_model_alias(name, alias, args.version)
print(f"{name}@{alias} -> version {args.version}")
