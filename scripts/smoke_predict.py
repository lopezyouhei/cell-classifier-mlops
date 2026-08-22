import argparse
import json
import sys
import time
from pathlib import Path

import requests

p = argparse.ArgumentParser()
p.add_argument("--url", required=True)
p.add_argument("--expected-sha")
p.add_argument("--expected-model-version")
p.add_argument("--fixture", default="tests/fixtures/sample.png")
p.add_argument("--expected", default="tests/fixtures/expected.json")
p.add_argument("--timeout", type=int, default=300)
args = p.parse_args()

base = args.url.rstrip("/")


def fail(message: str) -> None:
    print(f"Smoke Fail: {message}")
    sys.exit(1)


# check if service is up, with timeout due to cold start possibility
deadline = time.time() + args.timeout
while time.time() < deadline:
    try:
        if requests.get(f"{base}/health", timeout=10).status_code == 200:
            break
    except requests.RequestException:
        pass
    time.sleep(5)
else:
    fail(f"/health did not return 200 within {args.timeout}s")

# confirm what is being served
version = requests.get(f"{base}/version", timeout=10).json()
print(f"serving: {json.dumps(version)}")

if version.get("stub_mode"):
    fail("service is running in stub mode")
if args.expected_sha and version.get("git_sha") != args.expected_sha:
    fail(f"expected git sha: {args.expected_sha}, got {version.get('git_sha')}")
if args.expected_model_version and str(version.get("model_version")) != str(
    args.expected_model_version
):
    fail(
        f"expected model version: {args.expected_model_version}, got {version.get('model_version')}"
    )

# inference and check
expected = json.loads(Path(args.expected).read_text())["expected_class"]
with open(args.fixture, "rb") as f:
    r = requests.post(f"{base}/predict", files={"file": f}, timeout=60)

if r.status_code != 200:
    fail(f"/predict returned {r.status_code}: {r.text[:200]}")

body = r.json()
print(f"prediction: {json.dumps(body)}")

if body.get("predicted_class") != expected:
    fail(f"expected class: {expected}, got {body.get('predicted_class')}")

print("Smoke Passed")
