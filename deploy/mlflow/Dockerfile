FROM python:3.12-slim

RUN pip install --no-cache-dir "mlflow==3.15.0" "psycopg2-binary" "boto3"

EXPOSE 8080

CMD mlflow server --host 0.0.0.0 --port 8080 --workers 1 --backend-store-uri "$MLFLOW_BACKEND_URI" --artifacts-destination "$MLFLOW_ARTIFACTS_URI" --serve-artifacts --allowed-hosts "*"