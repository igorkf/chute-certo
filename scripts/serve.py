"""Start the prediction API, loading the model from MLflow/DagsHub."""

import os

import mlflow
import uvicorn
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(
    f"https://dagshub.com/{os.environ['DAGSHUB_REPO_OWNER']}"
    f"/{os.environ['DAGSHUB_REPO_NAME']}.mlflow"
)

if __name__ == "__main__":
    uvicorn.run(
        "chute_certo.serving.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
