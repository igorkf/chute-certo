"""Start the prediction API, loading the model from MLflow/DagsHub."""

import os

import dagshub
import uvicorn
from dotenv import load_dotenv

load_dotenv()

dagshub.init(
    repo_owner=os.environ["DAGSHUB_REPO_OWNER"],
    repo_name=os.environ["DAGSHUB_REPO_NAME"],
    mlflow=True,
)

if __name__ == "__main__":
    uvicorn.run(
        "chute_certo.serving.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
