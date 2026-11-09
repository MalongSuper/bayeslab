import json
import math
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


DATA_PATH = Path(__file__).with_name("datasets.json")
DATASETS: dict[str, dict[str, Any]] = {
    dataset["id"]: dataset for dataset in json.loads(DATA_PATH.read_text())
}

app = FastAPI(title="Naive Bayes Learning API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    features: dict[str, str | int]
    alpha: float = Field(default=1.0, ge=0, le=10)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/datasets")
def list_datasets() -> list[dict[str, Any]]:
    return [
        {
            "id": dataset["id"],
            "name": dataset["name"],
            "description": dataset["description"],
            "featureCount": len(dataset["features"]),
            "trainingCount": sum(
                row[dataset["split"]] == "train" for row in dataset["rows"]
            ),
            "testCount": sum(
                row[dataset["split"]] != "train" for row in dataset["rows"]
            ),
            "target": dataset["target"],
        }
        for dataset in DATASETS.values()
    ]


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> dict[str, Any]:
    dataset = DATASETS.get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return {
        **{key: value for key, value in dataset.items() if key != "rows"},
        "trainingRows": [
            row for row in dataset["rows"] if row[dataset["split"]] == "train"
        ],
        "testRows": [
            row for row in dataset["rows"] if row[dataset["split"]] != "train"
        ],
    }


@app.post("/api/datasets/{dataset_id}/predict")
def predict(dataset_id: str, request: PredictionRequest) -> dict[str, Any]:
    dataset = DATASETS.get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    feature_specs = dataset["features"]
    expected_features = {feature["key"] for feature in feature_specs}
    if set(request.features) != expected_features:
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one value for every feature.",
        )
    for feature in feature_specs:
        if request.features[feature["key"]] not in feature["categories"]:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid value for {feature['key']}.",
            )

    training_rows = [
        row for row in dataset["rows"] if row[dataset["split"]] == "train"
    ]
    classes = dataset["classes"]
    total = len(training_rows)
    class_counts = {
        label: sum(row[dataset["target"]] == label for row in training_rows)
        for label in classes
    }
    results = []

    for label in classes:
        class_count = class_counts[label]
        prior = class_count / total
        log_score = math.log(prior)
        details = []
        for feature in feature_specs:
            key = feature["key"]
            value = request.features[key]
            matching_count = sum(
                row[dataset["target"]] == label and row[key] == value
                for row in training_rows
            )
            category_count = len(feature["categories"])
            denominator = class_count + request.alpha * category_count
            numerator = matching_count + request.alpha
            probability = numerator / denominator if denominator else 0.0
            log_contribution = math.log(probability) if probability else -math.inf
            log_score += log_contribution
            details.append(
                {
                    "feature": key,
                    "featureLabel": feature["label"],
                    "value": value,
                    "matchingCount": matching_count,
                    "classCount": class_count,
                    "categoryCount": category_count,
                    "alpha": request.alpha,
                    "numerator": numerator,
                    "denominator": denominator,
                    "probability": probability,
                    "logContribution": (
                        log_contribution if math.isfinite(log_contribution) else None
                    ),
                }
            )
        results.append(
            {
                "classLabel": label,
                "classCount": class_count,
                "prior": prior,
                "logPrior": math.log(prior),
                "logScore": log_score if math.isfinite(log_score) else None,
                "details": details,
                "_comparisonScore": log_score,
            }
        )

    prediction = max(results, key=lambda result: result["_comparisonScore"])["classLabel"]
    maximum_score = max(result["_comparisonScore"] for result in results)
    if math.isfinite(maximum_score):
        posterior_weights = [
            math.exp(result["_comparisonScore"] - maximum_score)
            if math.isfinite(result["_comparisonScore"])
            else 0.0
            for result in results
        ]
        posterior_total = sum(posterior_weights)
        for result, weight in zip(results, posterior_weights):
            result["posterior"] = weight / posterior_total
    else:
        for result in results:
            result["posterior"] = None

    for result in results:
        del result["_comparisonScore"]
    return {
        "prediction": prediction,
        "alpha": request.alpha,
        "trainingCount": total,
        "classCounts": class_counts,
        "results": results,
    }
