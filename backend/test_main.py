import json
import math
import unittest

from fastapi import HTTPException

from main import DATASETS, PredictionRequest, get_dataset, list_datasets, predict


class NaiveBayesApiTests(unittest.TestCase):
    def test_notebook_splits_are_preserved(self) -> None:
        summaries = list_datasets()
        self.assertEqual(len(summaries), 4)
        for summary in summaries:
            with self.subTest(dataset=summary["id"]):
                detail = get_dataset(summary["id"])
                self.assertEqual(len(detail["trainingRows"]), 24)
                self.assertEqual(len(detail["testRows"]), 8)

    def test_prediction_includes_prior_and_smoothed_feature_steps(self) -> None:
        dataset = DATASETS["fruit"]
        features = {
            feature["key"]: feature["categories"][0]
            for feature in dataset["features"]
        }
        result = predict("fruit", PredictionRequest(features=features, alpha=1))

        self.assertEqual(result["trainingCount"], 24)
        self.assertEqual(sum(result["classCounts"].values()), 24)
        self.assertAlmostEqual(
            sum(class_result["posterior"] for class_result in result["results"]),
            1.0,
        )
        for class_result in result["results"]:
            self.assertAlmostEqual(
                class_result["prior"],
                class_result["classCount"] / result["trainingCount"],
            )
            for detail in class_result["details"]:
                expected = (detail["matchingCount"] + 1) / (
                    detail["classCount"] + detail["categoryCount"]
                )
                self.assertAlmostEqual(detail["probability"], expected)
                self.assertAlmostEqual(
                    detail["logContribution"], math.log(expected)
                )

    def test_binary_features_remain_numeric(self) -> None:
        dataset = DATASETS["machine"]
        features = {
            feature["key"]: feature["categories"][0]
            for feature in dataset["features"]
        }
        self.assertTrue(all(isinstance(value, int) for value in features.values()))
        result = predict("machine", PredictionRequest(features=features))
        self.assertIn(result["prediction"], dataset["classes"])

    def test_zero_smoothing_returns_json_safe_zero_probability(self) -> None:
        dataset = DATASETS["fruit"]
        features = {
            feature["key"]: feature["categories"][0]
            for feature in dataset["features"]
        }
        result = predict("fruit", PredictionRequest(features=features, alpha=0))
        self.assertTrue(
            any(
                detail["probability"] == 0 and detail["logContribution"] is None
                for class_result in result["results"]
                for detail in class_result["details"]
            )
        )
        json.dumps(result, allow_nan=False)

    def test_posterior_is_normalized_for_each_notebook_dataset(self) -> None:
        for dataset_id, dataset in DATASETS.items():
            with self.subTest(dataset=dataset_id):
                features = {
                    feature["key"]: feature["categories"][0]
                    for feature in dataset["features"]
                }
                result = predict(dataset_id, PredictionRequest(features=features))
                self.assertAlmostEqual(
                    sum(class_result["posterior"] for class_result in result["results"]),
                    1.0,
                )

    def test_unknown_feature_value_is_rejected(self) -> None:
        dataset = DATASETS["fruit"]
        features = {
            feature["key"]: feature["categories"][0]
            for feature in dataset["features"]
        }
        features["skin_color"] = "purple"
        with self.assertRaises(HTTPException):
            predict("fruit", PredictionRequest(features=features))


if __name__ == "__main__":
    unittest.main()
