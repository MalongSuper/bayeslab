# Bayes Lab

An interactive React learning app backed by a small FastAPI service. It uses the
four categorical datasets from the included Jupyter notebooks: fruit quality,
delivery delays, machine faults, and network access.

## Run locally

Requirements: Node.js/npm and Python 3.10 or newer.

```bash
npm install
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
npm run dev
```

Open the Vite URL printed by npm (normally http://localhost:5173). The dev
script starts both the React app and the API; Vite forwards `/api` requests to
FastAPI at http://localhost:8000. Stop both processes with Ctrl+C.

## What the lesson demonstrates

- Conditional probability: `P(A | B) = P(A ∩ B) / P(B)`.
- Naive Bayes: `P(C | X) ∝ P(C) × ∏ P(xᵢ | C)`, assuming features are
  conditionally independent given the class.
- Categorical Laplace smoothing:
  `P(x = v | C = c) = (count(v, c) + α) / (count(c) + αK)`.
- Prediction by comparing unnormalized log scores:
  `log P(c) + Σ log P(xᵢ | c)`.
- A live prior-probability summary and full smoothed likelihood table for every
  category in the selected dataset, followed by normalized `P(C | X)` values
  for the entered example.

The original notebook train/test split is preserved. Only training rows are
used to calculate priors and likelihoods; test examples are shown separately.
Adjust α in the app to see how smoothing changes the conditional probabilities
and class scores.
