import { useEffect, useMemo, useState } from "react";

const formatNumber = (value, digits = 4) =>
  value !== null && Number.isFinite(value) ? Number(value).toFixed(digits) : "-∞";

const displayValue = (value) => (value === 1 ? "1 · Present" : value === 0 ? "0 · Absent" : value);
const parseCategory = (feature, value) =>
  typeof feature?.categories[0] === "number" ? Number(value) : value;
const className = (value) => value.replaceAll("_", " ");
const formatAlpha = (value) => Number(value).toString();

function App() {
  const [datasets, setDatasets] = useState([]);
  const [datasetId, setDatasetId] = useState("");
  const [dataset, setDataset] = useState(null);
  const [values, setValues] = useState({});
  const [alpha, setAlpha] = useState(1);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [conditionalFeature, setConditionalFeature] = useState("");
  const [conditionalValue, setConditionalValue] = useState("");

  useEffect(() => {
    fetch("/api/datasets")
      .then((response) => {
        if (!response.ok) throw new Error("Could not load the example datasets.");
        return response.json();
      })
      .then((items) => {
        setDatasets(items);
        setDatasetId(items[0]?.id ?? "");
      })
      .catch((cause) => setError(cause.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!datasetId) return;
    setDataset(null);
    setPrediction(null);
    fetch(`/api/datasets/${datasetId}`)
      .then((response) => {
        if (!response.ok) throw new Error("Could not load this dataset.");
        return response.json();
      })
      .then((detail) => {
        setDataset(detail);
        const initialValues = Object.fromEntries(
          detail.features.map((feature) => [feature.key, feature.categories[0]]),
        );
        setValues(initialValues);
        setConditionalFeature(detail.features[0]?.key ?? "");
        setConditionalValue(detail.features[0]?.categories[0] ?? "");
      })
      .catch((cause) => setError(cause.message));
  }, [datasetId]);

  useEffect(() => {
    if (!dataset || Object.keys(values).length !== dataset.features.length) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      fetch(`/api/datasets/${dataset.id}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: values, alpha: Number(alpha) }),
        signal: controller.signal,
      })
        .then((response) => {
          if (!response.ok) throw new Error("Prediction failed. Check the selected values.");
          return response.json();
        })
        .then(setPrediction)
        .catch((cause) => {
          if (cause.name !== "AbortError") setError(cause.message);
        });
    }, 100);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [dataset, values, alpha]);

  const selectedSummary = datasets.find((item) => item.id === datasetId);
  const priorStatistics = useMemo(() => {
    if (!dataset) return [];
    return dataset.classes.map((classLabel) => {
      const count = dataset.trainingRows.filter(
        (row) => row[dataset.target] === classLabel,
      ).length;
      return {
        classLabel,
        count,
        probability: count / dataset.trainingRows.length,
      };
    });
  }, [dataset]);
  const probabilityRows = useMemo(() => {
    if (!dataset) return [];
    return dataset.features.flatMap((feature) =>
      feature.categories.map((category) => ({
        feature,
        category,
        likelihoods: dataset.classes.map((classLabel) => {
          const classCount = dataset.trainingRows.filter(
            (row) => row[dataset.target] === classLabel,
          ).length;
          const matchingCount = dataset.trainingRows.filter(
            (row) =>
              row[dataset.target] === classLabel &&
              row[feature.key] === category,
          ).length;
          const denominator =
            classCount + Number(alpha) * feature.categories.length;
          const numerator = matchingCount + Number(alpha);
          return {
            classLabel,
            matchingCount,
            classCount,
            denominator,
            numerator,
            probability: denominator ? numerator / denominator : null,
          };
        }),
      })),
    );
  }, [dataset, alpha]);
  const selectedConditionalFeature = dataset?.features.find(
    (feature) => feature.key === conditionalFeature,
  );
  const conditionalStats = useMemo(() => {
    if (!selectedConditionalFeature || conditionalValue === "") return [];
    const row = probabilityRows.find(
      (item) =>
        item.feature.key === selectedConditionalFeature.key &&
        item.category === conditionalValue,
    );
    return row?.likelihoods.map((likelihood) => ({
      ...likelihood,
      categoryCount: selectedConditionalFeature.categories.length,
    })) ?? [];
  }, [probabilityRows, selectedConditionalFeature, conditionalValue]);

  function changeConditionalFeature(key) {
    const feature = dataset.features.find((item) => item.key === key);
    setConditionalFeature(key);
    setConditionalValue(feature?.categories[0] ?? "");
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Bayes Lab home">
          <span className="brand-mark">b.</span>
          <span>bayes<span className="brand-light">lab</span></span>
        </a>
        <nav className="top-nav" aria-label="Main navigation">
          <a href="#learn">The basics</a>
          <a href="#predict">Try a prediction</a>
          <a href="#probabilities">Probability table</a>
          <a href="#calculation">Calculation</a>
        </nav>
        <span className="topbar-note"><span className="live-dot" /> Interactive lesson</span>
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero-copy">
            <div className="eyebrow"><span className="eyebrow-line" /> A PROBABILITY PLAYGROUND</div>
            <h1>Make predictions.<br /><span>See the probability.</span></h1>
            <p>
              Learn Naive Bayes by changing the evidence and following every probability
              from the training data to the final prediction.
            </p>
            <a className="hero-link" href="#predict">Start exploring <span>↓</span></a>
          </div>
          <div className="hero-visual" aria-hidden="true">
            <div className="orbit orbit-one" />
            <div className="orbit orbit-two" />
            <div className="hero-node node-center">P(C|X)</div>
            <div className="hero-node node-a">prior</div>
            <div className="hero-node node-b">evidence</div>
            <div className="hero-node node-c">predict</div>
            <div className="hero-visual-caption">LEARN BY DOING <span>✳</span> 04 EXAMPLE DATASETS</div>
          </div>
        </section>

        {error && <div className="error-banner" role="alert">{error}</div>}
        {loading && <div className="loading-state">Preparing your datasets…</div>}

        <section className="learn-section" id="learn">
          <div className="section-heading">
            <div>
              <div className="eyebrow dark-eyebrow">THE IDEA, IN THREE STEPS</div>
              <h2>Small assumptions. <span>Useful predictions.</span></h2>
            </div>
            <p>Naive Bayes combines what is common with what you observe.</p>
          </div>
          <div className="concept-grid">
            <article className="concept-card">
              <div className="concept-number">01 <span>FOUNDATION</span></div>
              <h3>Conditional probability</h3>
              <p>How likely is an event <em>given</em> that we already know something else?</p>
              <div className="equation"><span>P(A | B)</span><span className="eq-equals">=</span><span className="fraction"><b>P(A ∩ B)</b><i>P(B)</i></span></div>
              <div className="concept-foot">“Probability of A, knowing B”</div>
            </article>
            <article className="concept-card formula-card">
              <div className="concept-number">02 <span>THE MODEL</span></div>
              <h3>Bayes, feature by feature</h3>
              <p>Compare the classes using their prior and each feature’s likelihood.</p>
              <div className="equation equation-wide">P(C | X) ∝ P(C) × ∏ P(xᵢ | C)</div>
              <div className="concept-foot">Features are treated as conditionally independent given C.</div>
            </article>
            <article className="concept-card smoothing-card">
              <div className="concept-number">03 <span>NO ZEROES</span></div>
              <h3>Laplace smoothing</h3>
              <p>Add a little count to every category so unseen evidence doesn’t erase a class.</p>
              <div className="equation equation-wide smoothing-equation"><span>P(x=v | C=c)</span><span>=</span><span>(count + α) / (class count + αK)</span></div>
              <div className="concept-foot">α = 1 adds one pseudo-count; K is the category count.</div>
            </article>
          </div>
        </section>

        <section className="workbench" id="predict">
          <div className="workbench-heading">
            <div>
              <div className="eyebrow dark-eyebrow">YOUR TURN · INTERACTIVE WORKBENCH</div>
              <h2>Give the model some evidence.</h2>
            </div>
            <span className="step-pill"><span>01</span> CHOOSE DATASET & FEATURES</span>
          </div>

          <div className="workbench-grid">
            <aside className="input-panel">
              <div className="panel-title"><span className="panel-icon">⌘</span><div><h3>Set up an example</h3><p>Choose an example dataset and input.</p></div></div>
              <label className="field-label" htmlFor="dataset-select">EXAMPLE DATASET</label>
              <select id="dataset-select" value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>
                {datasets.map((item) => <option value={item.id} key={item.id}>{item.name}</option>)}
              </select>
              {selectedSummary && <div className="dataset-summary">{selectedSummary.description}<div className="dataset-counts"><span>{selectedSummary.trainingCount} train</span><span>{selectedSummary.testCount} test</span></div></div>}

              {dataset && <div className="feature-fields">
                <div className="field-label feature-label">FEATURE VALUES <span>{dataset.features.length} FEATURES</span></div>
                {dataset.features.map((feature, index) => (
                  <label className="feature-field" key={feature.key}>
                    <span><i>{String(index + 1).padStart(2, "0")}</i>{feature.label}</span>
                    <select value={values[feature.key] ?? ""} onChange={(event) => setValues((current) => ({ ...current, [feature.key]: parseCategory(feature, event.target.value) }))}>
                      {feature.categories.map((category) => <option value={category} key={category}>{displayValue(category)}</option>)}
                    </select>
                  </label>
                ))}
              </div>}

              <div className="alpha-control">
                <div className="alpha-label"><label htmlFor="alpha-slider">LAPLACE SMOOTHING <span>α</span></label><output>{Number(alpha).toFixed(1)}</output></div>
                <input id="alpha-slider" type="range" min="0" max="3" step="0.5" value={alpha} onChange={(event) => setAlpha(event.target.value)} />
                <div className="range-ends"><span>0 · none</span><span>3 · stronger</span></div>
                <p>Adjust α and watch every conditional probability change.</p>
              </div>
            </aside>

            <section className="result-panel" aria-live="polite">
              <div className="result-topline"><span>MODEL OUTPUT</span><span className="live-tag"><i /> LIVE</span></div>
              {prediction ? <>
                <div className="prediction-callout">
                  <div><span className="prediction-kicker">PREDICTED CLASS</span><h3>{className(prediction.prediction)}</h3></div>
                  <div className="prediction-check">✓</div>
                </div>
                <p className="score-intro">The model chooses the class with the highest unnormalized log score.</p>
                <div className="score-grid">
                  {prediction.results.map((result) => (
                    <article className={`score-card ${result.classLabel === prediction.prediction ? "score-winner" : ""}`} key={result.classLabel}>
                      <div className="score-class">{className(result.classLabel)}{result.classLabel === prediction.prediction && <span>WINNER</span>}</div>
                      <div className="score-value">{formatNumber(result.logScore, 3)}</div>
                      <div className="score-caption">unnormalized log score</div>
                    </article>
                  ))}
                </div>
                <div className="score-note"><span>i</span> These are log scores, not normalized probabilities. Larger (less negative) wins.</div>
                <div className="posterior-heading">
                  <strong>Normalized posterior probability</strong>
                  <code>P(C | X) = exp(log score) / Σ exp(log score)</code>
                </div>
                <div className="posterior-grid">
                  {prediction.results.map((result) => (
                    <div className="posterior-card" key={result.classLabel}>
                      <div>
                        <span>{className(result.classLabel)}</span>
                        <strong>{result.posterior === null ? "N/A" : `${(result.posterior * 100).toFixed(1)}%`}</strong>
                      </div>
                      <div className="posterior-track">
                        <i style={{ width: `${(result.posterior ?? 0) * 100}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </> : <div className="result-placeholder">Choose a dataset to calculate a prediction.</div>}
            </section>
          </div>
        </section>

        {dataset && <section className="probability-section" id="probabilities">
          <div className="section-heading probability-heading">
            <div>
              <div className="eyebrow dark-eyebrow">MODEL PROBABILITIES · TRAINING DATA ONLY</div>
              <h2>Start with priors. <span>Then look up every likelihood.</span></h2>
            </div>
            <span className="step-pill"><span>02</span> α = {Number(alpha).toFixed(1)} SMOOTHING</span>
          </div>
          <p className="probability-intro">
            The prior is the fraction of training examples in each class. Each
            conditional probability uses categorical Laplace smoothing:
            (count + α) / (class count + α × number of categories).
          </p>
          <div className="prior-cards">
            {priorStatistics.map((prior) => (
              <article className="prior-card" key={prior.classLabel}>
                <span className="prior-card-label">PRIOR PROBABILITY · P(C)</span>
                <h3>{className(prior.classLabel)}</h3>
                <strong>{formatNumber(prior.probability, 3)}</strong>
                <code>{prior.count} / {dataset.trainingRows.length}</code>
              </article>
            ))}
          </div>
          <div className="likelihood-table-card">
            <div className="likelihood-table-heading">
              <span>CONDITIONAL PROBABILITY TABLE</span>
              <span>All categories · α = {Number(alpha).toFixed(1)}</span>
            </div>
            <div className="table-scroll">
              <table className="likelihood-table">
                <thead>
                  <tr>
                    <th>FEATURE Xᵢ</th>
                    <th>CATEGORY</th>
                    {dataset.classes.map((classLabel) => (
                      <th key={classLabel}>P(Xᵢ = category | {className(classLabel)})</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dataset.features.map((feature) => {
                    const featureRows = probabilityRows.filter(
                      (row) => row.feature.key === feature.key,
                    );
                    return featureRows.map((row, index) => (
                      <tr key={`${feature.key}-${row.category}`}>
                        {index === 0 && (
                          <th className="feature-group" scope="rowgroup" rowSpan={featureRows.length}>
                            {feature.label}
                          </th>
                        )}
                        <td>{displayValue(row.category)}</td>
                        {row.likelihoods.map((likelihood) => (
                          <td key={likelihood.classLabel}>
                            <strong>{likelihood.probability === null ? "N/A" : formatNumber(likelihood.probability)}</strong>
                            <span className="cell-sub">
                              ({likelihood.matchingCount} + {formatAlpha(alpha)}) / ({likelihood.classCount} + {formatAlpha(alpha)} × {feature.categories.length})
                            </span>
                          </td>
                        ))}
                      </tr>
                    ));
                  })}
                </tbody>
              </table>
            </div>
            <div className="likelihood-table-foot">
              Counts and class totals use only the {dataset.trainingRows.length} training examples. The table updates when you change the dataset or α.
            </div>
          </div>
        </section>}

        {dataset && <section className="calculation-section" id="calculation">
          <div className="section-heading calculation-heading">
            <div>
              <div className="eyebrow dark-eyebrow">OPEN THE BLACK BOX · STEP-BY-STEP PREDICTION</div>
              <h2>Every step of the <span>calculation.</span></h2>
            </div>
            <span className="step-pill"><span>03</span> FROM PRIOR TO LOG SCORE</span>
          </div>
          <div className="calculation-intro">
            <span className="calculation-icon">∑</span>
            <p><strong>Naive Bayes log score</strong><br /><span>For each candidate class, start with its prior and add the log likelihood of every selected feature.</span></p>
            <div className="calculation-formula">log score(c) = log P(c) + ∑ log P(xᵢ | c)</div>
          </div>
          {prediction && <div className="class-calculations">
            {prediction.results.map((result) => (
              <article className={`class-calculation ${result.classLabel === prediction.prediction ? "calculation-winner" : ""}`} key={result.classLabel}>
                <div className="class-calculation-heading">
                  <div><span className="calculation-step">CLASS SCORE</span><h3>{className(result.classLabel)}</h3></div>
                  {result.classLabel === prediction.prediction && <span className="winner-tag">PREDICTED</span>}
                </div>
                <div className="prior-strip">
                  <span className="prior-label">START WITH THE PRIOR</span>
                  <code>log P({result.classLabel}) = log({result.classCount}/{prediction.trainingCount}) = {formatNumber(result.logPrior)}</code>
                  <span className="prior-value">{formatNumber(result.prior, 3)}</span>
                </div>
                <div className="table-scroll">
                  <table className="detail-table">
                    <thead><tr><th>FEATURE · VALUE</th><th>COUNT IN CLASS</th><th>SMOOTHED LIKELIHOOD</th><th>P(VALUE | CLASS)</th><th>LOG CONTRIBUTION</th></tr></thead>
                    <tbody>{result.details.map((detail) => (
                      <tr key={detail.feature}>
                        <td><strong>{detail.featureLabel}</strong><span className="cell-sub">{displayValue(detail.value)}</span></td>
                        <td>{detail.matchingCount} <span className="muted">/ {detail.classCount}</span></td>
                        <td className="fraction-cell">({detail.matchingCount} + {detail.alpha}) / ({detail.classCount} + {detail.alpha} × {detail.categoryCount})</td>
                        <td>{formatNumber(detail.probability)}</td>
                        <td>{formatNumber(detail.logContribution)}</td>
                      </tr>
                    ))}</tbody>
                  </table>
                </div>
                <div className="final-score">
                  <span>ADD THE LOGS <small>prior + {result.details.length} feature contributions</small></span>
                  <code>{formatNumber(result.logPrior)}{result.details.map((detail) => ` + ${formatNumber(detail.logContribution)}`).join("")}</code>
                  <strong>{formatNumber(result.logScore)}</strong>
                </div>
              </article>
            ))}
          </div>}
        </section>}

        {dataset && <section className="conditional-section">
          <div className="conditional-copy">
            <div className="eyebrow dark-eyebrow">ZOOM IN · ONE CONDITIONAL PROBABILITY</div>
            <h2>What does <span>“given the class”</span> mean?</h2>
            <p>Look at just one feature and ask how often its value appears among training examples of each class.</p>
            <div className="conditional-picker">
              <label>FEATURE <select value={conditionalFeature} onChange={(event) => changeConditionalFeature(event.target.value)}>{dataset.features.map((feature) => <option key={feature.key} value={feature.key}>{feature.label}</option>)}</select></label>
              <label>VALUE <select value={conditionalValue} onChange={(event) => setConditionalValue(parseCategory(selectedConditionalFeature, event.target.value))}>{selectedConditionalFeature?.categories.map((value) => <option key={value} value={value}>{displayValue(value)}</option>)}</select></label>
            </div>
          </div>
          <div className="conditional-table-card">
            <div className="table-card-heading"><span>TRAINING COUNTS → CONDITIONAL PROBABILITY</span><span>α = {Number(alpha).toFixed(1)}</span></div>
            <div className="table-scroll"><table className="conditional-table">
              <thead><tr><th>CLASS</th><th>COUNT WITH VALUE</th><th>CLASS COUNT</th><th>SMOOTHED P(VALUE | CLASS)</th></tr></thead>
              <tbody>{conditionalStats.map((row) => <tr key={row.classLabel}><td><strong>{className(row.classLabel)}</strong></td><td>{row.matchingCount}</td><td>{row.classCount}</td><td><strong>{formatNumber(row.probability)}</strong><span className="cell-sub">({row.matchingCount} + {alpha}) / ({row.classCount} + {alpha} × {row.categoryCount})</span></td></tr>)}</tbody>
            </table></div>
            <div className="conditional-table-foot">The denominator adds α once for each possible value of this feature (K = {selectedConditionalFeature?.categories.length}).</div>
          </div>
        </section>}

        {dataset && <section className="data-section">
          <div className="section-heading data-heading">
            <div><div className="eyebrow dark-eyebrow">THE EXAMPLE DATA</div><h2>Learned from <span>training rows only.</span></h2></div>
            <span className="step-pill"><span>04</span> {dataset.trainingRows.length} TRAINING EXAMPLES</span>
          </div>
          <p className="data-intro">The model counts come from the training split in the original notebook. Test rows are kept separate.</p>
          <div className="data-table-wrap"><table className="data-table">
            <thead><tr>{dataset.features.map((feature) => <th key={feature.key}>{feature.label}</th>)}<th>{dataset.target.replaceAll("_", " ")}</th><th>PARTITION</th></tr></thead>
            <tbody>{dataset.trainingRows.slice(0, 6).map((row, index) => <tr key={`${row[dataset.target]}-${index}`}>{dataset.features.map((feature) => <td key={feature.key}>{displayValue(row[feature.key])}</td>)}<td><span className="class-chip">{row[dataset.target].replaceAll("_", " ")}</span></td><td><span className="train-dot" /> train</td></tr>)}</tbody>
          </table></div>
          <div className="data-foot">Showing 6 of {dataset.trainingRows.length} training rows · All {dataset.testRows.length} test examples are excluded from the calculations above.</div>
        </section>}

        <footer className="footer">
          <a className="brand footer-brand" href="#top"><span className="brand-mark">b.</span><span>bayes<span className="brand-light">lab</span></span></a>
          <span>A tiny lab for a big idea.</span>
          <a href="#top">Back to top ↑</a>
        </footer>
      </main>
    </div>
  );
}

export default App;
