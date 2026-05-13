import { useMemo, useState } from "react";
import MarketLattice3D from "../components/MarketLattice3D.jsx";
import OptimizationPanel from "../components/OptimizationPanel.jsx";
import TickerForm from "../components/TickerForm.jsx";
import HiddenStateTimeline from "../components/charts/HiddenStateTimeline.jsx";
import RegimeDistributionChart from "../components/charts/RegimeDistributionChart.jsx";
import RegimePriceChart from "../components/charts/RegimePriceChart.jsx";
import ReturnsChart from "../components/charts/ReturnsChart.jsx";
import StockPriceChart from "../components/charts/StockPriceChart.jsx";
import TransitionHeatmap from "../components/charts/TransitionHeatmap.jsx";
import { predictTickerRegimes } from "../services/regimeApi.js";

const initialForm = {
  ticker: "SPY",
  startDate: "2023-01-01",
  endDate: "2024-01-01",
};

export default function Dashboard() {
  const [formValues, setFormValues] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  async function handleSubmit(submittedValues) {
    setFormValues(submittedValues);
    setStatus("loading");
    setError("");

    try {
      const data = await predictTickerRegimes(submittedValues);
      setResult(data);
      setStatus("success");
    } catch (requestError) {
      setResult(null);
      setError(requestError.message);
      setStatus("error");
    }
  }

  const summary = useMemo(() => buildSummary(result), [result]);
  const hasResult = Boolean(result);

  return (
    <div className="app-shell">
      <MarketLattice3D />

      <main className="dashboard">
        <section className="workspace-grid">
          <aside className="input-panel">
            <div className="panel-heading">
              <p className="eyebrow">HMM Market Regime Detection</p>
              <h1>{result?.ticker ?? formValues.ticker.toUpperCase()}</h1>
              <span>Run ticker analysis</span>
            </div>

            <TickerForm
              values={formValues}
              isLoading={status === "loading"}
              onSubmit={handleSubmit}
            />

            <div className="run-defaults">
              <span>Model defaults</span>
              <dl>
                <div>
                  <dt>States</dt>
                  <dd>3</dd>
                </div>
                <div>
                  <dt>Iterations</dt>
                  <dd>100</dd>
                </div>
                <div>
                  <dt>Tolerance</dt>
                  <dd>1e-6</dd>
                </div>
                <div>
                  <dt>Seed</dt>
                  <dd>42</dd>
                </div>
              </dl>
            </div>

            {error && <div className="alert">{error}</div>}
          </aside>

          <section className="results-panel">
            <SummaryBoard
              formValues={formValues}
              hasResult={hasResult}
              status={status}
              summary={summary}
              ticker={result?.ticker ?? formValues.ticker.toUpperCase()}
            />

            {status === "loading" && (
              <div className="loading-panel">
                <div className="loader" />
                <span>Loading regime prediction</span>
              </div>
            )}

            {hasResult && (
              <section className="chart-grid">
                <StockPriceChart result={result} />
                <RegimePriceChart result={result} />
                <HiddenStateTimeline result={result} />
                <ReturnsChart result={result} />
                <TransitionHeatmap result={result} />
                <RegimeDistributionChart result={result} />
              </section>
            )}

            {!hasResult && status === "idle" && (
              <section className="empty-state">
                <h2>Ready</h2>
                <p>Submit a ticker from the run ticket to populate the market summary and charts.</p>
              </section>
            )}
          </section>
        </section>

        <OptimizationPanel regimeResult={result} />
      </main>
    </div>
  );
}

function SummaryBoard({ formValues, hasResult, status, summary, ticker }) {
  const rows = [
    ["Latest close", summary.latestClose],
    ["Latest regime", summary.latestRegime],
    ["Observations", summary.observations],
    ["States", summary.states],
  ];

  return (
    <section className="summary-board">
      <div className="summary-symbol">
        <span>Instrument</span>
        <strong>{ticker}</strong>
        <small>
          {formValues.startDate} to {formValues.endDate}
        </small>
      </div>

      <div className="summary-table">
        <div className="summary-status">
          <span>Status</span>
          <strong>{status === "loading" ? "Running" : hasResult ? "Complete" : "Awaiting Run"}</strong>
        </div>
        {rows.map(([label, value]) => (
          <div className="summary-row" key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function buildSummary(result) {
  if (!result) {
    return {
      observations: "-",
      latestClose: "-",
      latestRegime: "-",
      states: "-",
    };
  }

  const latestClose = result.close_prices.at(-1);
  const uniqueStates = new Set(result.hidden_states);

  return {
    observations: result.dates.length.toLocaleString(),
    latestClose: latestClose == null ? "-" : `$${latestClose.toFixed(2)}`,
    latestRegime: result.predicted_regime_labels.at(-1) ?? "-",
    states: uniqueStates.size.toString(),
  };
}
