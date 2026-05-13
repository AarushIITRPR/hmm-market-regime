import { useMemo, useState } from "react";
import Header from "../components/Header.jsx";
import MetricCard from "../components/MetricCard.jsx";
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

  async function handleSubmit(event) {
    event.preventDefault();
    setStatus("loading");
    setError("");

    try {
      const data = await predictTickerRegimes(formValues);
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
      <Header />

      <main className="dashboard">
        <section className="control-band">
          <div>
            <p className="eyebrow">HMM Market Regime Detection</p>
            <h1>{result?.ticker ?? formValues.ticker.toUpperCase()}</h1>
          </div>
          <TickerForm
            values={formValues}
            isLoading={status === "loading"}
            onChange={setFormValues}
            onSubmit={handleSubmit}
          />
        </section>

        {error && <div className="alert">{error}</div>}

        {status === "loading" && (
          <div className="loading-panel">
            <div className="loader" />
            <span>Loading regime prediction</span>
          </div>
        )}

        {hasResult && (
          <>
            <section className="metric-grid">
              <MetricCard label="Observations" value={summary.observations} />
              <MetricCard label="Latest Close" value={summary.latestClose} />
              <MetricCard label="Latest Regime" value={summary.latestRegime} />
              <MetricCard label="States" value={summary.states} />
            </section>

            <section className="chart-grid">
              <StockPriceChart result={result} />
              <RegimePriceChart result={result} />
              <HiddenStateTimeline result={result} />
              <ReturnsChart result={result} />
              <TransitionHeatmap result={result} />
              <RegimeDistributionChart result={result} />
            </section>
          </>
        )}

        {!hasResult && status === "idle" && (
          <section className="empty-state">
            <h2>Ready</h2>
            <p>Run a ticker prediction to populate the dashboard.</p>
          </section>
        )}
      </main>
    </div>
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
