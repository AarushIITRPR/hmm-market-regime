import { useEffect, useMemo, useState } from "react";
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

const views = [
  { id: "price", label: "Price" },
  { id: "regimes", label: "Regime Price" },
  { id: "states", label: "Hidden States" },
  { id: "returns", label: "Returns" },
  { id: "transition", label: "Transition Matrix" },
  { id: "distribution", label: "Regime Distribution" },
];

export default function Dashboard() {
  const [formValues, setFormValues] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [activeView, setActiveView] = useState("price");

  useEffect(() => {
    handleSubmit(initialForm);
  }, []);

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
  const ticker = result?.ticker ?? formValues.ticker.toUpperCase();
  const hasResult = Boolean(result);

  return (
    <div className="app-shell">
      <main className="terminal">
        <QuoteHeader formValues={formValues} status={status} summary={summary} ticker={ticker} />

        <section className="chart-workspace">
          <aside className="settings-rail">
            <nav className="settings-tabs">
              <strong>Inputs</strong>
              <span>Outputs</span>
              <span>Model</span>
            </nav>

            <section className="settings-section">
              <h2>Ticker Analysis</h2>
              <TickerForm values={formValues} isLoading={status === "loading"} onSubmit={handleSubmit} />
            </section>

            <section className="settings-section">
              <h2>Prediction Output</h2>
              <DataRow label="Status" value={status === "loading" ? "Running" : hasResult ? "Complete" : "Ready"} />
              <DataRow label="Latest Close" value={summary.latestClose} />
              <DataRow label="Latest Return" value={summary.latestReturn} />
              <DataRow label="Latest Regime" value={summary.latestRegime} />
              <DataRow label="Observations" value={summary.observations} />
              <DataRow label="Hidden States" value={summary.states} />
            </section>

            <section className="settings-section">
              <h2>Model Parameters</h2>
              <DataRow label="n_states" value="3" />
              <DataRow label="n_iter" value="100" />
              <DataRow label="tol" value="1e-6" />
              <DataRow label="random_state" value="42" />
            </section>

            {error && <div className="alert">{error}</div>}
          </aside>

          <section className="chart-stage">
            <div className="view-toolbar">
              <strong>{formatRange(formValues)}</strong>
              {views.map((view) => (
                <button
                  className={activeView === view.id ? "active" : ""}
                  key={view.id}
                  onClick={() => setActiveView(view.id)}
                  type="button"
                >
                  {view.label}
                </button>
              ))}
            </div>

            <div className="legend-strip">
              <LegendChip color="#6e91bd" label={`${ticker} close ${summary.latestClose}`} />
              <LegendChip color="#5c943d" label={`Regime ${summary.latestRegime}`} />
              <LegendChip color="#c46565" label={`${summary.states} hidden states`} />
            </div>

            <div className="main-chart">
              {status === "loading" && (
                <div className="loading-panel">
                  <div className="loader" />
                  <span>Loading prediction</span>
                </div>
              )}

              {hasResult && status !== "loading" && (
                <ActiveView
                  activeView={activeView}
                  key={`${activeView}-${result.ticker}-${result.dates.length}`}
                  result={result}
                />
              )}

              {!hasResult && status !== "loading" && (
                <div className="empty-state">
                  <h2>Chart ready</h2>
                  <p>Run a ticker analysis from the input panel.</p>
                </div>
              )}
            </div>
          </section>
        </section>
      </main>
    </div>
  );
}

function ActiveView({ activeView, result }) {
  if (activeView === "price") {
    return <StockPriceChart result={result} />;
  }

  if (activeView === "regimes") {
    return <RegimePriceChart result={result} />;
  }

  if (activeView === "states") {
    return <HiddenStateTimeline result={result} />;
  }

  if (activeView === "returns") {
    return <ReturnsChart result={result} />;
  }

  if (activeView === "transition") {
    return <TransitionHeatmap result={result} />;
  }

  if (activeView === "distribution") {
    return <RegimeDistributionChart result={result} />;
  }

  return <StockPriceChart result={result} />;
}

function QuoteHeader({ formValues, status, summary, ticker }) {
  return (
    <header className="quote-header">
      <div className="company-block">
        <div className="company-title">
          <h1>{ticker} Market Regime</h1>
          <span>NasdaqGS</span>
        </div>
        <div className="quote-line">
          <strong>{ticker}</strong>
          <b>{summary.latestCloseRaw}</b>
          <span>USD</span>
          <em>{summary.latestReturn}</em>
        </div>
        <small>{status === "loading" ? "Running HMM prediction" : "Latest model output"}</small>
      </div>

      <QuoteStat label="Start Date" value={formValues.startDate} />
      <QuoteStat label="End Date" value={formValues.endDate} />
      <QuoteStat label="Latest Regime" value={summary.latestRegime} />
      <QuoteStat label="Observations" value={summary.observations} />
      <QuoteStat label="States" value={summary.states} />
    </header>
  );
}

function QuoteStat({ label, value }) {
  return (
    <div className="quote-stat">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function DataRow({ label, value }) {
  return (
    <div className="data-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function LegendChip({ color, label }) {
  return (
    <div className="legend-chip" style={{ borderLeftColor: color }}>
      <strong>{label}</strong>
    </div>
  );
}

function buildSummary(result) {
  if (!result) {
    return {
      observations: "-",
      latestClose: "-",
      latestCloseRaw: "-",
      latestRegime: "-",
      latestReturn: "-",
      states: "-",
    };
  }

  const latestClose = result.close_prices.at(-1);
  const latestReturn = result.returns.at(-1);
  const uniqueStates = new Set(result.hidden_states);

  return {
    observations: result.dates.length.toLocaleString(),
    latestClose: latestClose == null ? "-" : `$${latestClose.toFixed(2)}`,
    latestCloseRaw: latestClose == null ? "-" : latestClose.toFixed(2),
    latestRegime: result.predicted_regime_labels.at(-1) ?? "-",
    latestReturn: latestReturn == null ? "-" : `${(latestReturn * 100).toFixed(2)}%`,
    states: uniqueStates.size.toString(),
  };
}

function formatRange(formValues) {
  return `${formValues.startDate} - ${formValues.endDate}`;
}
