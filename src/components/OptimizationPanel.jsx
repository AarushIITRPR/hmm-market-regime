import { useMemo, useState } from "react";
import { runOptimization } from "../services/optimizationApi.js";

const defaultAssets = [
  {
    id: "asset-spy",
    ticker: "SPY",
    price: 100,
    expected_profit: 2.5,
    risk_score: 4,
    regime_label: "Low Volatility Bull",
  },
  {
    id: "asset-qqq",
    ticker: "QQQ",
    price: 120,
    expected_profit: 3.2,
    risk_score: 6,
    regime_label: "Neutral Volatility Bull",
  },
  {
    id: "asset-dia",
    ticker: "DIA",
    price: 95,
    expected_profit: 1.8,
    risk_score: 3,
    regime_label: "High Volatility Bear",
  },
];

export default function OptimizationPanel({ regimeResult }) {
  const [assets, setAssets] = useState(defaultAssets);
  const [scenario, setScenario] = useState({
    name: "Base Case",
    budget: 1000,
    max_risk: 45,
    max_units_per_asset: 10,
  });
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const latestRegimeAsset = useMemo(() => buildAssetFromRegime(regimeResult), [regimeResult]);

  function updateScenario(field, value) {
    setScenario((current) => ({
      ...current,
      [field]: field === "name" ? value : Number(value),
    }));
  }

  function updateAsset(index, field, value) {
    setAssets((current) =>
      current.map((asset, assetIndex) =>
        assetIndex === index
          ? { ...asset, [field]: field === "ticker" || field === "regime_label" ? value : Number(value) }
          : asset,
      ),
    );
  }

  function addAsset() {
    setAssets((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        ticker: "NEW",
        price: 100,
        expected_profit: 1,
        risk_score: 5,
        regime_label: "Unknown",
      },
    ]);
  }

  function removeAsset(index) {
    setAssets((current) => current.filter((_, assetIndex) => assetIndex !== index));
  }

  function useLatestRegimeAsset() {
    if (!latestRegimeAsset) {
      return;
    }
      setAssets((current) => [
        latestRegimeAsset,
        ...current.filter((asset) => asset.ticker !== latestRegimeAsset.ticker),
      ]);
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setStatus("loading");
    setError("");
    setResult(null);

    try {
      const data = await runOptimization({
        assets,
        scenarios: [scenario],
      });
      setResult(data.results[0]);
      setStatus("success");
    } catch (requestError) {
      setError(requestError.message);
      setStatus("error");
    }
  }

  return (
    <section className="optimization-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">OR-Tools Optimization</p>
          <h2>Simple Stock Allocation</h2>
        </div>
        <button type="button" onClick={useLatestRegimeAsset} disabled={!latestRegimeAsset}>
          Use Latest Regime
        </button>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="optimizer-controls">
          <label>
            <span>Budget</span>
            <input
              type="number"
              min="1"
              step="1"
              value={scenario.budget}
              onChange={(event) => updateScenario("budget", event.target.value)}
            />
          </label>
          <label>
            <span>Max Risk</span>
            <input
              type="number"
              min="0"
              step="1"
              value={scenario.max_risk}
              onChange={(event) => updateScenario("max_risk", event.target.value)}
            />
          </label>
          <label>
            <span>Max Units</span>
            <input
              type="number"
              min="0"
              step="1"
              value={scenario.max_units_per_asset}
              onChange={(event) => updateScenario("max_units_per_asset", event.target.value)}
            />
          </label>
          <button type="submit" disabled={status === "loading"}>
            {status === "loading" ? "Solving" : "Optimize"}
          </button>
        </div>

        <div className="asset-table">
          <div className="asset-row asset-header">
            <span>Ticker</span>
            <span>Price</span>
            <span>Profit</span>
            <span>Risk</span>
            <span>Regime</span>
            <span />
          </div>
          {assets.map((asset, index) => (
            <div className="asset-row" key={asset.id}>
              <input value={asset.ticker} onChange={(event) => updateAsset(index, "ticker", event.target.value)} />
              <input type="number" min="0.01" step="0.01" value={asset.price} onChange={(event) => updateAsset(index, "price", event.target.value)} />
              <input type="number" min="0" step="0.01" value={asset.expected_profit} onChange={(event) => updateAsset(index, "expected_profit", event.target.value)} />
              <input type="number" min="0" step="0.1" value={asset.risk_score} onChange={(event) => updateAsset(index, "risk_score", event.target.value)} />
              <input value={asset.regime_label} onChange={(event) => updateAsset(index, "regime_label", event.target.value)} />
              <button type="button" onClick={() => removeAsset(index)} disabled={assets.length === 1}>
                Remove
              </button>
            </div>
          ))}
        </div>

        <button className="secondary-action" type="button" onClick={addAsset}>
          Add Asset
        </button>
      </form>

      {error && <div className="alert optimizer-alert">{error}</div>}
      {result && <OptimizationResult result={result} />}
    </section>
  );
}

function OptimizationResult({ result }) {
  return (
    <div className="optimizer-result">
      <div className="optimizer-summary">
        <strong>Status: {result.status}</strong>
        <span>Objective: {result.objective_value.toFixed(2)}</span>
        <span>Cost: {result.total_cost.toFixed(2)}</span>
        <span>Risk: {result.total_risk.toFixed(2)}</span>
      </div>
      <div className="allocation-list">
        {result.allocations.map((allocation) => (
          <div className="allocation-row" key={allocation.ticker}>
            <strong>{allocation.ticker}</strong>
            <span>{allocation.units} units</span>
            <span>Profit {allocation.expected_profit.toFixed(2)}</span>
            <span>{allocation.regime_label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function buildAssetFromRegime(result) {
  if (!result || result.close_prices.length === 0) {
    return null;
  }

  const latestClose = result.close_prices.at(-1);
  const latestReturn = result.returns.at(-1) ?? 0;
  const latestRegime = result.predicted_regime_labels.at(-1) ?? "Unknown";
  const riskScore = latestRegime.includes("High") ? 8 : latestRegime.includes("Low") ? 3 : 5;

  return {
    id: `asset-${result.ticker.toLowerCase()}`,
    ticker: result.ticker,
    price: Number(latestClose.toFixed(2)),
    expected_profit: Number(Math.max(latestClose * latestReturn, 0.5).toFixed(2)),
    risk_score: riskScore,
    regime_label: latestRegime,
  };
}
