import ChartPanel from "./ChartPanel.jsx";
import Plot from "./Plot.jsx";
import { baseLayout, colorForState, plotConfig } from "./chartConfig.js";

export default function RegimePriceChart({ result }) {
  const colors = result.hidden_states.map(colorForState);

  return (
    <ChartPanel title="Regime-Colored Price">
      <Plot
        data={[
          {
            x: result.dates,
            y: result.close_prices,
            type: "scatter",
            mode: "lines+markers",
            name: "Close",
            line: { color: "rgba(15, 23, 42, 0.28)", width: 2 },
            marker: {
              color: colors,
              size: 7,
              line: { color: "#ffffff", width: 0.8 },
            },
            text: result.predicted_regime_labels,
            hovertemplate: "%{x}<br>Close: %{y:.2f}<br>%{text}<extra></extra>",
          },
        ]}
        layout={{
          ...baseLayout,
          yaxis: { ...baseLayout.yaxis, title: "Close" },
          height: 340,
        }}
        config={plotConfig}
        useResizeHandler
        className="plot"
      />
    </ChartPanel>
  );
}
