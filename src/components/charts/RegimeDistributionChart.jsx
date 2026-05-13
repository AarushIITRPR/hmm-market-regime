import ChartPanel from "./ChartPanel.jsx";
import Plot from "./Plot.jsx";
import { baseLayout, colorForState, plotConfig } from "./chartConfig.js";

export default function RegimeDistributionChart({ result }) {
  const states = [...new Set(result.hidden_states)].sort((a, b) => a - b);

  const traces = states.map((state) => {
    const values = result.returns
      .filter((_, index) => result.hidden_states[index] === state)
      .map((value) => value * 100);

    return {
      x: values,
      type: "histogram",
      histnorm: "probability",
      name: `State ${state}`,
      marker: { color: colorForState(state), opacity: 0.72 },
      hovertemplate: `State ${state}<br>Return: %{x:.3f}%<br>Probability: %{y:.3f}<extra></extra>`,
    };
  });

  return (
    <ChartPanel title="Regime Distribution">
      <Plot
        data={traces}
        layout={{
          ...baseLayout,
          barmode: "overlay",
          xaxis: { ...baseLayout.xaxis, title: "Return %" },
          yaxis: { ...baseLayout.yaxis, title: "Probability" },
          height: 320,
        }}
        config={plotConfig}
        useResizeHandler
        className="plot"
      />
    </ChartPanel>
  );
}
