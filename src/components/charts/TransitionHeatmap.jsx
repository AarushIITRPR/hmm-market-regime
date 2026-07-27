import ChartPanel from "./ChartPanel.jsx";
import Plot from "./Plot.jsx";
import { baseLayout, plotConfig } from "./chartConfig.js";

export default function TransitionHeatmap({ result }) {
  const labels = result.transition_probabilities.map((_, index) => `State ${index}`);
  const matrixSize = Math.max(labels.length * 74, 300);

  return (
    <ChartPanel title="Transition Probability Matrix">
      <Plot
        key={`transition-heatmap-${labels.length}`}
        data={[
          {
            z: result.transition_probabilities,
            x: labels,
            y: labels,
            type: "heatmap",
            colorscale: [
              [0, "#f8fbff"],
              [0.25, "#dbeafe"],
              [0.5, "#93c5fd"],
              [0.75, "#3b82f6"],
              [1, "#1e3a8a"],
            ],
            zmin: 0,
            zmax: 1,
            xgap: 2,
            ygap: 2,
            text: result.transition_probabilities.map((row) =>
              row.map((value) => value.toFixed(3)),
            ),
            texttemplate: "%{text}",
            textfont: {
              color: "#111827",
              size: 14,
            },
            hovertemplate: "From %{y}<br>To %{x}<br>Probability: %{z:.4f}<extra></extra>",
            colorbar: {
              title: "Probability",
              thickness: 14,
              len: 0.68,
            },
          },
        ]}
        layout={{
          ...baseLayout,
          height: Math.max(matrixSize + 116, 416),
          hovermode: "closest",
          margin: { l: 82, r: 82, t: 18, b: 64 },
          xaxis: {
            title: "Next State",
            type: "category",
            categoryorder: "array",
            categoryarray: labels,
            gridcolor: "rgba(209, 213, 219, 0.95)",
            linecolor: "#d1d5db",
            tickfont: { color: "#374151", size: 13 },
            ticks: "",
          },
          yaxis: {
            title: "Current State",
            type: "category",
            categoryorder: "array",
            categoryarray: labels,
            autorange: "reversed",
            gridcolor: "rgba(209, 213, 219, 0.95)",
            linecolor: "#d1d5db",
            tickfont: { color: "#374151", size: 13 },
            ticks: "",
            scaleanchor: "x",
          },
        }}
        config={plotConfig}
        useResizeHandler
        className="plot"
      />
    </ChartPanel>
  );
}
