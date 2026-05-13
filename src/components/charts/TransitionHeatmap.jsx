import ChartPanel from "./ChartPanel.jsx";
import Plot from "./Plot.jsx";
import { baseLayout, plotConfig } from "./chartConfig.js";

export default function TransitionHeatmap({ result }) {
  const labels = result.transition_probabilities.map((_, index) => `State ${index}`);

  return (
    <ChartPanel title="Transition Probability Matrix">
      <Plot
        data={[
          {
            z: result.transition_probabilities,
            x: labels,
            y: labels,
            type: "heatmap",
            colorscale: "Blues",
            zmin: 0,
            zmax: 1,
            text: result.transition_probabilities.map((row) =>
              row.map((value) => value.toFixed(3)),
            ),
            texttemplate: "%{text}",
            hovertemplate: "From %{y}<br>To %{x}<br>Probability: %{z:.4f}<extra></extra>",
          },
        ]}
        layout={{
          ...baseLayout,
          xaxis: { ...baseLayout.xaxis, title: "Next State" },
          yaxis: { ...baseLayout.yaxis, title: "Current State" },
          height: 320,
        }}
        config={plotConfig}
        useResizeHandler
        className="plot"
      />
    </ChartPanel>
  );
}
