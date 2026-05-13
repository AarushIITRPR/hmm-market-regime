import ChartPanel from "./ChartPanel.jsx";
import Plot from "./Plot.jsx";
import { baseLayout, colorForState, plotConfig } from "./chartConfig.js";

export default function HiddenStateTimeline({ result }) {
  return (
    <ChartPanel title="Hidden State Timeline">
      <Plot
        data={[
          {
            x: result.dates,
            y: result.hidden_states,
            type: "scatter",
            mode: "lines+markers",
            name: "Hidden State",
            line: { color: "#334155", width: 1.6, shape: "hv" },
            marker: {
              color: result.hidden_states.map(colorForState),
              size: 7,
            },
            text: result.predicted_regime_labels,
            hovertemplate: "%{x}<br>State: %{y}<br>%{text}<extra></extra>",
          },
        ]}
        layout={{
          ...baseLayout,
          yaxis: { ...baseLayout.yaxis, title: "State", dtick: 1 },
          height: 320,
        }}
        config={plotConfig}
        useResizeHandler
        className="plot"
      />
    </ChartPanel>
  );
}
