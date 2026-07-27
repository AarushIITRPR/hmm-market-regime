import ChartPanel from "./ChartPanel.jsx";
import Plot from "./Plot.jsx";
import { baseLayout, colorForState, plotConfig } from "./chartConfig.js";

export default function RegimeDistributionChart({ result }) {
  const states = [...new Set(result.hidden_states)].sort((a, b) => a - b);
  const finiteReturns = result.returns
    .map((value) => Number(value) * 100)
    .filter(Number.isFinite);
  const minReturn = finiteReturns.length ? Math.min(...finiteReturns) : -1;
  const maxReturn = finiteReturns.length ? Math.max(...finiteReturns) : 1;
  const padding = Math.max((maxReturn - minReturn) * 0.08, 0.05);

  const traces = states.map((state) => {
    const values = result.returns.reduce((stateReturns, value, index) => {
      const returnPct = Number(value) * 100;
      if (result.hidden_states[index] === state && Number.isFinite(returnPct)) {
        stateReturns.push(returnPct);
      }
      return stateReturns;
    }, []);

    return {
      autobinx: false,
      xbins: {
        start: minReturn - padding,
        end: maxReturn + padding,
        size: Math.max((maxReturn - minReturn) / 28, 0.05),
      },
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
          hovermode: "closest",
          xaxis: {
            ...baseLayout.xaxis,
            title: "Return %",
            type: "linear",
            range: [minReturn - padding, maxReturn + padding],
            tickformat: ".2f",
          },
          yaxis: {
            ...baseLayout.yaxis,
            title: "Probability",
            rangemode: "tozero",
          },
          height: 420,
        }}
        config={plotConfig}
        useResizeHandler
        className="plot"
      />
    </ChartPanel>
  );
}
