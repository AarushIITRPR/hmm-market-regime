export default function ChartPanel({ title, children }) {
  return (
    <article className="chart-panel">
      <header>
        <h2>{title}</h2>
      </header>
      {children}
    </article>
  );
}
