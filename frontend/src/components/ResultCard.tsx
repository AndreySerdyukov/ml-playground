import type { ModelInfo, PredictResponse } from "../api";

// Карточка результата: крупное значение (регрессия) или класс + вероятности.
export function ResultCard({
  model,
  result,
}: {
  model: ModelInfo;
  result: PredictResponse | null;
}) {
  if (!result) {
    return (
      <div className="flex items-center justify-center rounded-apple bg-mist p-8 text-center text-sm text-slate">
        Fill in the inputs and hit Predict.
      </div>
    );
  }

  const isRegression = result.task === "regression";
  const value = isRegression
    ? formatNumber(Number(result.prediction))
    : String(result.prediction);

  const probs = result.probabilities
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <div className="rounded-apple bg-mist p-8">
      <p className="text-sm capitalize text-slate">{model.target}</p>
      <p className="mt-1 text-4xl font-semibold tracking-tight text-ink">
        {value}
        {isRegression && model.target_unit ? (
          <span className="ml-1 text-xl text-slate">{model.target_unit}</span>
        ) : null}
      </p>

      {probs.length > 0 && (
        <div className="mt-6 space-y-2.5">
          {probs.map(([cls, p]) => (
            <div key={cls}>
              <div className="flex justify-between text-xs text-slate">
                <span>{cls}</span>
                <span>{(p * 100).toFixed(1)}%</span>
              </div>
              <div className="mt-1 h-1.5 w-full rounded-full bg-white">
                <div className="h-1.5 rounded-full bg-ink" style={{ width: `${p * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      )}

      {model.is_stub && (
        <p className="mt-6 text-xs text-slate">Demo weights – illustrative output.</p>
      )}
    </div>
  );
}

function formatNumber(n: number): string {
  const digits = Math.abs(n) >= 100 ? 0 : 3;
  return n.toLocaleString("en-US", { maximumFractionDigits: digits });
}
