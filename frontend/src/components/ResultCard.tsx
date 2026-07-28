import { useState } from "react";
import type { FeatureSpec, ModelInfo, PredictResponse } from "../api";

// Result card: large value + range (regression) or classes, plus an echo of the entered
// features - so the right column isn't empty and you can see exactly what was scored. For binary
// classification we show an interactive threshold slider: the label and precision/recall update live.
export function ResultCard({
  model,
  result,
  inputs,
}: {
  model: ModelInfo;
  result: PredictResponse | null;
  inputs?: Record<string, string> | null;
}) {
  // Decision threshold (starts at the model's operating threshold). We clamp it to the slider range and round to
  // the step so the handle starts "on a notch". Reset to default happens on model change via the remount of
  // PredictPanel (key=model.name), so no useEffect is needed.
  const [threshold, setThreshold] = useState<number>(() =>
    clampThreshold(model.default_threshold ?? 0.5),
  );

  if (!result) {
    return (
      <div className="flex items-center justify-center rounded-apple bg-mist p-8 text-center text-sm text-slate">
        Fill in the inputs and hit Predict
      </div>
    );
  }

  const isRegression = result.task === "regression";
  const pred = Number(result.prediction);
  // Range around the estimate based on the model's typical relative error (median APE).
  const err = isRegression ? model.typical_error_pct ?? null : null;

  // Uplift model: a numeric score, but we render the two treatment scenarios + a verdict (label category='Uplift').
  const isUplift = model.category === "Uplift" && !!result.scenarios;
  const pTreat = isUplift ? result.scenarios!["with_treatment"] ?? null : null;
  const pControl = isUplift ? result.scenarios!["without_treatment"] ?? null : null;

  const probs = result.probabilities
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  // Binary classification with an interactive threshold (model returned positive_class + 2 classes).
  const pos = model.positive_class ?? null;
  const isBinaryThresh =
    !isRegression &&
    !!result.probabilities &&
    !!pos &&
    pos in result.probabilities &&
    Object.keys(result.probabilities).length === 2;

  const pPos = isBinaryThresh ? result.probabilities![pos!] : null;
  const negClass = isBinaryThresh
    ? Object.keys(result.probabilities!).find((c) => c !== pos)!
    : null;
  // We derive the label from the chosen threshold (not from the backend's prediction) - that's the point of the slider.
  const label =
    isBinaryThresh && pPos != null ? (pPos >= threshold ? pos! : negClass!) : String(result.prediction);
  // Uplift is shown in signed percentage points ("+3.7 pp"); otherwise a number (regression) or a class.
  const value = isUplift
    ? `${pred > 0 ? "+" : ""}${(pred * 100).toFixed(1)} pp`
    : isRegression
      ? formatNumber(pred)
      : label;

  // Nearest point on the threshold curve → precision/recall at the chosen threshold.
  const curve = model.threshold_curve ?? null;
  const opPoint =
    isBinaryThresh && curve && curve.length
      ? curve.reduce((best, p) =>
          Math.abs(p.threshold - threshold) < Math.abs(best.threshold - threshold) ? p : best,
        )
      : null;

  return (
    <div className="rounded-apple bg-mist p-8">
      <p className="text-sm capitalize text-slate">{model.target}</p>
      <p className="mt-1 text-4xl font-semibold tracking-tight text-ink">
        {value}
        {isRegression && !isUplift && model.target_unit ? (
          <span className="ml-1 text-xl text-slate">{model.target_unit}</span>
        ) : null}
      </p>

      {isUplift && pTreat != null && pControl != null && (
        <div className="mt-5 space-y-2.5">
          {[
            { label: "Buy if SMS sent", p: pTreat, strong: true },
            { label: "Buy if no SMS", p: pControl, strong: false },
          ].map((s) => (
            <div key={s.label}>
              <div className="flex justify-between text-xs">
                <span className={s.strong ? "font-medium text-ink" : "text-slate"}>{s.label}</span>
                <span className="tabular-nums text-slate">{(s.p * 100).toFixed(1)}%</span>
              </div>
              <div className="mt-1 h-1.5 w-full rounded-full bg-canvas">
                <div
                  className={`h-1.5 rounded-full ${s.strong ? "bg-ink" : "bg-ink/35"}`}
                  style={{ width: `${s.p * 100}%` }}
                />
              </div>
            </div>
          ))}
          <div className="pt-1">
            {pred > 0 ? (
              <span className="inline-flex items-center rounded-full bg-ink px-3 py-1 text-xs font-medium text-canvas">
                Target this customer
              </span>
            ) : (
              <span className="text-xs text-slate">Not worth targeting (no lift)</span>
            )}
          </div>
        </div>
      )}

      {err != null && !isUplift && (
        <div className="mt-5">
          <p className="text-xs text-slate">±{Math.round(err * 100)}% typical range</p>
          <div className="relative mt-2 h-2 rounded-full bg-gradient-to-r from-hair via-ink/25 to-hair">
            <span className="absolute left-1/2 top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-mist bg-ink" />
          </div>
          <div className="mt-1.5 flex justify-between text-xs text-slate">
            <span>{formatNumber(pred * (1 - err))}</span>
            <span>{formatNumber(pred * (1 + err))}</span>
          </div>
        </div>
      )}

      {isBinaryThresh && (
        <div className="mt-5">
          <div className="flex items-baseline justify-between text-xs text-slate">
            <span>Decision threshold</span>
            <span className="tabular-nums text-ink">
              P({pos}) ≥ {threshold.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min={0.05}
            max={0.95}
            step={0.01}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            aria-label="Decision threshold"
            className="mt-2 w-full accent-ink"
          />
          {opPoint && (
            <div className="mt-1 flex justify-between text-xs text-slate">
              <span>
                precision <span className="tabular-nums text-ink">{pct(opPoint.precision)}</span>
              </span>
              <span>
                recall <span className="tabular-nums text-ink">{pct(opPoint.recall)}</span>
              </span>
            </div>
          )}
        </div>
      )}

      {probs.length > 0 && (
        <div className="mt-6 space-y-2.5">
          {probs.map(([cls, p]) => {
            // The class that "wins" at the current threshold (matches the label) - we highlight it,
            // so that when the marker crosses over the switch is visible right on the bars.
            const winning = isBinaryThresh && cls === label;
            return (
              <div key={cls}>
                <div className="flex justify-between text-xs">
                  <span className={winning ? "font-medium text-ink" : "text-slate"}>{cls}</span>
                  <span className="text-slate">{(p * 100).toFixed(1)}%</span>
                </div>
                <div className="relative mt-1 h-1.5 w-full rounded-full bg-canvas">
                  <div
                    className={`h-1.5 rounded-full ${isBinaryThresh && !winning ? "bg-ink/35" : "bg-ink"}`}
                    style={{ width: `${p * 100}%` }}
                  />
                  {/* Threshold marker on the positive-class bar - shows where the decision boundary lies. */}
                  {isBinaryThresh && cls === pos && (
                    <span
                      className="absolute top-1/2 h-3 w-0.5 -translate-y-1/2 bg-ink"
                      style={{ left: `${threshold * 100}%` }}
                    />
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {inputs && (
        <div className="mt-6 border-t border-hair pt-5">
          <p className="mb-3 text-[11px] font-medium uppercase tracking-wide text-slate">Inputs</p>
          <dl className="space-y-2">
            {model.features.map((f) => (
              <div key={f.name} className="flex items-baseline justify-between gap-3 text-sm">
                <dt className="shrink-0 text-slate">{f.label ?? f.name}</dt>
                <dd className="truncate text-right text-ink">{formatInput(f, inputs[f.name])}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}

      {model.is_stub && (
        <p className="mt-6 text-xs text-slate">Demo weights – illustrative output</p>
      )}
    </div>
  );
}

function pct(x: number): string {
  return `${Math.round(x * 100)}%`;
}

// Clamp the threshold to the slider range [0.05, 0.95] and to a step of 0.01 (handle starts exactly on a notch).
function clampThreshold(t: number): number {
  return Math.min(0.95, Math.max(0.05, Math.round(t * 100) / 100));
}

// Feature value for the echo: numbers are formatted with separators and a unit.
function formatInput(f: FeatureSpec, raw: string | undefined): string {
  if (raw == null || raw === "") return "–";
  if (f.type === "number") {
    const n = Number(raw);
    // Thousands separators only for large numbers - so a year (2012) doesn't become "2,012".
    const text = !Number.isFinite(n) ? raw : Math.abs(n) >= 10000 ? n.toLocaleString("en-US") : String(n);
    return f.unit ? `${text} ${f.unit}` : text;
  }
  return raw;
}

function formatNumber(n: number): string {
  const digits = Math.abs(n) >= 100 ? 0 : 3;
  return n.toLocaleString("en-US", { maximumFractionDigits: digits });
}
