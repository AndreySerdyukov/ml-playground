import { useState } from "react";
import type { FeatureSpec, ModelInfo, PredictResponse } from "../api";

// Карточка результата: крупное значение + диапазон (регрессия) или классы, плюс echo введённых
// признаков — так правая колонка не пустует и видно, что именно оценивали. Для бинарной
// классификации показываем интерактивный слайдер порога: лейбл и precision/recall меняются вживую.
export function ResultCard({
  model,
  result,
  inputs,
}: {
  model: ModelInfo;
  result: PredictResponse | null;
  inputs?: Record<string, string> | null;
}) {
  // Порог решения (стартует с рабочего порога модели). Клампим в диапазон слайдера и округляем до
  // шага, чтобы ручка стартовала «на насечке». Сброс на дефолт — при смене модели через remount
  // PredictPanel (key=model.name), поэтому useEffect не нужен.
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
  // Диапазон вокруг оценки по типичной относительной ошибке модели (median APE).
  const err = isRegression ? model.typical_error_pct ?? null : null;

  const probs = result.probabilities
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  // Бинарная классификация с интерактивным порогом (модель отдала positive_class + 2 класса).
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
  // Лейбл решаем от выбранного порога (а не от предсказания бэкенда) — в этом суть слайдера.
  const label =
    isBinaryThresh && pPos != null ? (pPos >= threshold ? pos! : negClass!) : String(result.prediction);
  const value = isRegression ? formatNumber(pred) : label;

  // Ближайшая точка кривой порога → precision/recall при выбранном пороге.
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
        {isRegression && model.target_unit ? (
          <span className="ml-1 text-xl text-slate">{model.target_unit}</span>
        ) : null}
      </p>

      {err != null && (
        <div className="mt-5">
          <p className="text-xs text-slate">±{Math.round(err * 100)}% typical range</p>
          <div className="relative mt-2 h-2 rounded-full bg-gradient-to-r from-hair via-ink/25 to-hair">
            <span className="absolute left-1/2 top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-ink" />
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
          {probs.map(([cls, p]) => (
            <div key={cls}>
              <div className="flex justify-between text-xs text-slate">
                <span>{cls}</span>
                <span>{(p * 100).toFixed(1)}%</span>
              </div>
              <div className="relative mt-1 h-1.5 w-full rounded-full bg-canvas">
                <div className="h-1.5 rounded-full bg-ink" style={{ width: `${p * 100}%` }} />
                {/* Маркер порога на баре positive-класса — видно, где проходит граница решения. */}
                {isBinaryThresh && cls === pos && (
                  <span
                    className="absolute top-1/2 h-3 w-0.5 -translate-y-1/2 bg-slate"
                    style={{ left: `${threshold * 100}%` }}
                  />
                )}
              </div>
            </div>
          ))}
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

// Порог в диапазон слайдера [0.05, 0.95] и на шаг 0.01 (ручка стартует ровно на насечке).
function clampThreshold(t: number): number {
  return Math.min(0.95, Math.max(0.05, Math.round(t * 100) / 100));
}

// Значение признака для echo: числа форматируем с разделителями и единицей.
function formatInput(f: FeatureSpec, raw: string | undefined): string {
  if (raw == null || raw === "") return "–";
  if (f.type === "number") {
    const n = Number(raw);
    // Разделители тысяч только для крупных чисел – чтобы год (2012) не стал «2,012».
    const text = !Number.isFinite(n) ? raw : Math.abs(n) >= 10000 ? n.toLocaleString("en-US") : String(n);
    return f.unit ? `${text} ${f.unit}` : text;
  }
  return raw;
}

function formatNumber(n: number): string {
  const digits = Math.abs(n) >= 100 ? 0 : 3;
  return n.toLocaleString("en-US", { maximumFractionDigits: digits });
}
