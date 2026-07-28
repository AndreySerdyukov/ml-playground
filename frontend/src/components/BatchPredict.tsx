import { useRef, useState } from "react";
import { predictBatch, type BatchPredictResponse, type ModelInfo } from "../api";

// Batch predict: upload a CSV/Excel file → table of per-row predictions + download of the result.
const PREVIEW_ROWS = 100;

export function BatchPredict({ model }: { model: ModelInfo }) {
  const [file, setFile] = useState<File | null>(null);
  const [res, setRes] = useState<BatchPredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  async function run(): Promise<void> {
    if (!file) return;
    setLoading(true);
    setError(null);
    setRes(null);
    try {
      setRes(await predictBatch(model.name, file));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  function downloadTemplate(): void {
    const headers = model.features.map((f) => f.name);
    const row = model.features.map((f) => (f.example != null ? String(f.example) : ""));
    download(`${model.name}_template.csv`, toCsv(headers, [row]));
  }

  function downloadResults(): void {
    if (!res) return;
    const headers = [...res.columns, "prediction"];
    const rows = res.rows.map((r) => [...res.columns.map((c) => toCell(r[c])), toCell(r.prediction)]);
    download(`${model.name}_predictions.csv`, toCsv(headers, rows));
  }

  return (
    <div className="space-y-6">
      <div className="rounded-apple border border-hair p-6 sm:p-8">
        <p className="text-[15px] text-ink">Upload a CSV or Excel file</p>
        <p className="mt-1 text-sm text-slate">
          One row per item, columns matching the model’s inputs. Extra columns are ignored.
        </p>

        <div className="mt-5 flex flex-wrap items-center gap-3">
          <input
            ref={fileRef}
            type="file"
            accept=".csv,.xlsx,.xls"
            className="hidden"
            onChange={(e) => {
              setFile(e.target.files?.[0] ?? null);
              setRes(null);
              setError(null);
            }}
          />
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            className="rounded-full border border-hair px-5 py-2.5 text-[15px] text-ink transition hover:bg-mist"
          >
            Choose file
          </button>
          <span className="truncate text-sm text-slate">{file ? file.name : "No file selected"}</span>
          <button type="button" onClick={run} disabled={!file || loading} className="btn-primary">
            {loading ? "Running…" : "Run batch"}
          </button>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
          <button type="button" onClick={downloadTemplate} className="text-link">
            Download template
          </button>
          <span className="text-slate">Columns: {model.features.map((f) => f.name).join(", ")}</span>
        </div>

        {error && <p className="mt-4 text-sm text-red-600">{error}</p>}
      </div>

      {res && (
        <div className="rounded-apple border border-hair p-6 sm:p-8">
          <div className="mb-4 flex items-center justify-between">
            <p className="text-[15px] text-ink">
              {res.count} {res.count === 1 ? "row" : "rows"} predicted
            </p>
            <button type="button" onClick={downloadResults} className="text-sm text-link">
              Download CSV
            </button>
          </div>
          <div className="overflow-x-auto rounded-xl border border-hair">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="bg-mist text-left text-slate">
                  {res.columns.map((c) => (
                    <th key={c} className="whitespace-nowrap px-3 py-2 font-medium">
                      {c}
                    </th>
                  ))}
                  <th className="whitespace-nowrap px-3 py-2 text-right font-semibold text-ink">
                    {res.target}
                    {res.target_unit ? `, ${res.target_unit}` : ""}
                  </th>
                </tr>
              </thead>
              <tbody>
                {res.rows.slice(0, PREVIEW_ROWS).map((r, i) => (
                  <tr key={i} className="border-t border-hair">
                    {res.columns.map((c) => (
                      <td key={c} className="whitespace-nowrap px-3 py-1.5 text-ink">
                        {cell(r[c])}
                      </td>
                    ))}
                    <td className="whitespace-nowrap px-3 py-1.5 text-right font-medium text-ink">
                      {cell(r.prediction)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {res.rows.length > PREVIEW_ROWS && (
            <p className="mt-3 text-xs text-slate">
              Showing first {PREVIEW_ROWS} of {res.count} – download the CSV for all rows
            </p>
          )}
        </div>
      )}
    </div>
  );
}

type CsvCell = string | number | boolean | null;

function toCell(v: unknown): CsvCell {
  if (v == null) return null;
  if (typeof v === "number" || typeof v === "boolean") return v;
  return String(v);
}

function cell(v: unknown): string {
  if (v == null) return "–";
  // Thousands separators only for large numbers - so a year (2012) doesn't become "2,012".
  if (typeof v === "number") return Math.abs(v) >= 10000 ? v.toLocaleString("en-US") : String(v);
  return String(v);
}

function toCsv(headers: string[], rows: CsvCell[][]): string {
  const esc = (v: CsvCell): string => {
    const s = v == null ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [headers.map(esc).join(","), ...rows.map((r) => r.map(esc).join(","))].join("\n");
}

function download(filename: string, content: string): void {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
