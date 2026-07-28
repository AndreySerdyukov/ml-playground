import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";

// Кастомный дропдаун (замена нативным select/datalist): единый стиль, поиск для длинных
// списков (model=972, company=91), клавиатура и закрытие по клику вне.
const MAX_VISIBLE = 100;

export function Combobox({
  value,
  onChange,
  options,
  placeholder = "Select…",
  id,
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
  placeholder?: string;
  id?: string;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const searchable = options.length > 8;

  const { visible, total } = useMemo(() => {
    const q = query.trim().toLowerCase();
    const matched = q ? options.filter((o) => o.toLowerCase().includes(q)) : options;
    return { visible: matched.slice(0, MAX_VISIBLE), total: matched.length };
  }, [options, query]);

  // Закрытие по клику вне компонента.
  useEffect(() => {
    if (!open) return;
    function onDoc(e: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  // При открытии фокус на поиск и сброс подсветки.
  useEffect(() => {
    if (!open) return;
    setActive(0);
    if (searchable) searchRef.current?.focus();
  }, [open, searchable]);

  function choose(v: string) {
    onChange(v);
    setOpen(false);
    setQuery("");
  }

  function onKeyDown(e: KeyboardEvent) {
    if (!open) {
      if (e.key === "ArrowDown" || e.key === "Enter" || e.key === " ") {
        setOpen(true);
        e.preventDefault();
      }
      return;
    }
    if (e.key === "ArrowDown") {
      setActive((a) => Math.min(a + 1, visible.length - 1));
      e.preventDefault();
    } else if (e.key === "ArrowUp") {
      setActive((a) => Math.max(a - 1, 0));
      e.preventDefault();
    } else if (e.key === "Enter") {
      if (visible[active]) choose(visible[active]);
      e.preventDefault();
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  }

  return (
    <div className="relative" ref={rootRef}>
      <button
        type="button"
        id={id}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={onKeyDown}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="field flex items-center justify-between text-left"
      >
        <span className={`truncate ${value ? "text-ink" : "text-slate"}`}>
          {value || placeholder}
        </span>
        <svg
          viewBox="0 0 20 20"
          className={`ml-2 h-4 w-4 shrink-0 text-slate transition-transform ${open ? "rotate-180" : ""}`}
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
        >
          <path d="M6 8l4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      {open && (
        <div className="absolute z-20 mt-1.5 w-full rounded-xl border border-hair bg-surface p-1 shadow-xl shadow-black/5">
          {searchable && (
            <input
              ref={searchRef}
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setActive(0);
              }}
              onKeyDown={onKeyDown}
              placeholder="Search…"
              className="mb-1 w-full rounded-lg border border-hair px-3 py-2 text-[14px] outline-none focus:border-ink"
            />
          )}
          <ul role="listbox" className="max-h-60 overflow-auto">
            {visible.length === 0 && (
              <li className="px-3 py-2 text-[14px] text-slate">No matches</li>
            )}
            {visible.map((opt, i) => (
              <li key={opt}>
                <button
                  type="button"
                  onClick={() => choose(opt)}
                  onMouseEnter={() => setActive(i)}
                  className={`flex w-full items-center justify-between rounded-lg px-3 py-2 text-left text-[14px] text-ink ${
                    i === active ? "bg-mist" : ""
                  } ${opt === value ? "font-medium" : ""}`}
                >
                  <span className="truncate">{opt}</span>
                  {opt === value && (
                    <svg viewBox="0 0 20 20" className="ml-2 h-4 w-4 shrink-0" fill="currentColor">
                      <path d="M8 13.5l-3.6-3.6 1.05-1.05L8 11.4l5.55-5.55L14.6 6.9z" />
                    </svg>
                  )}
                </button>
              </li>
            ))}
          </ul>
          {total > visible.length && (
            <p className="px-3 py-1.5 text-[12px] text-slate">
              Showing {visible.length} of {total} – refine your search
            </p>
          )}
        </div>
      )}
    </div>
  );
}
