import { useEffect, useState } from "react";
import { Route, Routes } from "react-router-dom";
import { TopNav } from "./components/TopNav";
import { Home } from "./pages/Home";
import { ModelPage } from "./pages/Model";
import { fetchModels, type ModelInfo } from "./api";

// Продуктовый порядок моделей в вебе (навбар + каталог), не алфавит из реестра. Неизвестные — в конец.
const MODEL_ORDER = ["cars", "wine", "diamonds", "loans", "uplift", "bayesian"];

function orderModels(models: ModelInfo[]): ModelInfo[] {
  const rank = (name: string) => {
    const i = MODEL_ORDER.indexOf(name);
    return i === -1 ? MODEL_ORDER.length : i;
  };
  return [...models].sort((a, b) => rank(a.name) - rank(b.name));
}

// Корневой лэйаут: стеклянный топ-навбар + маршруты + футер.
export default function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchModels()
      .then((ms) => setModels(orderModels(ms)))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="flex min-h-screen flex-col">
      <TopNav models={models} />
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<Home models={models} loading={loading} error={error} />} />
          <Route path="/models/:name" element={<ModelPage models={models} />} />
        </Routes>
      </main>
      <footer className="border-t border-hair bg-mist">
        <div className="mx-auto max-w-content px-6 py-8 text-xs text-slate">
          ML Playground – six trained ML models you can run in the browser, from notebook to prediction
        </div>
      </footer>
    </div>
  );
}
