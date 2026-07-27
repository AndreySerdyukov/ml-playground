import { useEffect, useState } from "react";
import { Route, Routes } from "react-router-dom";
import { TopNav } from "./components/TopNav";
import { Home } from "./pages/Home";
import { ModelPage } from "./pages/Model";
import { fetchModels, type ModelInfo } from "./api";

// Корневой лэйаут: стеклянный топ-навбар + маршруты + футер.
export default function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchModels()
      .then(setModels)
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
          ML Playground — interactive demos of trained models. Weights are illustrative stubs for now.
        </div>
      </footer>
    </div>
  );
}
