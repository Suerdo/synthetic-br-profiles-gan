import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Navigate, Route, Routes } from "react-router-dom";

import { AppLayout } from "./components/layout/AppLayout";
import { useEphemeralSession } from "./hooks/useEphemeralSession";
import { GeneratePage } from "./pages/GeneratePage";
import { GovernancePage } from "./pages/GovernancePage";
import { ModelsPage } from "./pages/ModelsPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 30_000
    }
  }
});

export function App() {
  const sessionId = useEphemeralSession();
  return (
    <QueryClientProvider client={queryClient}>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index element={<GeneratePage sessionId={sessionId} />} />
          <Route path="modelos" element={<ModelsPage />} />
          <Route path="governanca" element={<GovernancePage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </QueryClientProvider>
  );
}
