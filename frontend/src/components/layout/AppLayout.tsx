import { Outlet } from "react-router-dom";
import { Menu, X } from "lucide-react";
import { useState } from "react";
import { BrandLockup } from "./BrandLockup";
import { Sidebar } from "./Sidebar";

export function AppLayout() {
  const [open, setOpen] = useState(false);
  return (
    <div className="min-h-screen bg-app text-slateInk lg:flex">
      <div className="hidden lg:block">
        <Sidebar />
      </div>
      <div className="sticky top-0 z-30 flex items-center justify-between border-b border-borderSoft bg-white px-4 py-3 lg:hidden">
        <BrandLockup compact variant="light" />
        <button
          type="button"
          className="inline-flex items-center gap-2 rounded-lg border border-borderSoft px-3 py-2 text-sm font-semibold text-slateInk"
          onClick={() => setOpen(true)}
          aria-label="Abrir navegação"
        >
          <Menu className="h-4 w-4" aria-hidden="true" />
          Menu
        </button>
      </div>
      {open ? (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Navegação principal">
          <button type="button" className="absolute inset-0 bg-slate-950/40" aria-label="Fechar navegação" onClick={() => setOpen(false)} />
          <div className="relative h-full w-80 max-w-[86vw]">
            <Sidebar onNavigate={() => setOpen(false)} />
            <button
              type="button"
              className="absolute right-3 top-3 rounded-lg border border-slate-600 bg-slate-800 p-2 text-white"
              onClick={() => setOpen(false)}
              aria-label="Fechar menu"
            >
              <X className="h-4 w-4" aria-hidden="true" />
            </button>
          </div>
        </div>
      ) : null}
      <main className="min-w-0 flex-1">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
