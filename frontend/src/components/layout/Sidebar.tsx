import { Database, FileText, Scale } from "lucide-react";
import { NavLink } from "react-router-dom";

const items = [
  { to: "/", label: "Gerar dados", icon: Database },
  { to: "/modelos", label: "Modelos", icon: FileText },
  { to: "/governanca", label: "Governança", icon: Scale }
];

export function Sidebar() {
  return (
    <aside className="min-h-screen w-full border-r border-slate-700 bg-slateInk text-slate-100 lg:w-72">
      <div className="px-5 py-6">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-blue-300">Dados Sintéticos BR</p>
        <h1 className="mt-2 text-xl font-bold text-white">Plataforma Web</h1>
      </div>
      <nav className="space-y-1 px-3" aria-label="Navegação principal">
        {items.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-3 text-sm font-semibold transition ${
                  isActive
                    ? "border-l-4 border-blue-300 bg-navy text-white"
                    : "border-l-4 border-transparent text-slate-300 hover:bg-slate-800 hover:text-white"
                }`
              }
            >
              <Icon className="h-4 w-4" aria-hidden="true" />
              {item.label}
            </NavLink>
          );
        })}
      </nav>
    </aside>
  );
}
