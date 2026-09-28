import { Database, FileText, Scale } from "lucide-react";
import { NavLink } from "react-router-dom";

const items = [
  { to: "/", label: "Gerar Dados", icon: Database },
  { to: "/modelos", label: "Modelos", icon: FileText },
  { to: "/governanca", label: "Governança", icon: Scale }
];

export function Sidebar() {
  return (
    <aside className="w-full border-r border-slate-700 bg-slateInk text-slate-100 lg:min-h-screen lg:w-72">
      <div className="px-5 py-6">
        <h1 className="text-xl font-bold text-white">Dados Sintéticos Brasileiro</h1>
      </div>
      <nav className="space-y-1 px-3 pb-5" aria-label="Navegação principal">
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
