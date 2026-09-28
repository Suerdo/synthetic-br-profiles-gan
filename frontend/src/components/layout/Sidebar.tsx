import { Database, FileText, Scale } from "lucide-react";
import { NavLink } from "react-router-dom";
import { BrandLockup } from "./BrandLockup";

const items = [
  { to: "/", label: "Gerar Dados", icon: Database },
  { to: "/modelos", label: "Modelos", icon: FileText },
  { to: "/governanca", label: "Governança", icon: Scale }
];

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <aside className="h-full w-full border-r border-slate-700 bg-slateInk text-slate-100 lg:min-h-screen lg:w-72">
      <div className="px-5 py-6">
        <BrandLockup />
      </div>
      <nav className="space-y-1 px-3 pb-5" aria-label="Navegação principal">
        {items.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              onClick={onNavigate}
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
