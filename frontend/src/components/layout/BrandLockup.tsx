export function BrandLockup({ compact = false, variant = "dark" }: { compact?: boolean; variant?: "dark" | "light" }) {
  const titleColor = variant === "dark" ? "text-white" : "text-slateInk";
  const subtitleColor = variant === "dark" ? "text-slate-300" : "text-slate-600";
  const imageSize = compact ? "h-9 w-9" : "h-10 w-10";
  return (
    <div className="flex min-w-0 items-center gap-3" aria-label="Dados Sintéticos Brasileiros">
      <span className={`${imageSize} inline-flex shrink-0 items-center justify-center overflow-hidden rounded-xl border border-blue-200 bg-white p-1 shadow-sm`}>
        <img
          src="/brand/dados-sinteticos-br-logo.png"
          alt="Dados Sintéticos Brasileiros"
          width={compact ? 36 : 40}
          height={compact ? 36 : 40}
          className="h-full w-full object-contain"
        />
      </span>
      <span className="min-w-0 leading-tight">
        <span className={`${compact ? "text-sm" : "text-lg"} ${titleColor} block whitespace-nowrap font-bold leading-tight`}>
          Dados Sintéticos
        </span>
        <span className={`${compact ? "text-sm" : "text-lg"} ${titleColor} block whitespace-nowrap font-bold leading-tight`}>
          Brasileiros
        </span>
        {!compact ? <span className={`${subtitleColor} mt-1 block text-xs font-semibold leading-tight`}>Governança de dados sintéticos</span> : null}
      </span>
    </div>
  );
}
