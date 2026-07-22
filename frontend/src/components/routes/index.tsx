import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";

export const Route = createFileRoute("/")({
  component: Index,
});

type Field =
  | { key: string; label: string; type: "number"; hint?: string; step?: string; min?: number; max?: number }
  | { key: string; label: string; type: "select"; options: { value: string; label: string }[]; hint?: string };

const SECTIONS: { title: string; icon: string; fields: Field[] }[] = [
  {
    title: "Demographics",
    icon: "👤",
    fields: [
      { key: "age", label: "Age", type: "number", hint: "Years (e.g. 54)", min: 1, max: 120 },
      {
        key: "sex",
        label: "Sex",
        type: "select",
        options: [
          { value: "1", label: "Male" },
          { value: "0", label: "Female" },
        ],
      },
    ],
  },
  {
    title: "Symptoms",
    icon: "🫀",
    fields: [
      {
        key: "cp",
        label: "Chest Pain Type",
        type: "select",
        options: [
          { value: "0", label: "Typical Angina" },
          { value: "1", label: "Atypical Angina" },
          { value: "2", label: "Non-anginal Pain" },
          { value: "3", label: "Asymptomatic" },
        ],
      },
      {
        key: "exang",
        label: "Exercise-Induced Angina",
        type: "select",
        options: [
          { value: "1", label: "Yes" },
          { value: "0", label: "No" },
        ],
      },
    ],
  },
  {
    title: "Vitals & Labs",
    icon: "🧪",
    fields: [
      { key: "trestbps", label: "Resting BP", type: "number", hint: "mm Hg (e.g. 130)", min: 60, max: 260 },
      { key: "chol", label: "Cholesterol", type: "number", hint: "mg/dl (e.g. 246)", min: 80, max: 700 },
      {
        key: "fbs",
        label: "Fasting Blood Sugar > 120",
        type: "select",
        options: [
          { value: "1", label: "True" },
          { value: "0", label: "False" },
        ],
      },
      { key: "thalach", label: "Max Heart Rate", type: "number", hint: "bpm (e.g. 150)", min: 50, max: 250 },
    ],
  },
  {
    title: "ECG & Imaging",
    icon: "📈",
    fields: [
      {
        key: "restecg",
        label: "Resting ECG",
        type: "select",
        options: [
          { value: "0", label: "Normal" },
          { value: "1", label: "ST-T abnormality" },
          { value: "2", label: "LV hypertrophy" },
        ],
      },
      { key: "oldpeak", label: "ST Depression", type: "number", hint: "e.g. 1.4", step: "0.1", min: 0, max: 10 },
      {
        key: "slope",
        label: "ST Slope",
        type: "select",
        options: [
          { value: "0", label: "Upsloping" },
          { value: "1", label: "Flat" },
          { value: "2", label: "Downsloping" },
        ],
      },
      {
        key: "ca",
        label: "Major Vessels (ca)",
        type: "select",
        options: ["0", "1", "2", "3", "4"].map((v) => ({ value: v, label: v })),
      },
      {
        key: "thal",
        label: "Thalassemia",
        type: "select",
        options: [
          { value: "0", label: "Null" },
          { value: "1", label: "Fixed defect" },
          { value: "2", label: "Normal" },
          { value: "3", label: "Reversible defect" },
        ],
      },
    ],
  },
];

const ALL_FIELDS = SECTIONS.flatMap((s) => s.fields);

function scoreRisk(v: Record<string, string>): number {
  // Heuristic scoring inspired by common heart.csv feature weights.
  const n = (k: string) => parseFloat(v[k] || "0");
  let s = 0;
  s += Math.max(0, (n("age") - 40) / 40) * 0.9;
  s += n("sex") === 1 ? 0.25 : 0;
  s += n("cp") === 0 ? 0.6 : n("cp") === 3 ? 0.4 : 0.1;
  s += Math.max(0, (n("trestbps") - 120) / 80) * 0.6;
  s += Math.max(0, (n("chol") - 200) / 200) * 0.5;
  s += n("fbs") ? 0.2 : 0;
  s += n("restecg") === 1 ? 0.3 : n("restecg") === 2 ? 0.2 : 0;
  s += Math.max(0, (170 - n("thalach")) / 100) * 0.7;
  s += n("exang") ? 0.55 : 0;
  s += Math.min(1, n("oldpeak") / 4) * 0.7;
  s += n("slope") === 1 ? 0.3 : n("slope") === 2 ? 0.5 : 0;
  s += Math.min(1, n("ca") / 3) * 0.7;
  s += n("thal") === 3 ? 0.5 : n("thal") === 1 ? 0.3 : 0;
  const prob = 1 / (1 + Math.exp(-(s - 2.4)));
  return Math.round(prob * 100);
}

function Index() {
  const [values, setValues] = useState<Record<string, string>>({});
  const [result, setResult] = useState<{ risk: number } | null>(null);
  const [loading, setLoading] = useState(false);

  const complete = useMemo(
    () => ALL_FIELDS.every((f) => values[f.key] !== undefined && values[f.key] !== ""),
    [values],
  );

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!complete) return;
    setLoading(true);
    setResult(null);
    setTimeout(() => {
      setResult({ risk: scoreRisk(values) });
      setLoading(false);
    }, 700);
  };

  const reset = () => {
    setValues({});
    setResult(null);
  };

  const risk = result?.risk ?? 0;
  const level = risk >= 60 ? "high" : risk >= 30 ? "moderate" : "low";

  return (
    <main className="relative min-h-screen overflow-hidden text-foreground" style={{ background: "var(--gradient-bg)" }}>
      {/* ambient glow */}
      <div aria-hidden className="pointer-events-none absolute inset-0">
        <div className="absolute -top-40 -left-40 h-[500px] w-[500px] rounded-full opacity-40 blur-3xl" style={{ background: "var(--gradient-hero)" }} />
        <div className="absolute top-1/3 -right-40 h-[420px] w-[420px] rounded-full opacity-25 blur-3xl" style={{ background: "linear-gradient(135deg, var(--accent), var(--primary))" }} />
        <div className="absolute inset-0 opacity-[0.04]" style={{ backgroundImage: "radial-gradient(currentColor 1px, transparent 1px)", backgroundSize: "22px 22px" }} />
      </div>

      <div className="relative mx-auto max-w-6xl px-6 py-10 md:py-16">
        {/* Header */}
        <header className="mb-10 flex flex-col items-start gap-5 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-4">
            <div
              className="grid h-14 w-14 place-items-center rounded-2xl text-2xl"
              style={{ background: "var(--gradient-hero)", boxShadow: "var(--shadow-glow)" }}
            >
              <PulseHeart />
            </div>
            <div>
              <div className="flex items-center gap-2 text-xs uppercase tracking-[0.25em] text-muted-foreground">
                <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full" style={{ background: "var(--primary)" }} />
                CardioSense · v1.0
              </div>
              <h1 className="mt-1 text-3xl font-semibold tracking-tight md:text-4xl">
                Heart Disease{" "}
                <span
                  className="bg-clip-text text-transparent"
                  style={{ backgroundImage: "var(--gradient-hero)" }}
                >
                  Risk Predictor
                </span>
              </h1>
              <p className="mt-1 max-w-xl text-sm text-muted-foreground">
                Enter patient vitals — the model returns a probabilistic risk score based on the classic Cleveland heart dataset features.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 rounded-full border border-border bg-card/60 px-4 py-2 text-xs text-muted-foreground backdrop-blur">
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: "var(--success)" }} />
            Model online · 13 features
          </div>
        </header>

        <div className="grid gap-6 lg:grid-cols-[1.6fr_1fr]">
          {/* Form card */}
          <form
            onSubmit={submit}
            className="relative overflow-hidden rounded-3xl border border-border p-6 md:p-8"
            style={{ background: "var(--gradient-card)", boxShadow: "var(--shadow-elegant)", backdropFilter: "blur(12px)" }}
          >
            <div className="pointer-events-none absolute inset-x-0 top-0 h-px" style={{ background: "linear-gradient(90deg, transparent, oklch(1 0 0 / 30%), transparent)" }} />

            {SECTIONS.map((section) => (
              <section key={section.title} className="mb-8 last:mb-0">
                <div className="mb-4 flex items-center gap-3">
                  <span className="grid h-8 w-8 place-items-center rounded-lg border border-border bg-muted/40 text-base">
                    {section.icon}
                  </span>
                  <h2 className="text-sm font-semibold uppercase tracking-widest text-muted-foreground">
                    {section.title}
                  </h2>
                  <div className="h-px flex-1 bg-border" />
                </div>
                <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                  {section.fields.map((f) => (
                    <FieldInput
                      key={f.key}
                      field={f}
                      value={values[f.key] ?? ""}
                      onChange={(v) => setValues((prev) => ({ ...prev, [f.key]: v }))}
                    />
                  ))}
                </div>
              </section>
            ))}

            <div className="mt-6 flex flex-col-reverse gap-3 sm:flex-row sm:items-center sm:justify-between">
              <button
                type="button"
                onClick={reset}
                className="rounded-xl border border-border px-5 py-3 text-sm font-medium text-muted-foreground transition hover:bg-muted/40 hover:text-foreground"
              >
                Reset
              </button>
              <button
                type="submit"
                disabled={!complete || loading}
                className="group relative overflow-hidden rounded-xl px-6 py-3 text-sm font-semibold text-primary-foreground transition disabled:cursor-not-allowed disabled:opacity-50"
                style={{ background: "var(--gradient-hero)", boxShadow: "var(--shadow-glow)" }}
              >
                <span className="relative z-10 flex items-center justify-center gap-2">
                  {loading ? (
                    <>
                      <Spinner /> Analyzing…
                    </>
                  ) : (
                    <>🔍 Predict Risk</>
                  )}
                </span>
                <span className="absolute inset-0 -translate-x-full bg-white/20 transition-transform duration-500 group-hover:translate-x-0" />
              </button>
            </div>
          </form>

          {/* Result panel */}
          <aside className="lg:sticky lg:top-8 h-fit">
            <div
              className="relative overflow-hidden rounded-3xl border border-border p-6 md:p-8"
              style={{ background: "var(--gradient-card)", boxShadow: "var(--shadow-elegant)", backdropFilter: "blur(12px)" }}
            >
              <h3 className="text-xs font-semibold uppercase tracking-[0.25em] text-muted-foreground">
                Prediction
              </h3>

              {!result && !loading && (
                <div className="mt-8 flex flex-col items-center text-center">
                  <div className="relative mb-6 grid h-40 w-40 place-items-center">
                    <div className="absolute inset-0 animate-ping rounded-full opacity-20" style={{ background: "var(--gradient-hero)" }} />
                    <div className="grid h-32 w-32 place-items-center rounded-full border border-border bg-card">
                      <PulseHeart size={44} />
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Fill in all fields and press <span className="text-foreground">Predict Risk</span> to see the estimate.
                  </p>
                </div>
              )}

              {loading && (
                <div className="mt-8 flex flex-col items-center text-center">
                  <Spinner size={44} />
                  <p className="mt-4 text-sm text-muted-foreground">Analyzing patient vitals…</p>
                </div>
              )}

              {result && !loading && (
                <div className="mt-6 animate-[fadeIn_.5s_ease-out]">
                  <RiskGauge value={risk} level={level} />
                  <div
                    className="mt-6 rounded-2xl border p-4 text-sm"
                    style={{
                      borderColor:
                        level === "high"
                          ? "oklch(0.62 0.25 27 / 40%)"
                          : level === "moderate"
                            ? "oklch(0.75 0.18 80 / 40%)"
                            : "oklch(0.72 0.18 155 / 40%)",
                      background:
                        level === "high"
                          ? "oklch(0.62 0.25 27 / 10%)"
                          : level === "moderate"
                            ? "oklch(0.75 0.18 80 / 10%)"
                            : "oklch(0.72 0.18 155 / 10%)",
                    }}
                  >
                    <div className="mb-1 font-semibold">
                      {level === "high"
                        ? "Elevated risk detected"
                        : level === "moderate"
                          ? "Moderate risk"
                          : "Low risk"}
                    </div>
                    <p className="text-muted-foreground">
                      {level === "high"
                        ? "Model indicates a high probability of heart disease. Recommend clinical follow-up."
                        : level === "moderate"
                          ? "Some risk factors present. Consider monitoring and lifestyle review."
                          : "Vitals appear within a healthier envelope. Continue routine care."}
                    </p>
                    <p className="mt-3 text-[11px] uppercase tracking-widest text-muted-foreground">
                      Educational demo · not medical advice
                    </p>
                  </div>
                </div>
              )}
            </div>
          </aside>
        </div>

        <footer className="mt-10 text-center text-xs text-muted-foreground">
          Built with the Kaggle <span className="text-foreground">heart.csv</span> feature set · UCI Cleveland cohort
        </footer>
      </div>
    </main>
  );
}

function FieldInput({
  field,
  value,
  onChange,
}: {
  field: Field;
  value: string;
  onChange: (v: string) => void;
}) {
  const base =
    "w-full rounded-xl border border-border bg-input/60 px-3.5 py-2.5 text-sm text-foreground placeholder:text-muted-foreground/60 backdrop-blur transition focus:border-transparent focus:outline-none focus:ring-2 focus:ring-[color:var(--ring)]";
  return (
    <label className="group block">
      <span className="mb-1.5 block text-xs font-medium text-muted-foreground group-focus-within:text-foreground">
        {field.label}
      </span>
      {field.type === "number" ? (
        <input
          className={base}
          type="number"
          inputMode="decimal"
          step={field.step ?? "1"}
          min={field.min}
          max={field.max}
          value={value}
          placeholder={field.hint?.replace(/^.*e\.g\.\s*/, "")}
          onChange={(e) => onChange(e.target.value)}
        />
      ) : (
        <select
          className={base + " appearance-none pr-9"}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          style={{
            backgroundImage:
              "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23aaa' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'><polyline points='6 9 12 15 18 9'/></svg>\")",
            backgroundRepeat: "no-repeat",
            backgroundPosition: "right 12px center",
          }}
        >
          <option value="">Select…</option>
          {field.options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      )}
      {field.type === "number" && field.hint && (
        <span className="mt-1 block text-[11px] text-muted-foreground/70">{field.hint}</span>
      )}
    </label>
  );
}

function RiskGauge({ value, level }: { value: number; level: "low" | "moderate" | "high" }) {
  const size = 200;
  const stroke = 14;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const dash = (value / 100) * c;
  const color =
    level === "high" ? "oklch(0.66 0.24 20)" : level === "moderate" ? "oklch(0.78 0.18 75)" : "oklch(0.72 0.18 155)";
  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <defs>
            <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity="0.4" />
              <stop offset="100%" stopColor={color} />
            </linearGradient>
          </defs>
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="oklch(1 0 0 / 8%)" strokeWidth={stroke} />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke="url(#g)"
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={`${dash} ${c}`}
            style={{ transition: "stroke-dasharray 1s ease-out", filter: `drop-shadow(0 0 12px ${color})` }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <div className="text-5xl font-bold tracking-tight">{value}%</div>
          <div className="mt-1 text-[11px] uppercase tracking-[0.25em] text-muted-foreground">Risk</div>
        </div>
      </div>
    </div>
  );
}

function PulseHeart({ size = 26 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className="animate-pulse">
      <path
        d="M12 21s-7-4.5-9.5-9C.8 8.6 2.7 4 6.5 4c2 0 3.5 1 5.5 3 2-2 3.5-3 5.5-3 3.8 0 5.7 4.6 4 8-2.5 4.5-9.5 9-9.5 9z"
        fill="white"
      />
    </svg>
  );
}

function Spinner({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className="animate-spin">
      <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
      <path d="M21 12a9 9 0 0 0-9-9" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}
