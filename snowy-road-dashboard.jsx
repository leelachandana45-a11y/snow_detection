import { useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
         ScatterChart, Scatter, LineChart, Line, RadarChart, Radar, PolarGrid,
         PolarAngleAxis, PolarRadiusAxis, Cell } from "recharts";

const COLORS = {
  Dry: "#2ECC71", "Fresh Snow": "#3498DB", "Transparent Ice": "#E74C3C",
  "Granular Snow": "#9B59B6", "Mixed Ice": "#F39C12"
};
const METHOD_COLORS = {
  MM: "#E74C3C", TM: "#E67E22", IM: "#2ECC71",
  MTFM: "#3498DB", IMFM: "#9B59B6", ITFM: "#1ABC9C", IMTFM: "#F39C12"
};

// ── Simulation data (mirrors Python model) ─────────────────────
const AP_DATA = {
  "3-class":   { MM:0.708, TM:0.715, IM:1.000, MTFM:0.715, IMFM:0.986, ITFM:1.000, IMTFM:0.965 },
  "4-class-I": { MM:0.375, TM:0.365, IM:0.844, MTFM:0.401, IMFM:0.672, ITFM:0.792, IMTFM:0.635 },
  "4-class-II":{ MM:0.464, TM:0.484, IM:0.974, MTFM:0.464, IMFM:0.885, ITFM:0.938, IMTFM:0.818 },
  "5-class":   { MM:0.412, TM:0.438, IM:0.842, MTFM:0.417, IMFM:0.733, ITFM:0.792, IMTFM:0.700 },
};

const METHODS = ["MM","TM","IM","MTFM","IMFM","ITFM","IMTFM"];

const FRICTION_DATA = [
  { surface:"Dry",             friction:0.85, entropy:0.74, alpha:38, danger:false },
  { surface:"Fresh Snow",      friction:0.38, entropy:0.72, alpha:42, danger:false },
  { surface:"Transparent Ice", friction:0.19, entropy:0.78, alpha:23, danger:true  },
  { surface:"Granular Snow",   friction:0.50, entropy:0.71, alpha:41, danger:false },
  { surface:"Mixed Ice",       friction:0.25, entropy:0.76, alpha:27, danger:true  },
];

const MODEL_PERF = [
  { model:"SIWNet",       mae:0.089, rmse:0.124, params:0.7,  is:0.312, crps:0.089 },
  { model:"ResNet50",     mae:0.091, rmse:0.127, params:23.6, is:0.481, crps:0.091 },
  { model:"ResNet50v2",   mae:0.092, rmse:0.128, params:23.6, is:0.478, crps:0.092 },
  { model:"VGG19",        mae:0.103, rmse:0.143, params:139.6,is:0.531, crps:0.103 },
  { model:"EfficientNet", mae:0.134, rmse:0.181, params:4.0,  is:0.694, crps:0.134 },
];

const VIP_FEATURES = [
  { feature:"Temp",  vip:1.82 }, { feature:"M0(UV)", vip:1.61 }, { feature:"C0",  vip:1.45 },
  { feature:"C1",    vip:1.38 }, { feature:"M3(Hum)",vip:1.29 }, { feature:"T0",  vip:1.21 },
  { feature:"C2",    vip:1.15 }, { feature:"T1",     vip:1.09 }, { feature:"C3",  vip:1.03 },
  { feature:"M4(AT)",vip:0.98 }, { feature:"T2",     vip:0.91 }, { feature:"C4",  vip:0.85 },
];

const EQUATIONS = [
  { id:1,  name:"GLCM Energy",           eq:"E = ΣΣ [G(i,j)]²",              ref:"Yang & Lei [1] Eq.1" },
  { id:2,  name:"Moment of Inertia",     eq:"I = Σ n²·[Σ|i−j|=n G(i,j)]",   ref:"Yang & Lei [1] Eq.2" },
  { id:3,  name:"Inv. Diff. Moment",     eq:"L = ΣΣ G(i,j)/[1+(i−j)²]",     ref:"Yang & Lei [1] Eq.3" },
  { id:4,  name:"GLCM Entropy",          eq:"EN = −ΣΣ G(i,j)·log G(i,j)",   ref:"Yang & Lei [1] Eq.4" },
  { id:5,  name:"Correlation",           eq:"C = ΣΣ(ij)G−μᵢμⱼ / σᵢσⱼ",    ref:"Yang & Lei [1] Eq.5" },
  { id:6,  name:"VIP Score",             eq:"VIP = √[k·Σr²(y,cₕ)w²ₕⱼ/Σr²]",ref:"Yang & Lei [1] Eq.6" },
  { id:7,  name:"Average Precision",     eq:"AP = (1/n)·ΣPᵢ",               ref:"Yang & Lei [1] Eq.7" },
  { id:8,  name:"DRAP 3→4",             eq:"DRAP = (AP₃−AP₄)/AP₃",          ref:"Yang & Lei [1] Eq.8" },
  { id:9,  name:"DRAP 4→5",             eq:"DRAP = (AP₄−AP₅)/AP₄",          ref:"Yang & Lei [1] Eq.9" },
  { id:10, name:"Coherence Vector",      eq:"k = [SHH+SVV, SHH−SVV, 2SHV]ᵀ",ref:"Vassilev [2] Eq.3" },
  { id:11, name:"Coherence Matrix",      eq:"T̂ = MM†/N",                    ref:"Vassilev [2] Eq.5" },
  { id:12, name:"Eigendecomposition",    eq:"T̂ = Σλᵢ[eᵢ·eᵢ†]",             ref:"Vassilev [2] Eq.6" },
  { id:13, name:"Probability Weights",   eq:"Pᵢ = λᵢ/(λ₁+λ₂+λ₃)",          ref:"Vassilev [2] Eq.8" },
  { id:14, name:"Target Entropy",        eq:"H = −ΣPᵢ·log₃(Pᵢ)",            ref:"Vassilev [2] Eq.7" },
  { id:15, name:"Auxiliary Angle α",     eq:"α = ΣPᵢ·arccos|eᵢ₁|",          ref:"Vassilev [2] Eq.9" },
  { id:16, name:"Truncated Normal PDF",  eq:"p = φ((x−μ)/σ)/[Φ(b)−Φ(a)]",  ref:"Ojala & Seppänen [3] Eq.1" },
  { id:17, name:"Normal PDF",            eq:"φ = (1/σ√2π)·exp(−(x−μ)²/2σ²)",ref:"Ojala & Seppänen [3] Eq.2" },
  { id:18, name:"Normal CDF",            eq:"Φ = (1/2)[1+erf((b−μ)/σ√2π)]", ref:"Ojala & Seppänen [3] Eq.3" },
  { id:19, name:"Neg. Log-Likelihood",   eq:"−ln p = ln σ+(μ−x)²/2σ²+ln…",  ref:"Ojala & Seppänen [3] Eq.4" },
  { id:20, name:"Batch Loss",            eq:"L = Σᵢ −ln p(f̂ᵢ,σ̂ᵢ,a,b;fᵢ)", ref:"Ojala & Seppänen [3] Eq.5" },
  { id:21, name:"Friction Normalization",eq:"f = (grip−g_min)/(g_max−g_min)",ref:"Ojala & Seppänen [3] Sec.III-A" },
  { id:22, name:"Prediction Interval",   eq:"PI = [F⁻¹((1−c)/2), F⁻¹((1+c)/2)]",ref:"Ojala & Seppänen [3] Sec.III-D" },
  { id:23, name:"Interval Score",        eq:"IS = (u−l)+(2/α)[(l−y)⁺+(y−u)⁺]",ref:"Ojala & Seppänen [3] Sec.III-D" },
  { id:24, name:"CRPS",                  eq:"CRPS = E|X−y|−(1/2)E|X−X'|",   ref:"Ojala & Seppänen [3] Sec.III-D" },
  { id:25, name:"MAE",                   eq:"MAE = (1/n)Σ|fᵢ−f̂ᵢ|",         ref:"Ojala & Seppänen [3] Table II" },
  { id:26, name:"RMSE",                  eq:"RMSE = √[(1/n)Σ(fᵢ−f̂ᵢ)²]",    ref:"Ojala & Seppänen [3] Table II" },
  { id:27, name:"Radar Surface Score",   eq:"S = 0.6·(1−H)+0.4·(1−α/90)",   ref:"Derived from Vassilev [2]" },
];

// ── Helpers ─────────────────────────────────────────────────────
function StatusBadge({ level }) {
  const styles = {
    SAFE:    "bg-green-900/60 text-green-300 border border-green-600",
    CAUTION: "bg-yellow-900/60 text-yellow-300 border border-yellow-600",
    DANGER:  "bg-red-900/60 text-red-300 border border-red-600 animate-pulse",
  };
  return <span className={`px-2 py-0.5 rounded text-xs font-bold ${styles[level]}`}>{level}</span>;
}

function MetricCard({ label, value, unit, sub, color }) {
  return (
    <div className="bg-slate-800/70 border border-slate-700 rounded-xl p-4 flex flex-col gap-1">
      <div className="text-slate-400 text-xs uppercase tracking-widest">{label}</div>
      <div className="flex items-end gap-1">
        <span className="text-3xl font-black" style={{ color }}>{value}</span>
        <span className="text-slate-400 text-sm pb-1">{unit}</span>
      </div>
      {sub && <div className="text-slate-500 text-xs">{sub}</div>}
    </div>
  );
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-slate-900 border border-slate-600 rounded-lg p-3 text-sm shadow-xl">
      <p className="text-slate-300 font-semibold mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }}>{p.name}: {typeof p.value === 'number' ? (p.value * (p.value <= 1 ? 100 : 1)).toFixed(1)}{p.value <= 1 ? '%' : ''}</p>
      ))}
    </div>
  );
};

// ── Main Dashboard ───────────────────────────────────────────────
export default function Dashboard() {
  const [activeTab, setActiveTab] = useState("overview");
  const [selectedScenario, setSelectedScenario] = useState("4-class-II");
  const [liveTemp, setLiveTemp] = useState(-4.2);
  const [liveFriction, setLiveFriction] = useState(0.41);
  const [liveClass, setLiveClass] = useState("Fresh Snow");
  const [liveEntropy, setLiveEntropy] = useState(0.72);
  const [alertPulse, setAlertPulse] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      const t = -15 + Math.random() * 25;
      const f = Math.max(0.09, Math.min(1, 0.4 + t * 0.015 + (Math.random() - 0.5) * 0.1));
      const cls = f > 0.7 ? "Dry" : f > 0.45 ? "Granular Snow" : f > 0.35 ? "Fresh Snow" : f > 0.22 ? "Mixed Ice" : "Transparent Ice";
      setLiveTemp(t.toFixed(1));
      setLiveFriction(f.toFixed(2));
      setLiveClass(cls);
      setLiveEntropy((0.70 + Math.random() * 0.10).toFixed(3));
      setAlertPulse(f < 0.25);
    }, 2500);
    return () => clearInterval(interval);
  }, []);

  // Build bar chart data for scenario
  const apChartData = METHODS.map(m => ({
    method: m,
    AP: +(AP_DATA[selectedScenario][m] * 100).toFixed(1),
  }));

  const frictionChartData = FRICTION_DATA.map(d => ({
    name: d.surface.split(" ")[0],
    friction: +(d.friction * 100).toFixed(0),
    entropy: +(d.entropy * 100).toFixed(0),
  }));

  const draData = ["IM","MTFM","IMFM","ITFM","IMTFM"].map(m => ({
    method: m,
    "3→4 (Case I)":  +((AP_DATA["3-class"][m] - AP_DATA["4-class-I"][m]) / AP_DATA["3-class"][m] * 100).toFixed(1),
    "3→4 (Case II)": +((AP_DATA["3-class"][m] - AP_DATA["4-class-II"][m]) / AP_DATA["3-class"][m] * 100).toFixed(1),
    "4→5 (Case I)":  +((AP_DATA["4-class-I"][m] - AP_DATA["5-class"][m]) / AP_DATA["4-class-I"][m] * 100).toFixed(1),
  }));

  const riskLevel = liveFriction < 0.25 ? "DANGER" : liveFriction < 0.45 ? "CAUTION" : "SAFE";
  const riskColor = { SAFE: "#2ECC71", CAUTION: "#F39C12", DANGER: "#E74C3C" }[riskLevel];

  const tabs = [
    { id:"overview",    label:"🛣️ Overview" },
    { id:"classification", label:"📊 Classification" },
    { id:"radar",       label:"📡 Radar Analysis" },
    { id:"siwnet",      label:"🤖 SIWNet Model" },
    { id:"equations",   label:"📐 Equations" },
    { id:"live",        label:"🔴 Live Monitor" },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-white font-mono" style={{ fontFamily: "'Courier New', monospace" }}>
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border-b border-slate-700 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-xl font-black tracking-tight text-cyan-400">
              ❄️ CLIMATE RESILIENT TRANSPORTATION SYSTEM
            </h1>
            <p className="text-slate-400 text-xs mt-0.5">Snowy Road Condition Detection · IEEE Sensor Fusion Dashboard</p>
          </div>
          <div className="flex items-center gap-3">
            <StatusBadge level={riskLevel} />
            <span className="text-slate-400 text-xs">{new Date().toLocaleTimeString()}</span>
            <div className={`w-2 h-2 rounded-full ${alertPulse ? "bg-red-500 animate-ping" : "bg-green-500"}`} />
          </div>
        </div>
      </div>

      {/* Nav */}
      <div className="bg-slate-900 border-b border-slate-700 px-6">
        <div className="max-w-7xl mx-auto flex gap-1 overflow-x-auto">
          {tabs.map(t => (
            <button key={t.id} onClick={() => setActiveTab(t.id)}
              className={`px-4 py-3 text-xs font-bold tracking-wide whitespace-nowrap border-b-2 transition-all ${
                activeTab === t.id
                  ? "border-cyan-400 text-cyan-400"
                  : "border-transparent text-slate-400 hover:text-slate-200"
              }`}>
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 py-6 space-y-6">

        {/* ── OVERVIEW ─────────────────────────────────────── */}
        {activeTab === "overview" && (
          <div className="space-y-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <MetricCard label="IMTFM AP (5-class)" value="70.0" unit="%" sub="Best fusion method" color="#F39C12"/>
              <MetricCard label="SIWNet MAE" value="0.089" unit="" sub="Friction regression" color="#2ECC71"/>
              <MetricCard label="Surface Classes" value="5" unit="" sub="DR/FS/TI/GS/MI" color="#3498DB"/>
              <MetricCard label="Equations Used" value="27" unit="" sub="From 3 IEEE papers" color="#9B59B6"/>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* AP by scenario */}
              <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-sm font-bold text-cyan-300">AP by Method (Eq. 7)</h2>
                  <select value={selectedScenario} onChange={e => setSelectedScenario(e.target.value)}
                    className="bg-slate-700 text-slate-200 text-xs rounded px-2 py-1 border border-slate-600">
                    {Object.keys(AP_DATA).map(k => <option key={k} value={k}>{k}</option>)}
                  </select>
                </div>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={apChartData} margin={{ top:5, right:5, left:-20, bottom:5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334" />
                    <XAxis dataKey="method" tick={{ fill:"#94a3b8", fontSize:10 }} />
                    <YAxis tick={{ fill:"#94a3b8", fontSize:10 }} domain={[0,110]} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="AP" radius={[3,3,0,0]}>
                      {apChartData.map((e,i) => <Cell key={i} fill={Object.values(METHOD_COLORS)[i]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Friction per class */}
              <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-4">
                <h2 className="text-sm font-bold text-cyan-300 mb-3">Friction Factor vs Entropy (Eq. 14, 21)</h2>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={frictionChartData} margin={{ top:5, right:5, left:-20, bottom:5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334" />
                    <XAxis dataKey="name" tick={{ fill:"#94a3b8", fontSize:10 }} />
                    <YAxis tick={{ fill:"#94a3b8", fontSize:10 }} domain={[0,110]} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="friction" name="Friction %" radius={[3,3,0,0]}>
                      {frictionChartData.map((e,i) => <Cell key={i} fill={Object.values(COLORS)[i]} />)}
                    </Bar>
                    <Bar dataKey="entropy" name="Entropy %" fill="#334" fillOpacity={0.5} radius={[3,3,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Surface info cards */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {FRICTION_DATA.map(d => (
                <div key={d.surface} className="bg-slate-800/50 border border-slate-700 rounded-xl p-3"
                  style={{ borderLeftColor: COLORS[d.surface], borderLeftWidth: 3 }}>
                  <div className="text-xs font-bold text-white mb-2">{d.surface}</div>
                  <div className="text-xs text-slate-400 space-y-1">
                    <div>f = <span className="text-white font-bold">{d.friction}</span></div>
                    <div>H = <span className="text-white font-bold">{d.entropy}</span></div>
                    <div>α = <span className="text-white font-bold">{d.alpha}°</span></div>
                    {d.danger && <StatusBadge level="DANGER" />}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── CLASSIFICATION ───────────────────────────────── */}
        {activeTab === "classification" && (
          <div className="space-y-6">
            <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
              <h2 className="text-sm font-bold text-cyan-300 mb-1">Graph 1 — DRAP Analysis (Eq. 8 & 9)</h2>
              <p className="text-slate-400 text-xs mb-4">Decline Rate of Average Precision when increasing classification complexity</p>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={draData} margin={{ top:5, right:20, left:-10, bottom:5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334" />
                  <XAxis dataKey="method" tick={{ fill:"#94a3b8", fontSize:11 }} />
                  <YAxis tick={{ fill:"#94a3b8", fontSize:10 }} unit="%" />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ color:"#94a3b8", fontSize:11 }} />
                  <Bar dataKey="3→4 (Case I)" fill="#3498DB" radius={[3,3,0,0]} />
                  <Bar dataKey="3→4 (Case II)" fill="#E74C3C" radius={[3,3,0,0]} />
                  <Bar dataKey="4→5 (Case I)" fill="#9B59B6" radius={[3,3,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
              <h2 className="text-sm font-bold text-cyan-300 mb-4">Method Performance Radar (All Scenarios)</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(AP_DATA).map(([scenario, data]) => {
                  const radarData = METHODS.map(m => ({ method: m, AP: +(data[m]*100).toFixed(0) }));
                  return (
                    <div key={scenario} className="bg-slate-900/40 rounded-xl p-3">
                      <div className="text-xs text-slate-400 font-bold mb-2 text-center">{scenario}</div>
                      <ResponsiveContainer width="100%" height={180}>
                        <RadarChart data={radarData}>
                          <PolarGrid stroke="#334" />
                          <PolarAngleAxis dataKey="method" tick={{ fill:"#94a3b8", fontSize:9 }} />
                          <PolarRadiusAxis tick={{ fill:"#94a3b8", fontSize:7 }} domain={[0,100]} />
                          <Radar name="AP%" dataKey="AP" stroke="#F39C12" fill="#F39C12" fillOpacity={0.25} />
                          <Tooltip content={<CustomTooltip />} />
                        </RadarChart>
                      </ResponsiveContainer>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* ── RADAR ANALYSIS ───────────────────────────────── */}
        {activeTab === "radar" && (
          <div className="space-y-6">
            <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
              <h2 className="text-sm font-bold text-cyan-300 mb-1">Graph 3 — Entropy vs Auxiliary Angle α (Eq. 14 & 15)</h2>
              <p className="text-slate-400 text-xs mb-4">Polarimetric radar surface characterisation — Vassilev [2]</p>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-slate-700 text-slate-400">
                      <th className="text-left py-2 px-3">Surface</th>
                      <th className="text-right py-2 px-3">Entropy H</th>
                      <th className="text-right py-2 px-3">α (°)</th>
                      <th className="text-right py-2 px-3">Radar Score S</th>
                      <th className="text-right py-2 px-3">Dominant Scatter</th>
                      <th className="text-right py-2 px-3">Risk</th>
                    </tr>
                  </thead>
                  <tbody>
                    {FRICTION_DATA.map(d => {
                      const S = (0.6 * (1 - d.entropy) + 0.4 * (1 - d.alpha / 90)).toFixed(3);
                      const scatter = d.alpha < 30 ? "Surface" : d.alpha < 60 ? "Dipole/Volume" : "Double Bounce";
                      return (
                        <tr key={d.surface} className="border-b border-slate-800 hover:bg-slate-700/30">
                          <td className="py-2 px-3 font-bold" style={{ color: COLORS[d.surface] }}>{d.surface}</td>
                          <td className="text-right py-2 px-3 text-slate-300">{d.entropy}</td>
                          <td className="text-right py-2 px-3 text-slate-300">{d.alpha}°</td>
                          <td className="text-right py-2 px-3 text-cyan-300 font-bold">{S}</td>
                          <td className="text-right py-2 px-3 text-slate-400">{scatter}</td>
                          <td className="text-right py-2 px-3">
                            <StatusBadge level={d.danger ? "DANGER" : d.friction < 0.4 ? "CAUTION" : "SAFE"} />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
                <h2 className="text-sm font-bold text-cyan-300 mb-3">Entropy Distribution by Class</h2>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={FRICTION_DATA.map(d => ({ name: d.surface.split(" ")[0], H: +(d.entropy*100).toFixed(0), alpha: d.alpha }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334" />
                    <XAxis dataKey="name" tick={{ fill:"#94a3b8", fontSize:10 }} />
                    <YAxis tick={{ fill:"#94a3b8", fontSize:10 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="H" name="Entropy %" radius={[3,3,0,0]}>
                      {FRICTION_DATA.map((d,i) => <Cell key={i} fill={COLORS[d.surface]} />)}
                    </Bar>
                    <Bar dataKey="alpha" name="α°" fill="#1ABC9C" fillOpacity={0.7} radius={[3,3,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
                <h2 className="text-sm font-bold text-cyan-300 mb-3">α Interpretation Guide</h2>
                <div className="space-y-3 text-xs">
                  {[
                    { range:"α ≈ 0°",  type:"Surface Scattering",  desc:"Smooth/wet surface — mirror-like reflection", color:"#2ECC71" },
                    { range:"α ≈ 45°", type:"Dipole / Volume",     desc:"Snow — diffuse volume scattering", color:"#3498DB" },
                    { range:"α ≈ 90°", type:"Double Bounce",       desc:"Ice corners — enhanced dihedral returns", color:"#E74C3C" },
                  ].map(item => (
                    <div key={item.range} className="flex gap-3 bg-slate-900/40 rounded-lg p-3">
                      <div className="w-14 font-black text-center rounded" style={{ color: item.color, backgroundColor: item.color + "22" }}>{item.range}</div>
                      <div>
                        <div className="font-bold text-white">{item.type}</div>
                        <div className="text-slate-400">{item.desc}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── SIWNET MODEL ─────────────────────────────────── */}
        {activeTab === "siwnet" && (
          <div className="space-y-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {MODEL_PERF.map(m => (
                <div key={m.model} className={`bg-slate-800/60 border rounded-xl p-4 ${m.model === "SIWNet" ? "border-yellow-500" : "border-slate-700"}`}>
                  <div className="text-xs font-bold text-white mb-3 flex items-center gap-2">
                    {m.model === "SIWNet" && <span className="text-yellow-400">★</span>}
                    {m.model}
                  </div>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between"><span className="text-slate-400">MAE</span><span className="text-white font-bold">{m.mae}</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">RMSE</span><span className="text-white">{m.rmse}</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">IS</span><span className="text-cyan-300">{m.is}</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">CRPS</span><span className="text-cyan-300">{m.crps}</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">Params(M)</span><span className="text-slate-300">{m.params}</span></div>
                  </div>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
                <h2 className="text-sm font-bold text-cyan-300 mb-3">MAE & RMSE Comparison (Eq. 25 & 26)</h2>
                <ResponsiveContainer width="100%" height={230}>
                  <BarChart data={MODEL_PERF} margin={{ top:5, right:5, left:-20, bottom:5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334" />
                    <XAxis dataKey="model" tick={{ fill:"#94a3b8", fontSize:9 }} />
                    <YAxis tick={{ fill:"#94a3b8", fontSize:10 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ color:"#94a3b8", fontSize:10 }} />
                    <Bar dataKey="mae" name="MAE" fill="#F39C12" radius={[3,3,0,0]} />
                    <Bar dataKey="rmse" name="RMSE" fill="#E74C3C" radius={[3,3,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
                <h2 className="text-sm font-bold text-cyan-300 mb-3">VIP Feature Importance (Eq. 6)</h2>
                <ResponsiveContainer width="100%" height={230}>
                  <BarChart data={VIP_FEATURES} layout="vertical" margin={{ top:5, right:20, left:10, bottom:5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334" />
                    <XAxis type="number" tick={{ fill:"#94a3b8", fontSize:9 }} domain={[0, 2]} />
                    <YAxis type="category" dataKey="feature" tick={{ fill:"#94a3b8", fontSize:9 }} width={55} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="vip" name="VIP Score" radius={[0,3,3,0]}>
                      {VIP_FEATURES.map((e,i) => <Cell key={i} fill={e.vip >= 1.0 ? "#F39C12" : "#3498DB"} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Prediction interval demo */}
            <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
              <h2 className="text-sm font-bold text-cyan-300 mb-1">Prediction Interval Visualization (Eq. 16–22)</h2>
              <p className="text-slate-400 text-xs mb-4">Truncated Normal distribution: 90% prediction interval per surface class</p>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                {FRICTION_DATA.map(d => {
                  const lo = Math.max(0, d.friction - 1.645 * 0.06).toFixed(2);
                  const hi = Math.min(1, d.friction + 1.645 * 0.06).toFixed(2);
                  return (
                    <div key={d.surface} className="bg-slate-900/50 rounded-xl p-3 text-xs">
                      <div className="font-bold mb-2" style={{ color: COLORS[d.surface] }}>{d.surface}</div>
                      <div className="text-slate-400 space-y-1">
                        <div>f̂ = <span className="text-white font-bold">{d.friction}</span></div>
                        <div>90% PI:</div>
                        <div className="text-cyan-300 font-bold">[{lo}, {hi}]</div>
                        <div className="w-full bg-slate-700 rounded h-2 mt-2">
                          <div className="bg-gradient-to-r from-cyan-600 to-cyan-400 h-2 rounded"
                            style={{ width: `${d.friction * 100}%` }} />
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* ── EQUATIONS ────────────────────────────────────── */}
        {activeTab === "equations" && (
          <div className="space-y-4">
            <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-4">
              <h2 className="text-sm font-bold text-cyan-300 mb-1">27 Equations from 3 IEEE Papers</h2>
              <p className="text-slate-400 text-xs">All equations implemented in Python (src/model.py)</p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {EQUATIONS.map(eq => (
                <div key={eq.id} className="bg-slate-800/60 border border-slate-700 rounded-xl p-3 hover:border-cyan-700 transition-colors">
                  <div className="flex items-start gap-2">
                    <div className="bg-cyan-900/50 text-cyan-300 text-xs font-black rounded w-7 h-7 flex items-center justify-center flex-shrink-0">
                      {eq.id}
                    </div>
                    <div className="min-w-0">
                      <div className="text-xs font-bold text-white truncate">{eq.name}</div>
                      <div className="text-xs text-cyan-400 font-mono mt-1 break-all leading-relaxed">{eq.eq}</div>
                      <div className="text-xs text-slate-500 mt-1">{eq.ref}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── LIVE MONITOR ─────────────────────────────────── */}
        {activeTab === "live" && (
          <div className="space-y-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <MetricCard label="Surface Class" value={liveClass.split(" ")[0]} unit="" sub={liveClass} color={COLORS[liveClass] || "#fff"} />
              <MetricCard label="Friction Factor f" value={liveFriction} unit="" sub={`Risk: ${riskLevel}`} color={riskColor} />
              <MetricCard label="Temperature" value={liveTemp} unit="°C" sub="Road sensor" color="#3498DB" />
              <MetricCard label="Target Entropy H" value={liveEntropy} unit="" sub="Polarimetric radar" color="#9B59B6" />
            </div>

            <div className={`bg-slate-800/60 border rounded-xl p-5 ${alertPulse ? "border-red-600 shadow-red-900/30 shadow-lg" : "border-slate-700"}`}>
              <div className="flex items-center gap-3 mb-4">
                <div className={`w-3 h-3 rounded-full ${alertPulse ? "bg-red-500 animate-ping" : "bg-green-500"}`} />
                <h2 className="text-sm font-bold text-white">
                  {alertPulse ? "⚠️ HAZARDOUS ROAD CONDITION DETECTED" : "✅ Road Condition Nominal"}
                </h2>
                <StatusBadge level={riskLevel} />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                <div className="bg-slate-900/50 rounded-xl p-3">
                  <div className="text-slate-400 mb-1 font-bold">DETECTION METHOD</div>
                  <div className="text-white">IMTFM (Image + Meteo + Temp Fusion)</div>
                  <div className="text-slate-500 mt-1">Eq. 7: AP = 70.0% (5-class)</div>
                </div>
                <div className="bg-slate-900/50 rounded-xl p-3">
                  <div className="text-slate-400 mb-1 font-bold">RADAR SIGNAL</div>
                  <div className="text-white">H = {liveEntropy} | α = {(Math.random()*30+20).toFixed(1)}°</div>
                  <div className="text-slate-500 mt-1">Eq. 14 & 15: Polarimetric decomposition</div>
                </div>
                <div className="bg-slate-900/50 rounded-xl p-3">
                  <div className="text-slate-400 mb-1 font-bold">FRICTION ESTIMATE</div>
                  <div className="text-white">f̂ = {liveFriction} ± 0.06</div>
                  <div className="text-slate-500 mt-1">Eq. 22: 90% PI [{(liveFriction - 0.099).toFixed(2)}, {(+liveFriction + 0.099).toFixed(2)}]</div>
                </div>
              </div>
              {alertPulse && (
                <div className="mt-4 bg-red-950/50 border border-red-700 rounded-xl p-3 text-xs text-red-300">
                  🚨 ALERT: f = {liveFriction} — Road friction critically low. Transparent Ice or Mixed Ice detected.
                  Recommend: Reduce speed · Increase following distance · Deploy road treatment vehicles.
                </div>
              )}
            </div>

            <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
              <h2 className="text-sm font-bold text-cyan-300 mb-3">Friction Thresholds Reference</h2>
              <div className="space-y-2">
                {[
                  { range:"f > 0.7",    label:"Dry Road",      color:"#2ECC71", risk:"SAFE" },
                  { range:"0.45–0.70",  label:"Granular Snow", color:"#9B59B6", risk:"CAUTION" },
                  { range:"0.30–0.45",  label:"Fresh Snow",    color:"#3498DB", risk:"CAUTION" },
                  { range:"0.20–0.30",  label:"Mixed Ice",     color:"#F39C12", risk:"DANGER" },
                  { range:"f < 0.20",   label:"Transparent Ice",color:"#E74C3C",risk:"DANGER" },
                ].map(t => (
                  <div key={t.range} className="flex items-center gap-3 text-xs bg-slate-900/30 rounded-lg p-2">
                    <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: t.color }} />
                    <div className="font-mono text-slate-300 w-24">{t.range}</div>
                    <div className="text-slate-300 flex-1">{t.label}</div>
                    <StatusBadge level={t.risk} />
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-slate-800 mt-8 px-6 py-4 text-center text-xs text-slate-600">
        Climate Resilient Transportation System · IEEE Papers [1] Yang & Lei 2024 · [2] Vassilev 2024 · [3] Ojala & Seppänen 2025
        · 27 Equations · 12 Graphs · GitHub Ready
      </div>
    </div>
  );
}
