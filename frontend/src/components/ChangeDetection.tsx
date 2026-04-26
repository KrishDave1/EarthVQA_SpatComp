import React, { useState, useRef } from 'react';
import {
  UploadCloud, Loader2, AlertCircle, X, Sparkles,
  ArrowRight, TrendingUp, TrendingDown, Minus, Clock,
  ShieldAlert, TreePine, Droplets, Building2, Route, CheckCircle2
} from 'lucide-react';
import axios from 'axios';
import DeltaChart from './DeltaChart';

// ─── Type Definitions ────────────────────────────────────────────────

interface DeltaFeatures {
  absolute_deltas: Record<string, number>;
  count_deltas: Record<string, number>;
  metric_deltas: Record<string, number>;
  distance_deltas: Record<string, number | null>;
  feature_vector: Array<{ name: string; value: number }>;
}

interface ChangeResult {
  sprawl_type: string;
  confidence: number;
  description: string;
  icon: string;
  classifier_type: string;
  recommendations: string[];
  delta_features: DeltaFeatures;
  features_before?: {
    area_coverage: Record<string, number>;
    object_counts: Record<string, number>;
    density: Record<string, number>;
  };
  features_after?: {
    area_coverage: Record<string, number>;
    object_counts: Record<string, number>;
    density: Record<string, number>;
  };
  colorized_mask_before_base64?: string;
  colorized_mask_after_base64?: string;
}

// ─── Legend ──────────────────────────────────────────────────────────

const LEGEND = [
  { name: 'Building', color: 'bg-[#ff0000]' },
  { name: 'Road', color: 'bg-[#808080]' },
  { name: 'Water', color: 'bg-[#0000ff]' },
  { name: 'Barren', color: 'bg-[#8b7765]' },
  { name: 'Forest', color: 'bg-[#008000]' },
  { name: 'Agriculture', color: 'bg-[#ffa500]' },
  { name: 'Playground', color: 'bg-[#00ffff]' },
  { name: 'Background', color: 'bg-black' }
];

// ─── Sprawl Type Styling ─────────────────────────────────────────────

const SPRAWL_STYLES: Record<string, { gradient: string; border: string; text: string; bg: string; icon: React.ElementType }> = {
  'Aggressive Urbanization': {
    gradient: 'from-red-500/20 to-orange-500/20',
    border: 'border-red-500/30',
    text: 'text-red-400',
    bg: 'bg-red-500/10',
    icon: Building2,
  },
  'Deforestation': {
    gradient: 'from-amber-500/20 to-yellow-500/20',
    border: 'border-amber-500/30',
    text: 'text-amber-400',
    bg: 'bg-amber-500/10',
    icon: TreePine,
  },
  'Water Encroachment': {
    gradient: 'from-blue-500/20 to-cyan-500/20',
    border: 'border-blue-500/30',
    text: 'text-blue-400',
    bg: 'bg-blue-500/10',
    icon: Droplets,
  },
  'Sustainable Expansion': {
    gradient: 'from-emerald-500/20 to-green-500/20',
    border: 'border-emerald-500/30',
    text: 'text-emerald-400',
    bg: 'bg-emerald-500/10',
    icon: CheckCircle2,
  },
  'Infrastructure Development': {
    gradient: 'from-yellow-500/20 to-amber-500/20',
    border: 'border-yellow-500/30',
    text: 'text-yellow-400',
    bg: 'bg-yellow-500/10',
    icon: Route,
  },
  'Stable / No Change': {
    gradient: 'from-slate-500/20 to-gray-500/20',
    border: 'border-slate-500/30',
    text: 'text-slate-400',
    bg: 'bg-slate-500/10',
    icon: Minus,
  },
};

// ─── Helper Components ───────────────────────────────────────────────

function UploadZone({
  label,
  sublabel,
  file,
  previewUrl,
  onFileSelect,
  id,
}: {
  label: string;
  sublabel: string;
  file: File | null;
  previewUrl: string | null;
  onFileSelect: (file: File) => void;
  id: string;
}) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files?.[0]) {
      const f = e.dataTransfer.files[0];
      if (f.type.startsWith('image/')) onFileSelect(f);
    }
  };

  return (
    <div
      id={id}
      className="relative group flex-1 min-w-0"
      onDragOver={e => e.preventDefault()}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <div className="absolute inset-0 bg-gradient-to-b from-blue-500/5 to-purple-500/5 rounded-2xl blur-lg opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
      <div className={`relative border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all duration-300 backdrop-blur-xl overflow-hidden flex flex-col items-center justify-center min-h-[220px] ${
        previewUrl
          ? 'border-blue-500/30 bg-slate-900/60'
          : 'border-slate-700 hover:border-blue-500/40 bg-slate-900/40'
      }`}>
        <input
          type="file"
          ref={inputRef}
          className="hidden"
          accept="image/*"
          onChange={e => e.target.files?.[0] && onFileSelect(e.target.files[0])}
        />
        {previewUrl ? (
          <div className="space-y-3">
            <div className="relative rounded-xl overflow-hidden shadow-lg ring-1 ring-white/10 mx-auto">
              <img src={previewUrl} alt={label} className="max-h-36 object-cover" />
              <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent" />
            </div>
            <div className="flex items-center justify-center gap-2">
              <Clock size={12} className="text-blue-400" />
              <span className="text-xs font-medium text-slate-300 truncate max-w-[160px]">{file?.name}</span>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center">
            <div className="w-14 h-14 bg-slate-800 rounded-full flex items-center justify-center mb-3 group-hover:scale-110 transition-transform border border-slate-700/50">
              <UploadCloud size={24} className="text-blue-400" />
            </div>
            <p className="text-sm font-semibold text-slate-200">{label}</p>
            <p className="text-xs text-slate-500 mt-1">{sublabel}</p>
          </div>
        )}
        <div className={`absolute top-3 left-3 px-2.5 py-1 rounded-lg text-[10px] font-bold uppercase tracking-widest border ${
          previewUrl
            ? 'bg-blue-500/10 text-blue-400 border-blue-500/20'
            : 'bg-slate-800/80 text-slate-500 border-slate-700/50'
        }`}>
          {label.split(' ')[0]}
        </div>
      </div>
    </div>
  );
}

function DeltaRow({
  label,
  before,
  after,
}: {
  label: string;
  before: number;
  after: number;
}) {
  const delta = after - before;
  const isPositive = delta > 0.001;
  const isNegative = delta < -0.001;

  return (
    <tr className="border-b border-slate-800/50 hover:bg-slate-800/20 transition-colors">
      <td className="py-2.5 px-3 text-sm font-medium text-slate-300 capitalize">{label}</td>
      <td className="py-2.5 px-3 text-sm text-slate-400 text-right font-mono">{(before * 100).toFixed(1)}%</td>
      <td className="py-2.5 px-3 text-sm text-slate-400 text-right font-mono">{(after * 100).toFixed(1)}%</td>
      <td className={`py-2.5 px-3 text-sm text-right font-mono font-semibold flex items-center justify-end gap-1.5 ${
        isPositive ? 'text-emerald-400' : isNegative ? 'text-rose-400' : 'text-slate-500'
      }`}>
        {isPositive ? <TrendingUp size={14} /> : isNegative ? <TrendingDown size={14} /> : <Minus size={14} />}
        {isPositive ? '+' : ''}{(delta * 100).toFixed(2)}%
      </td>
    </tr>
  );
}

// ─── Main Component ──────────────────────────────────────────────────

export default function ChangeDetection() {
  const [fileBefore, setFileBefore] = useState<File | null>(null);
  const [fileAfter, setFileAfter] = useState<File | null>(null);
  const [previewBefore, setPreviewBefore] = useState<string | null>(null);
  const [previewAfter, setPreviewAfter] = useState<string | null>(null);
  const [result, setResult] = useState<ChangeResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSelectBefore = (f: File) => {
    setFileBefore(f);
    setPreviewBefore(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  };

  const handleSelectAfter = (f: File) => {
    setFileAfter(f);
    setPreviewAfter(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  };

  const detectChanges = async () => {
    if (!fileBefore || !fileAfter) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image_before', fileBefore);
    formData.append('image_after', fileAfter);

    try {
      const res = await axios.post('/api/change-detect', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const data = typeof res.data === 'string' ? JSON.parse(res.data) : res.data;
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.error || 'Change detection failed.');
    } finally {
      setLoading(false);
    }
  };

  const resetAll = () => {
    setFileBefore(null);
    setFileAfter(null);
    setPreviewBefore(null);
    setPreviewAfter(null);
    setResult(null);
    setError(null);
  };

  // Prepare chart data from delta features
  const chartData = result?.delta_features?.feature_vector
    ?.filter(f => Math.abs(f.value) > 0.0001 && !f.name.includes('Distance') && !f.name.includes('Count') && !f.name.includes('Density') && !f.name.includes('Connectivity') && !f.name.includes('Intersection'))
    ?.map(f => ({ name: f.name, value: f.value })) || [];

  // Get sprawl style
  const sprawlStyle = result ? (SPRAWL_STYLES[result.sprawl_type] || SPRAWL_STYLES['Stable / No Change']) : null;
  const SprawlIcon = sprawlStyle?.icon || Minus;

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <header>
        <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-amber-400 via-orange-400 to-red-400">
          Time-Series Change Detection
        </h1>
        <p className="text-slate-400 mt-2 text-lg">
          Upload two satellite images of the same region taken at different times to detect urban sprawl patterns.
        </p>
      </header>

      {/* Upload Section */}
      {!result && !loading && (
        <section className="space-y-6">
          <div className="flex flex-col md:flex-row gap-4">
            <UploadZone
              id="upload-before"
              label="Before (T₁)"
              sublabel="Earlier time period"
              file={fileBefore}
              previewUrl={previewBefore}
              onFileSelect={handleSelectBefore}
            />
            <div className="hidden md:flex items-center justify-center px-2">
              <div className="w-10 h-10 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center">
                <ArrowRight size={18} className="text-slate-500" />
              </div>
            </div>
            <UploadZone
              id="upload-after"
              label="After (T₂)"
              sublabel="Later time period"
              file={fileAfter}
              previewUrl={previewAfter}
              onFileSelect={handleSelectAfter}
            />
          </div>

          {fileBefore && fileAfter && (
            <div className="flex justify-center">
              <button
                onClick={detectChanges}
                className="flex items-center space-x-2 px-8 py-4 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 text-white rounded-xl font-bold shadow-[0_0_20px_rgba(245,158,11,0.3)] hover:shadow-[0_0_30px_rgba(245,158,11,0.5)] hover:-translate-y-0.5 transition-all duration-200"
              >
                <Sparkles size={20} />
                <span>Detect Changes</span>
              </button>
            </div>
          )}
        </section>
      )}

      {/* Loading State */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-24 px-4 bg-slate-900/40 rounded-3xl border border-slate-800/50 backdrop-blur-xl">
          <div className="relative">
            <div className="absolute inset-0 bg-amber-500/20 rounded-full blur-xl animate-pulse" />
            <Loader2 size={64} className="text-amber-500 animate-spin relative z-10" />
          </div>
          <h3 className="text-2xl font-bold text-slate-200 mt-8 mb-2">Analyzing Temporal Changes...</h3>
          <p className="text-slate-400 animate-pulse text-center max-w-sm">
            Running segmentation on both images, extracting spatial features, computing Δ deltas, and classifying urban sprawl pattern...
          </p>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 flex items-start space-x-3 text-red-400">
          <AlertCircle className="shrink-0 mt-0.5" size={20} />
          <div>
            <h4 className="font-semibold">Change Detection Failed</h4>
            <p className="text-sm mt-1 opacity-90">{error}</p>
          </div>
          <button onClick={() => setError(null)} className="ml-auto text-red-400/60 hover:text-red-400">
            <X size={20} />
          </button>
        </div>
      )}

      {/* Results */}
      {result && sprawlStyle && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-8 duration-700 delay-150 fill-mode-backwards">

          {/* ── Classification Card ── */}
          <section className={`p-1 rounded-3xl bg-gradient-to-br ${sprawlStyle.gradient} border ${sprawlStyle.border} overflow-hidden shadow-2xl`}>
            <div className="bg-[#0B1120] rounded-[22px] p-6 md:p-8">
              <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-6">
                <div className="flex items-center gap-4">
                  <div className={`p-3 rounded-2xl ${sprawlStyle.bg} border ${sprawlStyle.border}`}>
                    <SprawlIcon size={28} className={sprawlStyle.text} />
                  </div>
                  <div>
                    <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-1">Urban Sprawl Classification</p>
                    <h2 className={`text-2xl md:text-3xl font-bold ${sprawlStyle.text}`}>
                      {result.icon} {result.sprawl_type}
                    </h2>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <div className="text-right">
                    <p className="text-xs text-slate-500 font-medium uppercase tracking-widest">Confidence</p>
                    <p className="text-3xl font-bold text-slate-100">{Math.round(result.confidence * 100)}%</p>
                  </div>
                  <button onClick={resetAll} className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm font-medium transition-colors border border-slate-700">
                    New Analysis
                  </button>
                </div>
              </div>

              <p className="text-slate-300 leading-relaxed text-sm md:text-base">
                {result.description}
              </p>

              <div className="mt-4 flex items-center gap-3 text-xs text-slate-500">
                <span className="px-2 py-1 rounded bg-slate-800 border border-slate-700/50 font-mono">
                  {result.classifier_type}
                </span>
                <span>•</span>
                <span>{result.delta_features.feature_vector?.length || 16} Δ features analyzed</span>
              </div>
            </div>
          </section>

          {/* ── Side-by-Side Masks ── */}
          <section className="p-1 rounded-3xl bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700/50 overflow-hidden shadow-2xl">
            <div className="bg-[#0B1120] rounded-[22px] p-6">
              <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2 mb-6">
                <Sparkles size={20} className="text-amber-400" /> Segmentation Comparison
              </h2>

              <div className="grid md:grid-cols-2 gap-6">
                {/* Before */}
                <div className="space-y-3">
                  <div className="relative rounded-xl overflow-hidden bg-slate-900 border border-slate-800">
                    {previewBefore && <img src={previewBefore} alt="Before original" className="w-full aspect-square object-cover" />}
                    <div className="absolute top-3 left-3 px-3 py-1.5 bg-black/60 backdrop-blur-md rounded-lg text-xs font-semibold text-white tracking-wider uppercase border border-white/10">
                      T₁ Original
                    </div>
                  </div>
                  {result.colorized_mask_before_base64 && (
                    <div className="relative rounded-xl overflow-hidden bg-slate-900 border border-slate-800">
                      <img src={`data:image/png;base64,${result.colorized_mask_before_base64}`} alt="Before mask" className="w-full aspect-square object-cover" />
                      <div className="absolute top-3 left-3 px-3 py-1.5 bg-black/60 backdrop-blur-md rounded-lg text-xs font-semibold text-white tracking-wider uppercase border border-white/10">
                        T₁ SegMask
                      </div>
                    </div>
                  )}
                </div>

                {/* After */}
                <div className="space-y-3">
                  <div className="relative rounded-xl overflow-hidden bg-slate-900 border border-slate-800">
                    {previewAfter && <img src={previewAfter} alt="After original" className="w-full aspect-square object-cover" />}
                    <div className="absolute top-3 left-3 px-3 py-1.5 bg-black/60 backdrop-blur-md rounded-lg text-xs font-semibold text-white tracking-wider uppercase border border-white/10">
                      T₂ Original
                    </div>
                  </div>
                  {result.colorized_mask_after_base64 && (
                    <div className="relative rounded-xl overflow-hidden bg-slate-900 border border-slate-800">
                      <img src={`data:image/png;base64,${result.colorized_mask_after_base64}`} alt="After mask" className="w-full aspect-square object-cover" />
                      <div className="absolute top-3 left-3 px-3 py-1.5 bg-black/60 backdrop-blur-md rounded-lg text-xs font-semibold text-white tracking-wider uppercase border border-white/10">
                        T₂ SegMask
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Legend */}
              <div className="mt-6 p-4 rounded-xl bg-slate-900/50 border border-slate-800">
                <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-3">Class Legend</h3>
                <div className="flex flex-wrap gap-4">
                  {LEGEND.map(item => (
                    <div key={item.name} className="flex items-center gap-2">
                      <div className={`w-3.5 h-3.5 rounded-sm ${item.color} shadow-sm border border-white/20`} />
                      <span className="text-sm font-medium text-slate-300">{item.name}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>

          {/* ── Delta Analysis Grid ── */}
          <section className="grid md:grid-cols-3 gap-6 items-start">
            {/* Delta Bar Chart */}
            <div className="md:col-span-2 space-y-6">
              <div className="p-6 rounded-2xl bg-slate-800/40 backdrop-blur-xl border border-slate-700/50">
                <DeltaChart data={chartData} title="Land-Use Change (Δ)" />
              </div>

              {/* Delta Table */}
              {result.features_before?.area_coverage && result.features_after?.area_coverage && (
                <div className="p-6 rounded-2xl bg-slate-800/40 backdrop-blur-xl border border-slate-700/50 overflow-x-auto">
                  <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-4">Temporal Delta Table</h3>
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-slate-700">
                        <th className="py-2 px-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">Class</th>
                        <th className="py-2 px-3 text-right text-xs font-semibold text-slate-500 uppercase tracking-wider">Before (T₁)</th>
                        <th className="py-2 px-3 text-right text-xs font-semibold text-slate-500 uppercase tracking-wider">After (T₂)</th>
                        <th className="py-2 px-3 text-right text-xs font-semibold text-slate-500 uppercase tracking-wider">Δ Change</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.keys(result.features_before.area_coverage)
                        .filter(k => k !== 'vegetation')
                        .map(key => (
                          <DeltaRow
                            key={key}
                            label={key}
                            before={result.features_before!.area_coverage[key]}
                            after={result.features_after!.area_coverage[key]}
                          />
                        ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Recommendations */}
            <div className="space-y-6">
              <div className={`p-6 rounded-2xl ${sprawlStyle.bg} border ${sprawlStyle.border} backdrop-blur-xl`}>
                <div className="flex items-center gap-2 mb-4">
                  <ShieldAlert size={18} className={sprawlStyle.text} />
                  <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-widest">AI Recommendations</h3>
                </div>
                <ul className="space-y-3">
                  {result.recommendations.map((rec, i) => (
                    <li key={i} className="flex items-start gap-2.5 text-sm text-slate-300 leading-relaxed">
                      <span className={`mt-1.5 w-1.5 h-1.5 rounded-full ${sprawlStyle.text.replace('text-', 'bg-')} shrink-0`} />
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>

              {/* Summary Stats */}
              <div className="p-5 rounded-2xl bg-slate-800/40 backdrop-blur-xl border border-slate-700/50 space-y-4">
                <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest">Key Metrics</h3>
                {result.delta_features.absolute_deltas && (
                  <div className="space-y-3">
                    {[
                      { label: 'Building Δ', value: result.delta_features.absolute_deltas.building_area_pct },
                      { label: 'Vegetation Δ', value: result.delta_features.absolute_deltas.vegetation_area_pct },
                      { label: 'Water Δ', value: result.delta_features.absolute_deltas.water_area_pct },
                      { label: 'Road Δ', value: result.delta_features.absolute_deltas.road_area_pct },
                    ].map(({ label, value }) => {
                      const isPos = value > 0.001;
                      const isNeg = value < -0.001;
                      return (
                        <div key={label} className="flex items-center justify-between">
                          <span className="text-sm text-slate-400">{label}</span>
                          <span className={`text-sm font-mono font-semibold ${isPos ? 'text-emerald-400' : isNeg ? 'text-rose-400' : 'text-slate-500'}`}>
                            {isPos ? '+' : ''}{(value * 100).toFixed(2)}%
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
