import React, { useState, useRef } from 'react';
import { UploadCloud, Locate, Trees, Waves, Route, Sparkles, AlertCircle, Loader2, ArrowRight, X } from 'lucide-react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from 'recharts';
import ScoreCard from './ScoreCard';
import axios from 'axios';

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

interface PlanningReport {
  overall_score: number;
  overall_suitability: string;
  scene_type: string;
  decisions: Array<{
    category: string;
    score: number;
    severity: string;
    recommendation: string;
  }>;
  summary: string;
}

export default function Dashboard() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [report, setReport] = useState<PlanningReport | null>(null);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelection = (file: File) => {
    if (file.type.startsWith('image/')) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setReport(null);
      setMaskUrl(null);
      setError(null);
    } else {
      setError("Please select a valid image file.");
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;
    
    setLoading(true);
    setError(null);
    
    // We will simulate the multipart upload
    const formData = new FormData();
    formData.append('image', selectedImage);
    
    try {
      const analyzeRes = await axios.post('/api/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      console.log("Analysis Success:", analyzeRes.data);
      if (analyzeRes.data.planning_report) {
         setReport(analyzeRes.data.planning_report);
      }
      if (analyzeRes.data.colorized_mask_base64) {
        setMaskUrl(`data:image/png;base64,${analyzeRes.data.colorized_mask_base64}`);
      }
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.error || "An error occurred during analysis.");
    } finally {
      setLoading(false);
    }
  };

  // Map category to icon
  const getIcon = (category: string) => {
    const cat = category.toLowerCase();
    if (cat.includes('density')) return Locate;
    if (cat.includes('green') || cat.includes('vegetation')) return Trees;
    if (cat.includes('flood') || cat.includes('water')) return Waves;
    if (cat.includes('infra') || cat.includes('road')) return Route;
    return Locate;
  };

  // Prepare Radar Data
  const radarData = report?.decisions?.map(d => ({
    subject: d.category,
    A: Math.round(d.score * 100),
    fullMark: 100,
  })) || [];

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <header>
        <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-400">
          Smart City Planning Dashboard
        </h1>
        <p className="text-slate-400 mt-2 text-lg">
          Upload satellite imagery for AI-driven semantic segmentation and automated spatial analysis.
        </p>
      </header>

      {!report && !loading && (
        <section className="relative group">
          <div className="absolute inset-0 bg-gradient-to-b from-blue-500/10 to-purple-500/10 rounded-3xl blur-xl opacity-50 group-hover:opacity-100 transition-opacity duration-500" />
          <div 
            className="relative border-2 border-dashed border-slate-700 hover:border-blue-500/50 rounded-3xl p-12 text-center bg-slate-900/50 backdrop-blur-xl transition-all duration-300 cursor-pointer overflow-hidden flex flex-col items-center justify-center min-h-[400px]"
            onDragOver={e => e.preventDefault()}
            onDrop={handleFileDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              className="hidden" 
              accept="image/*"
              onChange={(e) => e.target.files && handleFileSelection(e.target.files[0])}
            />
            
            {previewUrl ? (
              <div className="relative z-10 space-y-6">
                <div className="relative inline-block rounded-2xl overflow-hidden shadow-2xl ring-1 ring-white/10">
                  <img src={previewUrl} alt="Preview" className="max-h-64 object-cover" />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent flex items-end justify-center pb-4">
                    <span className="text-sm font-medium text-white bg-black/40 px-3 py-1 rounded-full backdrop-blur-md">
                      {selectedImage?.name}
                    </span>
                  </div>
                </div>
                <button 
                  onClick={(e) => { e.stopPropagation(); analyzeImage(); }}
                  className="mx-auto flex items-center space-x-2 px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white rounded-xl font-bold shadow-[0_0_20px_rgba(79,70,229,0.4)] hover:shadow-[0_0_30px_rgba(79,70,229,0.6)] hover:-translate-y-0.5 transition-all duration-200"
                >
                  <Sparkles size={20} />
                  <span>Run Analysis (28.5M Params)</span>
                </button>
              </div>
            ) : (
              <div className="relative z-10 flex flex-col items-center">
                <div className="w-20 h-20 bg-slate-800 rounded-full flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300 border border-slate-700/50 shadow-inner">
                  <UploadCloud size={32} className="text-blue-400" />
                </div>
                <h3 className="text-xl font-semibold text-slate-200 mb-2">Drag & Drop Satellite Image</h3>
                <p className="text-slate-400 max-w-md">
                  Upload a 512x512 PNG. The SemanticFPN model will automatically extract 8-class segmentation features.
                </p>
                <div className="mt-8 flex items-center space-x-4">
                  <span className="px-4 py-2 rounded-lg bg-slate-800/50 text-slate-300 text-sm border border-slate-700/50 font-medium">Browse Files</span>
                </div>
              </div>
            )}
          </div>
        </section>
      )}

      {loading && (
        <div className="flex flex-col items-center justify-center py-24 px-4 bg-slate-900/40 rounded-3xl border border-slate-800/50 backdrop-blur-xl">
          <div className="relative">
            <div className="absolute inset-0 bg-blue-500/20 rounded-full blur-xl animate-pulse" />
            <Loader2 size={64} className="text-blue-500 animate-spin relative z-10" />
          </div>
          <h3 className="text-2xl font-bold text-slate-200 mt-8 mb-2">Analyzing Scene Context...</h3>
          <p className="text-slate-400 animate-pulse text-center max-w-sm">
            SemanticFPN is segmenting 262,144 pixels. Spatial Engine is calculating densities and distances...
          </p>
        </div>
      )}

      {error && (
        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 flex items-start space-x-3 text-red-400">
          <AlertCircle className="shrink-0 mt-0.5" size={20} />
          <div>
            <h4 className="font-semibold">Analysis Failed</h4>
            <p className="text-sm mt-1 opacity-90">{error}</p>
          </div>
          <button onClick={() => setError(null)} className="ml-auto text-red-400/60 hover:text-red-400">
            <X size={20} />
          </button>
        </div>
      )}

      {report && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-8 duration-700 delay-150 fill-mode-backwards">
          {/* Main Visualizer */}
          <section className="p-1 rounded-3xl bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700/50 overflow-hidden shadow-2xl">
           <div className="bg-[#0B1120] rounded-[22px] p-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
                  <Sparkles size={20} className="text-blue-400" /> Model Output
                </h2>
                <div className="flex items-center gap-3 mt-2 text-sm">
                  <span className="px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700/50 capitalize font-medium flex items-center gap-1.5">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                    {report.scene_type} Environment
                  </span>
                  <span className="text-slate-500 flex items-center gap-1.5">
                    <ArrowRight size={14} /> AI SegMask
                  </span>
                </div>
              </div>
              <button onClick={() => setReport(null)} className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm font-medium transition-colors border border-slate-700">
                New Analysis
              </button>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="relative group rounded-xl overflow-hidden bg-slate-900 border border-slate-800">
                <img src={previewUrl!} alt="Original" className="w-full aspect-square object-cover" />
                <div className="absolute top-3 left-3 px-3 py-1.5 bg-black/60 backdrop-blur-md rounded-lg text-xs font-semibold text-white tracking-wider uppercase border border-white/10">Original</div>
              </div>
              <div className="relative group rounded-xl overflow-hidden bg-slate-900 border border-slate-800">
                {maskUrl ? (
                  <img src={maskUrl} alt="Segmentation Mask" className="w-full aspect-square object-cover" />
                ) : (
                  <div className="w-full aspect-square flex items-center justify-center text-slate-500">Processing Mask...</div>
                )}
                <div className="absolute top-3 left-3 px-3 py-1.5 bg-black/60 backdrop-blur-md rounded-lg text-xs font-semibold text-white tracking-wider uppercase border border-white/10">8-Class SegMask</div>
              </div>
            </div>

            {/* Added Legend Section Here */}
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

          {/* Assessment Grid */}
          <section className="grid md:grid-cols-3 gap-6 items-start">
            <div className="md:col-span-2 space-y-6">
              <h2 className="text-xl font-bold text-slate-100 px-1">Domain Assessments</h2>
              <div className="grid sm:grid-cols-2 gap-4">
                {report.decisions.map((d, i) => (
                  <ScoreCard 
                    key={i}
                    title={d.category}
                    score={d.score}
                    label={d.severity}
                    severity={d.severity as any}
                    recommendation={d.recommendation}
                    icon={getIcon(d.category)}
                  />
                ))}
              </div>
            </div>

            <div className="space-y-6">
              <h2 className="text-xl font-bold text-slate-100 px-1">Suitability Profile</h2>
              <div className="p-6 rounded-2xl bg-slate-800/40 backdrop-blur-xl border border-slate-700/50 flex flex-col items-center">
                <div className="w-full h-64 mb-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
                      <PolarGrid stroke="#334155" />
                      <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 12, fontWeight: 500 }} />
                      <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                      <Radar name="Score" dataKey="A" stroke="#3b82f6" strokeWidth={2} fill="#3b82f6" fillOpacity={0.3} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#0B1120', borderColor: '#334155', borderRadius: '8px' }}
                        itemStyle={{ color: '#f8fafc' }}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
                
                <div className="text-center w-full pt-4 border-t border-slate-700/60">
                  <p className="text-sm text-slate-400 mb-1 font-medium uppercase tracking-widest">Overall Verdict</p>
                  <p className="text-2xl font-bold text-slate-100">{report.overall_suitability}</p>
                  <div className="mt-3 flex items-center justify-center space-x-2">
                    <span className="text-3xl font-bold text-blue-400">{Math.round(report.overall_score * 100)}</span>
                    <span className="text-slate-500 font-medium">/ 100</span>
                  </div>
                </div>
              </div>
              
              <div className="p-5 rounded-2xl bg-blue-500/5 border border-blue-500/20 text-slate-300 text-sm leading-relaxed">
                <p className="font-semibold text-blue-400 mb-2 flex items-center"><Sparkles size={16} className="mr-1.5"/> AI Summary</p>
                {report.summary}
              </div>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
