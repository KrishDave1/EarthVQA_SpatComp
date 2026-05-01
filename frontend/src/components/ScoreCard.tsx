import type { LucideIcon } from 'lucide-react';

interface ScoreCardProps {
  title: string;
  score: number; // 0 to 1
  label: string; // e.g. "Needs Improvement"
  severity: string;
  icon: LucideIcon;
}

export default function ScoreCard({ title, score, label, severity, icon: Icon }: ScoreCardProps) {
  // Determine color based on severity
  let color = 'text-green-400';
  let bg = 'bg-green-400/10';
  let border = 'border-green-400/20';
  let dot = 'bg-green-400';

  // Red: bad states
  if (['high', 'critical', 'insufficient', 'poor'].includes(severity)) {
    color = 'text-red-400';
    bg = 'bg-red-400/10';
    border = 'border-red-400/20';
    dot = 'bg-red-400';
  // Yellow: moderate / in-between states
  } else if (['moderate', 'adequate'].includes(severity)) {
    color = 'text-yellow-400';
    bg = 'bg-yellow-400/10';
    border = 'border-yellow-400/20';
    dot = 'bg-yellow-400';
  // Green: good states (including 'low' for density/flood which means low concern)
  } else if (['good', 'low'].includes(severity)) {
    color = 'text-green-400';
    bg = 'bg-green-400/10';
    border = 'border-green-400/20';
    dot = 'bg-green-400';
  }

  // Convert 0-1 score to percentage
  const percent = Math.round(score * 100);

  return (
    <div className={`p-5 rounded-2xl bg-slate-800/40 backdrop-blur-xl border border-slate-700/50 hover:border-slate-600 transition-all duration-300 relative overflow-hidden group`}>
      <div className={`absolute top-0 right-0 w-32 h-32 ${bg} rounded-full blur-[50px] -mr-10 -mt-10 pointer-events-none transition-opacity opacity-50 group-hover:opacity-100`} />
      
      <div className="flex items-start justify-between relative z-10">
        <div className="flex items-center space-x-3">
          <div className={`p-2.5 rounded-xl ${bg} ${border} border`}>
            <Icon size={20} className={color} />
          </div>
          <div>
            <h3 className="text-slate-300 font-medium">{title}</h3>
            <div className="flex items-center space-x-2 mt-1">
              <span className={`w-2 h-2 rounded-full ${dot} shadow-[0_0_8px_currentColor]`} />
              <span className="text-sm font-semibold tracking-wide uppercase text-slate-400">{label}</span>
            </div>
          </div>
        </div>
        
        <div className="text-right">
          <div className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-100 to-slate-400">
            {percent}
            <span className="text-sm text-slate-500 font-medium ml-1">/ 100</span>
          </div>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="w-full h-1.5 bg-slate-700/50 rounded-full mt-5 overflow-hidden">
        <div 
          className={`h-full rounded-full ${dot} transition-all duration-1000 ease-out`} 
          style={{ width: `${percent}%` }}
        />
      </div>

    </div>
  );
}
