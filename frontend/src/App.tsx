import { useState } from 'react';
import { LayoutDashboard, MessageSquare, Image as ImageIcon, Menu, X, Satellite } from 'lucide-react';
import Dashboard from './components/Dashboard';
import VQA from './components/VQA';
import Gallery from './components/Gallery';

export type TView = 'dashboard' | 'vqa' | 'gallery';

function App() {
  const [activeView, setActiveView] = useState<TView>('dashboard');
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const NavItem = ({ view, icon: Icon, label }: { view: TView, icon: any, label: string }) => (
    <button
      onClick={() => {
        setActiveView(view);
        setIsMobileMenuOpen(false);
      }}
      className={`w-full flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-200 ${
        activeView === view 
          ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30 shadow-[0_0_15px_rgba(59,130,246,0.15)]' 
          : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200 border border-transparent'
      }`}
    >
      <Icon size={20} />
      <span className="font-medium">{label}</span>
    </button>
  );

  return (
    <div className="min-h-screen bg-[#0B1120] flex selection:bg-blue-500/30">
      {/* Mobile Header */}
      <div className="md:hidden fixed top-0 left-0 right-0 h-16 bg-[#0B1120]/80 backdrop-blur-md z-50 border-b border-slate-800 flex items-center justify-between px-4">
        <div className="flex items-center space-x-2 text-blue-400">
          <Satellite size={24} />
          <span className="font-bold text-lg text-slate-100 tracking-wide">EarthVQA</span>
        </div>
        <button onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)} className="text-slate-300">
          {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Sidebar Navigation */}
      <aside className={`
        fixed md:sticky top-0 left-0 z-40 h-screen w-64 bg-slate-900/40 backdrop-blur-xl border-r border-slate-800/60
        transform transition-transform duration-300 ease-in-out
        ${isMobileMenuOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
      `}>
        <div className="p-6 h-full flex flex-col">
          <div className="hidden md:flex items-center space-x-3 mb-12 text-blue-400">
            <div className="p-2 bg-blue-500/10 rounded-xl border border-blue-500/20">
              <Satellite size={28} />
            </div>
            <div>
              <h1 className="font-bold text-xl text-slate-100 tracking-wide">EarthVQA</h1>
              <p className="text-xs text-blue-400/80 uppercase tracking-widest font-semibold mt-0.5">Smart City</p>
            </div>
          </div>

          <nav className="flex-1 space-y-2 mt-16 md:mt-0">
            <NavItem view="dashboard" icon={LayoutDashboard} label="Planning Dashboard" />
            <NavItem view="vqa" icon={MessageSquare} label="VQA Assistant" />
            <NavItem view="gallery" icon={ImageIcon} label="Sample Gallery" />
          </nav>

          <div className="mt-auto">
            <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50 backdrop-blur-sm">
              <p className="text-xs text-slate-400 mb-2">GPU Status</p>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-green-400 shadow-[0_0_8px_rgba(74,222,128,0.8)] animate-pulse" />
                <span className="text-sm font-medium text-slate-200">Connected (API)</span>
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 min-w-0 p-4 pt-20 md:p-8 md:pt-8 bg-[#0B1120] relative">
        <div className="absolute top-0 right-0 w-96 h-96 bg-blue-500/5 rounded-full blur-[100px] pointer-events-none" />
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-purple-500/5 rounded-full blur-[100px] pointer-events-none" />
        
        <div className="max-w-7xl mx-auto relative z-10">
          {activeView === 'dashboard' && <Dashboard />}
          {activeView === 'vqa' && <VQA />}
          {activeView === 'gallery' && <Gallery />}
        </div>
      </main>
    </div>
  );
}

export default App;
