import { Image as ImageIcon } from 'lucide-react';

const mockGallery = [
  { id: '4191', name: '4191.png', type: 'suburban', warning: 'flood risk' },
  { id: '4192', name: '4192.png', type: 'urban', warning: 'flood risk' },
  { id: '4193', name: '4193.png', type: 'suburban', warning: 'flood risk' },
  { id: '4195', name: '4195.png', type: 'urban', warning: null },
  { id: '4196', name: '4196.png', type: 'rural', warning: 'flood risk' },
  { id: '4197', name: '4197.png', type: 'rural', warning: null },
  { id: '4200', name: '4200.png', type: 'rural', warning: 'flood risk' },
  { id: '5384', name: '5384.png', type: 'urban', warning: null },
];

export default function Gallery() {
  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-700">
      <header className="mb-10">
        <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-orange-400 to-rose-400">
          Sample Data Gallery
        </h1>
        <p className="text-slate-400 mt-2 text-lg">
          Browse real EarthVQA test dataset pre-analyzed outputs. (Frontend currently uses placeholder cards for these endpoints).
        </p>
      </header>

      <div className="grid sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
        {mockGallery.map((item) => (
          <div key={item.id} className="group rounded-2xl bg-slate-800/40 backdrop-blur-md border border-slate-700/50 hover:border-slate-500 overflow-hidden cursor-pointer transition-all hover:shadow-[0_0_20px_rgba(251,146,60,0.15)] hover:-translate-y-1 block relative">
            <div className="aspect-square bg-slate-900 border-b border-slate-700/50 flex items-center justify-center relative overflow-hidden">
               {/* Decorative background grid since we don't have the images exposed statically easily right now */}
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(251,146,60,0.05)_1px,transparent_1px)] bg-[size:10px_10px]" />
              <ImageIcon size={48} className="text-slate-700/50 group-hover:text-slate-600 transition-colors" />
              <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-black/40 backdrop-blur-sm">
                <span className="px-4 py-2 bg-orange-500 text-white rounded-lg font-medium shadow-xl">Analyze</span>
              </div>
            </div>
            <div className="p-4">
              <h3 className="text-slate-200 font-bold mb-1 truncate">{item.name}</h3>
              <div className="flex flex-wrap gap-2 mt-3">
                <span className="px-2 py-0.5 rounded text-xs font-semibold uppercase tracking-wider bg-slate-700/50 text-slate-300">
                  {item.type}
                </span>
                {item.warning && (
                  <span className="px-2 py-0.5 rounded text-xs font-semibold uppercase tracking-wider bg-red-500/10 text-red-400 border border-red-500/20">
                    {item.warning}
                  </span>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
