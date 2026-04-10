import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, Send, Loader2, Bot, User, Image as ImageIcon, CheckCircle2 } from 'lucide-react';
import axios from 'axios';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  intent?: string;
  confidence?: number;
}

export default function VQA() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([
    { role: 'assistant', content: "Upload a satellite image, then ask me questions about it! For example: 'Is there flood risk?' or 'What are the land use types?'" }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [imageUploaded, setImageUploaded] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleImageSelection = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setImageUploaded(true);
      setMessages([
        { role: 'assistant', content: `Image '${file.name}' loaded. What would you like to know about this scene?` }
      ]);
    }
  };

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !selectedImage) return;

    const userQ = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userQ }]);
    setLoading(true);

    const formData = new FormData();
    formData.append('image', selectedImage);
    formData.append('question', userQ);

    try {
      const res = await axios.post('/api/ask', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      const responseData = typeof res.data === 'string' 
        ? JSON.parse(res.data) 
        : res.data;
        
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: responseData.answer,
        intent: responseData.intent?.type,
        confidence: responseData.confidence
      }]);
    } catch (err: any) {
      console.error(err);
      setMessages(prev => [...prev, { role: 'assistant', content: "Sorry, I couldn't process that question. Ensure the backend is running." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-700">
      <header className="mb-8">
        <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400">
          Visual Question Answering
        </h1>
        <p className="text-slate-400 mt-2 text-lg">
          Query spatial intelligence directly via natural language. Uses EarthVQA SOBA + spatial logic.
        </p>
      </header>

      <div className="grid md:grid-cols-3 gap-6 h-[700px]">
        {/* Left Col: Image Upload & Preview */}
        <div className="h-full flex flex-col space-y-4">
          <div className="flex-1 rounded-2xl bg-slate-900/50 backdrop-blur-xl border border-slate-700/50 flex flex-col overflow-hidden relative">
            <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/10 rounded-full blur-[40px] pointer-events-none" />
            
            {!imageUploaded ? (
              <label className="flex-1 flex flex-col items-center justify-center p-6 cursor-pointer hover:bg-slate-800/30 transition-colors group">
                <input type="file" className="hidden" accept="image/*" onChange={handleImageSelection} />
                <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <UploadCloud size={24} className="text-emerald-400" />
                </div>
                <p className="font-medium text-slate-300">Upload Image to Chat</p>
                <p className="text-xs text-slate-500 mt-2 text-center">PNG required for segmentation</p>
              </label>
            ) : (
              <div className="flex-1 flex flex-col">
                <div className="p-4 border-b border-slate-800/60 bg-slate-900/80 flex items-center justify-between">
                  <div className="flex items-center space-x-2 text-emerald-400 text-sm font-medium">
                    <ImageIcon size={16} />
                    <span className="truncate max-w-[120px] text-slate-300">{selectedImage?.name}</span>
                  </div>
                  <label className="cursor-pointer text-xs font-semibold text-slate-400 hover:text-slate-200 uppercase tracking-widest px-2 py-1 rounded bg-slate-800">
                    <input type="file" className="hidden" accept="image/*" onChange={handleImageSelection} />
                    Change
                  </label>
                </div>
                <div className="flex-1 p-2 flex items-center justify-center">
                  <img src={previewUrl!} alt="Selected Target" className="max-h-full max-w-full rounded-xl object-contain border border-slate-700/50 shadow-2xl" />
                </div>
              </div>
            )}
          </div>

          {imageUploaded && (
            <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-sm text-emerald-300 flex items-center gap-3">
              <CheckCircle2 size={18} className="shrink-0" />
              <span>Image primed. SemanticFPN & SOBA ready.</span>
            </div>
          )}
        </div>

        {/* Right Col: Chat Interface */}
        <div className="md:col-span-2 h-full flex flex-col rounded-2xl bg-slate-800/40 backdrop-blur-xl border border-slate-700/50 shadow-2xl overflow-hidden relative">
          {/* Output History */}
          <div className="flex-1 p-6 overflow-y-auto space-y-6 scroll-smooth">
            {messages.map((msg, idx) => (
              <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`flex max-w-[85%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                  
                  <div className={`shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                    msg.role === 'user' ? 'ml-3 bg-blue-600' : 'mr-3 bg-teal-600'
                  }`}>
                    {msg.role === 'user' ? <User size={16} className="text-white"/> : <Bot size={16} className="text-white"/>}
                  </div>
                  
                  <div className={`rounded-2xl p-4 ${
                    msg.role === 'user' 
                      ? 'bg-blue-600 text-white rounded-tr-sm shadow-[0_4px_15px_rgba(37,99,235,0.2)]' 
                      : 'bg-slate-700/60 border border-slate-600/50 text-slate-200 rounded-tl-sm shadow-xl'
                  }`}>
                    <p className="leading-relaxed">{msg.content}</p>
                    {msg.intent && (
                      <div className="mt-3 pt-3 border-t border-slate-600/50 flex flex-wrap gap-2 text-xs font-mono">
                        <span className="px-2 py-0.5 rounded bg-black/30 text-teal-300">intent: {msg.intent}</span>
                        {msg.confidence && (
                          <span className="px-2 py-0.5 rounded bg-black/30 text-emerald-300">conf: {msg.confidence.toFixed(2)}</span>
                        )}
                      </div>
                    )}
                  </div>
                  
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="flex flex-row max-w-[85%]">
                  <div className="shrink-0 w-8 h-8 rounded-full mr-3 bg-teal-600 flex items-center justify-center">
                    <Bot size={16} className="text-white"/>
                  </div>
                  <div className="rounded-2xl rounded-tl-sm p-4 bg-slate-700/60 border border-slate-600/50 flex items-center space-x-2">
                    <Loader2 size={16} className="text-teal-400 animate-spin" />
                    <span className="text-slate-400 text-sm">Reasoning spatially...</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Box */}
          <div className="p-4 bg-slate-900/60 border-t border-slate-700/50 relative z-10">
            <form onSubmit={handleSend} className="relative flex items-center">
              <input
                type="text"
                disabled={!imageUploaded || loading}
                value={input}
                onChange={e => setInput(e.target.value)}
                placeholder={imageUploaded ? "Ask a spatial reasoning question..." : "Upload an image first..."}
                className="w-full bg-slate-800/80 border border-slate-600 text-slate-200 rounded-xl px-5 py-4 pr-14 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent placeholder:text-slate-500 transition-shadow shadow-inner disabled:opacity-50 disabled:cursor-not-allowed"
              />
              <button 
                type="submit" 
                disabled={!imageUploaded || loading || !input.trim()}
                className="absolute right-2 p-2 bg-teal-500 hover:bg-teal-400 text-slate-900 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Send size={20} className="ml-0.5" />
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
