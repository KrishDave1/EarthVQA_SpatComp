import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';

interface DeltaItem {
  name: string;
  value: number;
}

interface DeltaChartProps {
  data: DeltaItem[];
  title?: string;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const val = payload[0].value;
    const isPositive = val >= 0;
    return (
      <div className="bg-[#0B1120] border border-slate-700 rounded-lg px-4 py-3 shadow-xl">
        <p className="text-slate-300 text-sm font-medium mb-1">{label}</p>
        <p className={`text-lg font-bold ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
          {isPositive ? '+' : ''}{(val * 100).toFixed(2)}%
        </p>
      </div>
    );
  }
  return null;
};

export default function DeltaChart({ data, title }: DeltaChartProps) {
  // Filter out near-zero values and sort by absolute magnitude
  const filtered = data
    .filter(d => Math.abs(d.value) > 0.0001)
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  if (filtered.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-slate-500 text-sm">
        No significant changes detected.
      </div>
    );
  }

  // Transform values to percentage points for display
  const chartData = filtered.map(d => ({
    name: d.name.replace('Δ ', '').replace(' Area %', '').replace(' %', ''),
    value: d.value,
    displayValue: d.value * 100,
    fill: d.value >= 0 ? '#34d399' : '#f87171',
  }));

  return (
    <div className="w-full">
      {title && (
        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-4">
          {title}
        </h3>
      )}
      <div className="w-full" style={{ height: Math.max(200, chartData.length * 40 + 40) }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
            <XAxis
              type="number"
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              axisLine={{ stroke: '#334155' }}
              tickLine={{ stroke: '#334155' }}
              tickFormatter={(v: number) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`}
            />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fill: '#cbd5e1', fontSize: 12, fontWeight: 500 }}
              axisLine={{ stroke: '#334155' }}
              tickLine={false}
              width={75}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(148, 163, 184, 0.05)' }} />
            <ReferenceLine x={0} stroke="#475569" strokeWidth={1.5} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={24}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} fillOpacity={0.85} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
