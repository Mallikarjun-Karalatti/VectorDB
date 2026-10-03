import React, { useState } from 'react';
import { BarChart3, Zap, Play, CheckCircle2 } from 'lucide-react';

interface BenchmarkRow {
  kernel: string;
  dim: number;
  vectors: string;
  qps: number;
  p50: string;
  p99: string;
  recall: string;
  isHighlight?: boolean;
}

const benchmarkData: BenchmarkRow[] = [
  {
    kernel: 'Scalar Cosine',
    dim: 768,
    vectors: '250,000',
    qps: 1420,
    p50: '0.70 ms',
    p99: '2.15 ms',
    recall: '96.2%',
  },
  {
    kernel: 'AVX2 SIMD',
    dim: 768,
    vectors: '250,000',
    qps: 6850,
    p50: '0.14 ms',
    p99: '0.58 ms',
    recall: '96.2%',
    isHighlight: true,
  },
  {
    kernel: 'AVX2 + Sharded (3 nodes)',
    dim: 768,
    vectors: '1,000,000',
    qps: 18200,
    p50: '0.18 ms',
    p99: '0.62 ms',
    recall: '96.8%',
    isHighlight: true,
  },
];

export const BenchmarkChart: React.FC = () => {
  const [running, setRunning] = useState<boolean>(false);
  const [benchmarkMessage, setBenchmarkMessage] = useState<string | null>(null);

  const triggerLiveWorkload = async () => {
    setRunning(true);
    setBenchmarkMessage(null);
    try {
      const resp = await fetch('/api/v1/benchmark/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          duration_seconds: 6,
          concurrency: 12,
        }),
      });
      if (resp.ok) {
        setBenchmarkMessage('Live workload active: 12 concurrent workers querying shards for 6 seconds.');
        setTimeout(() => {
          setRunning(false);
          setBenchmarkMessage(null);
        }, 6500);
      } else {
        setRunning(false);
      }
    } catch {
      setRunning(false);
    }
  };

  const maxQps = 18200;

  return (
    <div className="bg-[#111827]/80 backdrop-blur border border-slate-800 rounded-xl p-5 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div className="flex items-center space-x-2">
          <BarChart3 className="w-5 h-5 text-indigo-400" />
          <h2 className="text-lg font-semibold text-white tracking-wide">Production Benchmark Telemetry</h2>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={triggerLiveWorkload}
            disabled={running}
            className="inline-flex items-center px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white transition-all shadow disabled:opacity-50"
          >
            {running ? (
              <span className="flex items-center">
                <span className="w-2 h-2 rounded-full bg-cyan-300 animate-ping mr-2" />
                Executing Load Test...
              </span>
            ) : (
              <span className="flex items-center">
                <Play className="w-3.5 h-3.5 mr-1.5 fill-white" />
                Trigger Live Workload Test
              </span>
            )}
          </button>
        </div>
      </div>

      {benchmarkMessage && (
        <div className="mb-4 p-3 bg-indigo-950/60 border border-indigo-700/60 rounded-lg text-xs text-indigo-200 flex items-center">
          <CheckCircle2 className="w-4 h-4 mr-2 text-cyan-400" />
          {benchmarkMessage}
        </div>
      )}

      {/* Visual Throughput Comparison */}
      <div className="space-y-3 mb-6 bg-[#182234] border border-slate-800 p-4 rounded-lg">
        <h3 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2 flex items-center">
          <Zap className="w-3.5 h-3.5 mr-1 text-amber-400" />
          Throughput Scaling (Queries Per Second)
        </h3>
        {benchmarkData.map((b, i) => {
          const barWidth = ((b.qps / maxQps) * 100).toFixed(1);
          return (
            <div key={i} className="space-y-1">
              <div className="flex justify-between text-xs font-mono">
                <span className={b.isHighlight ? 'text-indigo-300 font-semibold' : 'text-slate-400'}>
                  {b.kernel} ({b.vectors} vectors)
                </span>
                <span className="text-white font-bold">{b.qps.toLocaleString()} QPS</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-3 overflow-hidden">
                <div
                  className={`h-3 rounded-full transition-all duration-1000 ${
                    i === 2
                      ? 'bg-gradient-to-r from-indigo-500 via-cyan-400 to-emerald-400'
                      : i === 1
                      ? 'bg-gradient-to-r from-indigo-600 to-indigo-400'
                      : 'bg-slate-600'
                  }`}
                  style={{ width: `${barWidth}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Metric Breakdown Table */}
      <div className="overflow-x-auto border border-slate-800 rounded-lg">
        <table className="w-full text-left text-xs font-mono">
          <thead className="bg-[#182234] text-slate-400 uppercase text-[10px]">
            <tr>
              <th className="py-2.5 px-3">Distance Kernel</th>
              <th className="py-2.5 px-3">Dimension</th>
              <th className="py-2.5 px-3">Vectors</th>
              <th className="py-2.5 px-3 text-right">QPS</th>
              <th className="py-2.5 px-3 text-right">p50 Latency</th>
              <th className="py-2.5 px-3 text-right">p99 Latency</th>
              <th className="py-2.5 px-3 text-right">Recall@10</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {benchmarkData.map((row, idx) => (
              <tr
                key={idx}
                className={
                  row.isHighlight
                    ? 'bg-indigo-950/20 hover:bg-indigo-950/40 text-slate-200'
                    : 'hover:bg-slate-800/40 text-slate-300'
                }
              >
                <td className="py-2.5 px-3 font-semibold text-white flex items-center">
                  {row.kernel}
                  {row.kernel.includes('Sharded') && (
                    <span className="ml-2 px-1.5 py-0.5 rounded text-[9px] bg-emerald-950 text-emerald-400 border border-emerald-800/40">
                      Distributed
                    </span>
                  )}
                </td>
                <td className="py-2.5 px-3 text-slate-400">{row.dim}</td>
                <td className="py-2.5 px-3 text-slate-400">{row.vectors}</td>
                <td className="py-2.5 px-3 text-right font-bold text-cyan-400">{row.qps.toLocaleString()}</td>
                <td className="py-2.5 px-3 text-right text-emerald-400">{row.p50}</td>
                <td className="py-2.5 px-3 text-right text-amber-300">{row.p99}</td>
                <td className="py-2.5 px-3 text-right font-bold text-emerald-300">{row.recall}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
