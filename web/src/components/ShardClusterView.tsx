import React from 'react';
import { ShardStat } from '../types';
import { Database, HardDrive, CheckCircle2, ShieldCheck, Activity } from 'lucide-react';

interface Props {
  shards: ShardStat[];
  totalVectors: number;
}

export const ShardClusterView: React.FC<Props> = ({ shards, totalVectors }) => {
  return (
    <div className="bg-[#111827]/80 backdrop-blur border border-slate-800 rounded-xl p-5 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Database className="w-5 h-5 text-indigo-400" />
          <h2 className="text-lg font-semibold text-white tracking-wide">Shard Topology & WAL Partitions</h2>
        </div>
        <div className="flex items-center space-x-2 text-xs text-slate-400">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <span>FNV-1a Consistent Hashing Active</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {shards && shards.length > 0 ? (
          shards.map((shard) => {
            const percentage = totalVectors > 0 ? ((shard.vector_count / totalVectors) * 100).toFixed(1) : '33.3';
            const memMB = (shard.memory_bytes / (1024 * 1024)).toFixed(2);

            return (
              <div
                key={shard.id}
                className="bg-[#182234] border border-slate-700/60 rounded-lg p-4 hover:border-indigo-500/50 transition-all duration-200 group relative overflow-hidden"
              >
                <div className="absolute top-0 right-0 left-0 h-1 bg-gradient-to-r from-indigo-500 to-cyan-400 opacity-80" />
                
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
                    <span className="font-mono text-sm font-bold text-white">Shard-{shard.id}</span>
                  </div>
                  <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-emerald-950/80 text-emerald-400 border border-emerald-800/40">
                    <CheckCircle2 className="w-3 h-3 mr-1" />
                    {shard.status}
                  </span>
                </div>

                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-xs text-slate-400 mb-1">
                      <span>Indexed Vectors</span>
                      <span className="font-mono font-medium text-slate-200">
                        {shard.vector_count.toLocaleString()} ({percentage}%)
                      </span>
                    </div>
                    <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                      <div
                        className="bg-indigo-500 h-1.5 rounded-full transition-all duration-500"
                        style={{ width: `${Math.min(100, Math.max(5, parseFloat(percentage)))}%` }}
                      />
                    </div>
                  </div>

                  <div className="pt-2 border-t border-slate-700/40 flex items-center justify-between text-xs text-slate-400">
                    <div className="flex items-center space-x-1.5">
                      <HardDrive className="w-3.5 h-3.5 text-slate-400" />
                      <span>Memory RSS</span>
                    </div>
                    <span className="font-mono font-medium text-slate-300">{memMB} MB</span>
                  </div>

                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <div className="flex items-center space-x-1.5">
                      <Activity className="w-3.5 h-3.5 text-indigo-400" />
                      <span>WAL Frame Store</span>
                    </div>
                    <span className="font-mono text-[11px] text-cyan-400 truncate max-w-[130px]" title={shard.wal_path}>
                      CRC32 Append-Only
                    </span>
                  </div>
                </div>
              </div>
            );
          })
        ) : (
          <div className="col-span-3 text-center py-8 text-slate-500 text-sm">
            Initializing cluster shards...
          </div>
        )}
      </div>
    </div>
  );
};
