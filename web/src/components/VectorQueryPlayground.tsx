import React, { useState } from 'react';
import { Search, Sparkles, Sliders, Clock, Layers, PlusCircle, CheckCircle2 } from 'lucide-react';
import { SearchResponse } from '../types';

interface Props {
  dimension: number;
}

export const VectorQueryPlayground: React.FC<Props> = ({ dimension = 128 }) => {
  const [k, setK] = useState<number>(10);
  const [efSearch, setEfSearch] = useState<number>(64);
  const [vectorInput, setVectorInput] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Ingestion state
  const [newId, setNewId] = useState<string>('');
  const [insertSuccess, setInsertSuccess] = useState<string | null>(null);

  const generateRandomVector = () => {
    const vec = Array.from({ length: dimension }, () => Number((Math.random() * 2 - 1).toFixed(4)));
    setVectorInput(JSON.stringify(vec));
    setError(null);
  };

  const handleSearch = async () => {
    let query: number[] = [];
    try {
      if (!vectorInput.trim()) {
        const defaultVec = Array.from({ length: dimension }, () => Number((Math.random() * 2 - 1).toFixed(4)));
        query = defaultVec;
        setVectorInput(JSON.stringify(defaultVec));
      } else {
        query = JSON.parse(vectorInput);
      }

      if (!Array.isArray(query) || query.length !== dimension) {
        throw new Error(`Vector must have exactly ${dimension} elements (got ${query.length})`);
      }
    } catch (e: any) {
      setError(e.message || 'Invalid vector format. Must be a JSON array of floats.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const resp = await fetch('/api/v1/collections/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          k,
          ef_search: efSearch,
        }),
      });

      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || 'Search request failed');
      }

      const data: SearchResponse = await resp.json();
      setResults(data);
    } catch (err: any) {
      setError(err.message || 'Failed to execute KNN search');
    } finally {
      setLoading(false);
    }
  };

  const handleInsert = async () => {
    let vec: number[] = [];
    try {
      if (!vectorInput.trim()) {
        throw new Error('Vector coordinates cannot be empty');
      }
      vec = JSON.parse(vectorInput);
      if (!Array.isArray(vec) || vec.length !== dimension) {
        throw new Error(`Vector must have exactly ${dimension} elements`);
      }
      const idNum = parseInt(newId, 10);
      if (isNaN(idNum) || idNum <= 0) {
        throw new Error('Please enter a valid positive numerical ID');
      }

      const resp = await fetch('/api/v1/collections/vectors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: idNum,
          vector: vec,
        }),
      });

      if (!resp.ok) {
        throw new Error(await resp.text());
      }

      const res = await resp.json();
      setInsertSuccess(`Indexed vector ${idNum} into Shard-${res.shard_id} with WAL durability.`);
      setTimeout(() => setInsertSuccess(null), 5000);
    } catch (err: any) {
      setError(err.message);
    }
  };

  return (
    <div className="bg-[#111827]/80 backdrop-blur border border-slate-800 rounded-xl p-5 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Search className="w-5 h-5 text-cyan-400" />
          <h2 className="text-lg font-semibold text-white tracking-wide">Interactive Vector Search Playground</h2>
        </div>
        <button
          onClick={generateRandomVector}
          className="inline-flex items-center px-3 py-1.5 rounded-lg text-xs font-medium bg-indigo-600/30 text-indigo-300 hover:bg-indigo-600/50 border border-indigo-500/40 transition-colors"
        >
          <Sparkles className="w-3.5 h-3.5 mr-1 text-indigo-400" />
          Generate Synthetic Vector ({dimension}D)
        </button>
      </div>

      <div className="space-y-4">
        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-[#182234] border border-slate-800 p-4 rounded-lg">
          <div>
            <div className="flex justify-between text-xs text-slate-300 mb-1">
              <span className="flex items-center">
                <Sliders className="w-3.5 h-3.5 mr-1 text-cyan-400" />
                Nearest Neighbors (K)
              </span>
              <span className="font-mono font-bold text-white">{k}</span>
            </div>
            <input
              type="range"
              min="1"
              max="50"
              value={k}
              onChange={(e) => setK(parseInt(e.target.value, 10))}
              className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
            />
          </div>

          <div>
            <div className="flex justify-between text-xs text-slate-300 mb-1">
              <span className="flex items-center">
                <Sliders className="w-3.5 h-3.5 mr-1 text-indigo-400" />
                Search Exploration Factor (efSearch)
              </span>
              <span className="font-mono font-bold text-white">{efSearch}</span>
            </div>
            <input
              type="range"
              min="16"
              max="256"
              step="8"
              value={efSearch}
              onChange={(e) => setEfSearch(parseInt(e.target.value, 10))}
              className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-400"
            />
          </div>
        </div>

        {/* Vector Textarea */}
        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1">
            Query Coordinates JSON Array (or leave empty to auto-generate)
          </label>
          <textarea
            value={vectorInput}
            onChange={(e) => setVectorInput(e.target.value)}
            placeholder={`[0.1234, -0.5678, ... ${dimension} float coordinates]`}
            className="w-full h-20 bg-[#0b0f19] border border-slate-700 rounded-lg p-2.5 font-mono text-xs text-slate-200 focus:outline-none focus:border-indigo-500 resize-none"
          />
        </div>

        {error && (
          <div className="p-3 bg-rose-950/40 border border-rose-800 text-rose-300 text-xs rounded-lg">
            {error}
          </div>
        )}

        {insertSuccess && (
          <div className="p-3 bg-emerald-950/40 border border-emerald-800 text-emerald-300 text-xs rounded-lg flex items-center">
            <CheckCircle2 className="w-4 h-4 mr-1.5 text-emerald-400" />
            {insertSuccess}
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
          <button
            onClick={handleSearch}
            disabled={loading}
            className="flex-1 min-w-[200px] inline-flex items-center justify-center px-4 py-2.5 rounded-lg text-sm font-semibold bg-gradient-to-r from-indigo-500 to-cyan-500 text-white hover:from-indigo-600 hover:to-cyan-600 shadow-md transition-all disabled:opacity-50"
          >
            {loading ? (
              <span className="flex items-center">Searching HNSW Graph...</span>
            ) : (
              <span className="flex items-center">
                <Search className="w-4 h-4 mr-2" />
                Execute Parallel KNN Search
              </span>
            )}
          </button>

          <div className="flex items-center space-x-2">
            <input
              type="number"
              placeholder="Vector ID"
              value={newId}
              onChange={(e) => setNewId(e.target.value)}
              className="w-28 bg-[#0b0f19] border border-slate-700 rounded-lg px-2.5 py-2 text-xs font-mono text-white focus:outline-none focus:border-emerald-500"
            />
            <button
              onClick={handleInsert}
              className="inline-flex items-center px-3 py-2 rounded-lg text-xs font-medium bg-emerald-600/30 text-emerald-300 hover:bg-emerald-600/50 border border-emerald-500/40 transition-colors"
            >
              <PlusCircle className="w-3.5 h-3.5 mr-1" />
              Ingest & Index
            </button>
          </div>
        </div>

        {/* Results Panel */}
        {results && (
          <div className="mt-4 border-t border-slate-800 pt-4">
            <div className="flex flex-wrap items-center justify-between text-xs text-slate-400 mb-3 bg-[#182234] p-3 rounded-lg border border-slate-700/60">
              <div className="flex items-center space-x-2">
                <Clock className="w-4 h-4 text-cyan-400" />
                <span>Search Latency:</span>
                <span className="font-mono font-bold text-emerald-400">{results.took_ms.toFixed(3)} ms</span>
              </div>
              <div className="flex items-center space-x-2">
                <Layers className="w-4 h-4 text-indigo-400" />
                <span>Global Matches:</span>
                <span className="font-mono font-bold text-white">{results.matches.length}</span>
              </div>
              <div className="text-slate-400">
                <span>Top-K Priority Queue Merged across 3 local shards</span>
              </div>
            </div>

            <div className="overflow-x-auto max-h-60 overflow-y-auto border border-slate-800 rounded-lg">
              <table className="w-full text-left text-xs">
                <thead className="bg-[#182234] text-slate-400 uppercase font-mono text-[10px] sticky top-0">
                  <tr>
                    <th className="py-2.5 px-3">Rank</th>
                    <th className="py-2.5 px-3">Vector ID</th>
                    <th className="py-2.5 px-3">Cosine Distance</th>
                    <th className="py-2.5 px-3">Answering Shard</th>
                    <th className="py-2.5 px-3">Similarity</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 font-mono">
                  {results.matches.map((match, idx) => {
                    const similarity = Math.max(0, 1.0 - match.distance);
                    return (
                      <tr key={idx} className="hover:bg-slate-800/40 transition-colors">
                        <td className="py-2 px-3 text-slate-400">#{idx + 1}</td>
                        <td className="py-2 px-3 font-semibold text-white">{match.id}</td>
                        <td className="py-2 px-3 text-cyan-300">{match.distance.toFixed(5)}</td>
                        <td className="py-2 px-3">
                          <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-indigo-950 text-indigo-300 border border-indigo-800/50">
                            Shard-{match.shard_id ?? 0}
                          </span>
                        </td>
                        <td className="py-2 px-3">
                          <div className="flex items-center space-x-2">
                            <div className="w-16 bg-slate-800 h-1.5 rounded-full overflow-hidden">
                              <div
                                className="bg-gradient-to-r from-cyan-400 to-indigo-500 h-1.5 rounded-full"
                                style={{ width: `${Math.min(100, similarity * 100)}%` }}
                              />
                            </div>
                            <span className="text-[11px] text-slate-400">{(similarity * 100).toFixed(1)}%</span>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
