import React from 'react';
import StrategyToggleCard from '@/components/strategy-toggle-card';
import SignalFeed from '@/components/signal-feed';
import WeightsDashboard from '@/components/weights-dashboard';
import LiveHoldingsPanel from '@/components/live-holdings-panel';
import { useToggle } from '@/data-hooks/useToggle';
import { useSignals } from '@/data-hooks/useSignals';
import { useHoldings } from '@/data-hooks/useHoldings';
import { usePnL } from '@/data-hooks/usePnL';

const Dashboard = () => {
  const { toggles, updateToggle, loading: toggleLoading } = useToggle();
  const { signals, loading: signalLoading } = useSignals();
  const { holdings, loading: holdingsLoading } = useHoldings();
  const { pnlData, loading: pnlLoading } = usePnL();

  const strategyList = Object.keys(toggles || {});

  return (
    <div className="min-h-screen p-6 bg-gray-50 text-gray-800">
      <h1 className="text-3xl font-bold mb-6">Strategy Control Center</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 mb-8">
        {strategyList.map((strategy) => (
          <StrategyToggleCard
            key={strategy}
            name={strategy}
            enabled={toggles[strategy]}
            onToggle={(val) => updateToggle(strategy, val)}
            pnl={pnlData[strategy]}
            loading={toggleLoading || pnlLoading}
          />
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="bg-white p-4 rounded-xl shadow">
          <h2 className="text-xl font-semibold mb-4">📈 Signal Feed</h2>
          <SignalFeed signals={signals} loading={signalLoading} />
        </div>

        <div className="bg-white p-4 rounded-xl shadow">
          <h2 className="text-xl font-semibold mb-4">⚖️ Weight Allocation</h2>
          <WeightsDashboard />
        </div>
      </div>

      <div className="mt-10 bg-white p-4 rounded-xl shadow">
        <h2 className="text-xl font-semibold mb-4">💼 Live Holdings Panel</h2>
        <LiveHoldingsPanel holdings={holdings} loading={holdingsLoading} />
      </div>
    </div>
  );
};

export default Dashboard;