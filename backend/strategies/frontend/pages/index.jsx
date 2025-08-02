import React from 'react';
import Link from 'next/link';

const Home = () => {
  return (
    <main className="flex flex-col items-center justify-center min-h-screen px-4 bg-gradient-to-br from-slate-100 via-white to-slate-200 text-gray-800">
      <div className="text-center max-w-2xl">
        <h1 className="text-5xl font-bold mb-6">📊 Macro Arbitrage Strategy Hub</h1>
        <p className="text-lg mb-8">
          Run over 87+ alpha & diversified strategies. Track P&L, signals, weights, and holdings—all live and in one place.
        </p>

        <Link href="/dashboard">
          <a className="inline-block px-6 py-3 text-white text-lg bg-blue-600 rounded-xl hover:bg-blue-700 transition-all shadow-lg">
            Go to Dashboard →
          </a>
        </Link>
      </div>

      <footer className="mt-20 text-sm text-gray-500">
        Built with ❤️ using Bolt + OpenAI + Recharts
      </footer>
    </main>
  );
};

export default Home;