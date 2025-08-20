// frontend/pages/index.js
// Main landing page of your arbitrage dashboard

import Head from "next/head";
import dynamic from "next/dynamic";

// Lazy-load main dashboard component to improve startup time
const Dashboard = dynamic(() => import("@/components/Dashboard"), {
  ssr: false,
});

export default function Home() {
  return (
    <>
      <Head>
        <title>Global Arbitrage Dashboard</title>
        <meta
          name="description"
          content="Real-time global arbitrage monitoring and execution platform"
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <main className="min-h-screen bg-gray-900 text-white">
        <Dashboard />
      </main>
    </>
  );
}