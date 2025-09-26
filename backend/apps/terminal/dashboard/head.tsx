export default function Head() {
  return (
    <>
      <title>Trading Dashboard | Hedge Fund Terminal</title>
      <meta name="description" content="Real-time multi-asset trading dashboard with analytics, risk, and portfolio insights." />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <meta charSet="utf-8" />

      {/* Open Graph (for link previews) */}
      <meta property="og:title" content="Trading Dashboard" />
      <meta property="og:description" content="Monitor markets, portfolio, and risk in real-time with Bloomberg-style terminal UI." />
      <meta property="og:type" content="website" />
      <meta property="og:site_name" content="Hedge Fund Terminal" />

      {/* Favicon */}
      <link rel="icon" href="/favicon.ico" />

      {/* Theme color (dark terminal aesthetic) */}
      <meta name="theme-color" content="#0b0b0b" />
    </>
  );
}