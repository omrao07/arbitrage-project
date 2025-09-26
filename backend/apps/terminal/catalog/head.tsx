export default function Head() {
  return (
    <>
      <title>Catalog · Terminal</title>
      <meta
        name="description"
        content="Explore and compare Bloomberg, Koyfin, and Hammer Pro functions. Full searchable catalog of market, analytics, risk, execution, and portfolio features."
      />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <meta name="theme-color" content="#0b0b0b" />

      {/* favicons (replace /icons/* with your actual paths) */}
      <link rel="icon" href="/icons/favicon.ico" sizes="any" />
      <link rel="icon" type="image/svg+xml" href="/icons/icon.svg" />
      <link rel="apple-touch-icon" href="/icons/apple-touch-icon.png" />
      <link rel="manifest" href="/manifest.json" />

      {/* Open Graph for social */}
      <meta property="og:title" content="Terminal Catalog" />
      <meta
        property="og:description"
        content="Full catalog of Bloomberg, Koyfin, and Hammer Pro functions and features."
      />
      <meta property="og:type" content="website" />
      <meta property="og:url" content="https://yourdomain.com/catalog" />
      <meta property="og:image" content="https://yourdomain.com/og-image.png" />

      {/* Twitter cards */}
      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:title" content="Terminal Catalog" />
      <meta
        name="twitter:description"
        content="Search and compare every Bloomberg, Koyfin, and Hammer Pro feature."
      />
      <meta name="twitter:image" content="https://yourdomain.com/og-image.png" />
    </>
  );
}