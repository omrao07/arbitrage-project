/**
 * screener/head.tsx
 * Minimal, zero-import <Head> for the Screener routes (Next.js App Router).
 * No external links/imports.
 */

export default function Head() {
  const title = "Screeners · Dashboard";
  const description =
    "Build, run, and save multi-asset screens with filters, rules, and presets.";
  const url = "https://example.app/screener"; // replace at deploy
  const theme = "#0a0a0a";

  return (
    <>
      <title>{title}</title>
      <meta charSet="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <meta name="theme-color" content={theme} />
      <meta name="color-scheme" content="dark light" />
      <meta name="robots" content="noindex,nofollow" />
      <meta name="description" content={description} />

      {/* Open Graph */}
      <meta property="og:type" content="website" />
      <meta property="og:title" content={title} />
      <meta property="og:description" content={description} />
      <meta property="og:url" content={url} />
      <meta property="og:site_name" content="Dashboard" />

      {/* Twitter */}
      <meta name="twitter:card" content="summary" />
      <meta name="twitter:title" content={title} />
      <meta name="twitter:description" content={description} />

      {/* App icons (inline data URI placeholder; replace with real /icon.png if you have one) */}
      <link
        rel="icon"
        type="image/png"
        href={`data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'><rect width='64' height='64' rx='12' fill='%230a0a0a'/><path d='M16 40h12V16h8v32h12' stroke='%23a3e635' stroke-width='6' fill='none' stroke-linecap='round'/></svg>`}
      />
    </>
  );
}