// frontend/config/topics.ts

export type Topic = {
  id: string;       // unique identifier
  label: string;    // human-readable name
  description?: string; // optional helper text
  color?: string;   // UI hint (badge color, chart line color, etc.)
};

/**
 * Core project-wide topics.
 * These can be used in:
 *  - news filters
 *  - alerts panel
 *  - knowledge graph view
 *  - chat agent routing
 */
export const TOPICS: Topic[] = [
  {
    id: "macro",
    label: "Macroeconomics",
    description: "Rates, inflation, central banks, economic indicators",
    color: "blue",
  },
  {
    id: "equities",
    label: "Equities",
    description: "Stocks, indices, corporate actions",
    color: "green",
  },
  {
    id: "fx",
    label: "Foreign Exchange",
    description: "Currencies, USD strength, INR volatility",
    color: "yellow",
  },
  {
    id: "crypto",
    label: "Cryptocurrencies",
    description: "BTC, ETH, on-chain metrics, altcoins",
    color: "purple",
  },
  {
    id: "commodities",
    label: "Commodities",
    description: "Oil, gold, copper, agriculture markets",
    color: "orange",
  },
  {
    id: "derivatives",
    label: "Derivatives",
    description: "Options, futures, swaps, structured products",
    color: "red",
  },
  {
    id: "esg",
    label: "ESG",
    description: "Sustainability, governance, carbon, social risk",
    color: "teal",
  },
  {
    id: "risk",
    label: "Risk & Stress",
    description: "Liquidity stress, volatility spikes, contagion",
    color: "pink",
  },
];

/**
 * Quick lookup map for O(1) access by id
 */
export const TOPIC_MAP: Record<string, Topic> = Object.fromEntries(
  TOPICS.map((t) => [t.id, t])
);