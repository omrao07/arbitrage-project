// depth.tsx
// Order Book Depth Chart (bids vs asks)

import React from "react"
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts"
import { Card, CardContent } from "../../ components/ui/card"

// Types
interface DepthPoint {
  price: number
  size: number
  cumulative: number
  side: "bid" | "ask"
}

interface DepthProps {
  bids: { price: number; size: number }[]
  asks: { price: number; size: number }[]
}

// Helper to compute cumulative size
const buildDepth = (
  levels: { price: number; size: number }[],
  side: "bid" | "ask"
): DepthPoint[] => {
  let cum = 0
  const sorted =
    side === "bid"
      ? [...levels].sort((a, b) => b.price - a.price)
      : [...levels].sort((a, b) => a.price - b.price)

  return sorted.map((l) => {
    cum += l.size
    return { ...l, cumulative: cum, side }
  })
}

const DepthChart: React.FC<DepthProps> = ({ bids, asks }) => {
  const bidDepth = buildDepth(bids, "bid")
  const askDepth = buildDepth(asks, "ask")
  const data = [...bidDepth, ...askDepth]

  return (
    <Card className="w-full h-[400px]">
      <CardContent className="h-full p-4">
        <h2 className="text-lg font-semibold mb-2">Order Book Depth</h2>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <XAxis
              dataKey="price"
              type="number"
              domain={["auto", "auto"]}
              tickFormatter={(v) => v.toFixed(2)}
            />
            <YAxis
              dataKey="cumulative"
              domain={[0, "auto"]}
              tickFormatter={(v) => v.toFixed(0)}
            />
            <Tooltip
              formatter={(value, name, props) => {
                if (name === "cumulative") return [value, "Cumulative Size"]
                if (name === "size") return [value, "Size"]
                return [value, name]
              }}
              labelFormatter={(label) => `Price: ${label}`}
            />
            <Area
              type="stepAfter"
              dataKey="cumulative"
              data={bidDepth}
              stroke="#22c55e"
              fill="#22c55e"
              opacity={0.3}
            />
            <Area
              type="stepAfter"
              dataKey="cumulative"
              data={askDepth}
              stroke="#ef4444"
              fill="#ef4444"
              opacity={0.3}
            />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}

export default DepthChart
