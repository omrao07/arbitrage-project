// frontend/components/AltData.tsx
import React, { useEffect, useState } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, AreaChart, Area } from "recharts";

interface AltDataPoint {
  timestamp: string;
  value: number;
}

export default function AltData() {
  const [cardSpend, setCardSpend] = useState<AltDataPoint[]>([]);
  const [satLights, setSatLights] = useState<AltDataPoint[]>([]);
  const [shipping, setShipping] = useState<AltDataPoint[]>([]);
  const [geo, setGeo] = useState<AltDataPoint[]>([]);

  useEffect(() => {
    async function fetchData() {
      try {
        const [cardRes, lightsRes, shipRes, geoRes] = await Promise.all([
          fetch("/api/altdata/card_spend"),
          fetch("/api/altdata/satellite_lights"),
          fetch("/api/altdata/shipping_traffic"),
          fetch("/api/altdata/geo_spatial"),
        ]);
        setCardSpend(await cardRes.json());
        setSatLights(await lightsRes.json());
        setShipping(await shipRes.json());
        setGeo(await geoRes.json());
      } catch (err) {
        console.error("AltData fetch error:", err);
      }
    }
    fetchData();
  }, []);

  const renderChart = (data: AltDataPoint[], color: string, label: string) => (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={data}>
        <defs>
          <linearGradient id={`color-${label}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={color} stopOpacity={0.8}/>
            <stop offset="95%" stopColor={color} stopOpacity={0}/>
          </linearGradient>
        </defs>
        <XAxis dataKey="timestamp" hide />
        <YAxis />
        <Tooltip />
        <Area
          type="monotone"
          dataKey="value"
          stroke={color}
          fillOpacity={1}
          fill={`url(#color-${label})`}
        />
                  </AreaChart>
                </ResponsiveContainer>
              );
            ;
      
        return (
          <div>
            <h2>Card Spend</h2>
            {renderChart(cardSpend, "#8884d8", "cardSpend")}
            <h2>Satellite Lights</h2>
            {renderChart(satLights, "#82ca9d", "satLights")}
            <h2>Shipping Traffic</h2>
            {renderChart(shipping, "#ffc658", "shipping")}
            <h2>Geo Spatial</h2>
            {renderChart(geo, "#ff7300", "geo")}
          </div>
        );
        }