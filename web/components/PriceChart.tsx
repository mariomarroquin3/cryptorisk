"use client";

import { useEffect, useRef } from "react";
import {
  ColorType,
  createChart,
  createSeriesMarkers,
  IChartApi,
  LineSeries,
  SeriesMarker,
  Time,
} from "lightweight-charts";

export interface PricePoint {
  time: string; // "YYYY-MM-DD"
  value: number;
}

export interface ChartProps {
  price: PricePoint[];
  varLine: PricePoint[];
  esLine: PricePoint[];
  breaches: PricePoint[];
  livePrice?: number | null;
}

export function PriceChart({ price, varLine, esLine, breaches, livePrice }: ChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#d7dce1",
        fontFamily: "JetBrains Mono, ui-monospace, Consolas, monospace",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "#1e242b" },
        horzLines: { color: "#1e242b" },
      },
      rightPriceScale: { borderColor: "#1e242b" },
      timeScale: { borderColor: "#1e242b" },
      autoSize: true,
    });
    chartRef.current = chart;

    const priceSeries = chart.addSeries(LineSeries, {
      color: "#d7dce1",
      lineWidth: 2,
      title: "price",
    });
    priceSeries.setData(price);

    const varSeries = chart.addSeries(LineSeries, {
      color: "#ffb020",
      lineWidth: 1,
      lineStyle: 2,
      title: "VaR",
    });
    varSeries.setData(varLine);

    const esSeries = chart.addSeries(LineSeries, {
      color: "#ff4d4f",
      lineWidth: 1,
      lineStyle: 2,
      title: "ES",
    });
    esSeries.setData(esLine);

    if (breaches.length > 0) {
      const markers: SeriesMarker<Time>[] = breaches.map((b) => ({
        time: b.time as Time,
        position: "belowBar",
        color: "#ff4d4f",
        shape: "circle",
        text: "breach",
      }));
      createSeriesMarkers(priceSeries, markers);
    }

    if (livePrice != null && Number.isFinite(livePrice)) {
      priceSeries.createPriceLine({
        price: livePrice,
        color: "#3ddc84",
        lineWidth: 1,
        lineStyle: 3,
        title: "live",
      });
    }

    chart.timeScale().fitContent();

    return () => {
      chart.remove();
      chartRef.current = null;
    };
  }, [price, varLine, esLine, breaches, livePrice]);

  return <div ref={containerRef} className="h-[420px] w-full" />;
}
