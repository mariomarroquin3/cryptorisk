"use client";

import { useEffect, useRef } from "react";
import {
  ColorType,
  createChart,
  createSeriesMarkers,
  IChartApi,
  IPriceLine,
  ISeriesApi,
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
  /** Today's live re-fit when the backtest band does not reach it: drawn as
   * detached points, never joined to the (older) walk-forward line. */
  liveVar?: PricePoint | null;
  liveEs?: PricePoint | null;
}

export function PriceChart({ price, varLine, esLine, breaches, livePrice, liveVar, liveEs }: ChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const priceSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const liveLineRef = useRef<IPriceLine | null>(null);
  const livePriceRef = useRef(livePrice);

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

    for (const [pt, color, title] of [[liveVar, "#ffb020", "VaR live"], [liveEs, "#ff4d4f", "ES live"]] as const) {
      if (!pt) continue;
      const dot = chart.addSeries(LineSeries, {
        color,
        lineVisible: false,
        pointMarkersVisible: true,
        pointMarkersRadius: 3.5,
        title,
      });
      dot.setData([pt]);
    }

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

    priceSeriesRef.current = priceSeries;
    const lp = livePriceRef.current;
    liveLineRef.current =
      lp != null && Number.isFinite(lp)
        ? priceSeries.createPriceLine({ price: lp, color: "#3ddc84", lineWidth: 1, lineStyle: 3, title: "live" })
        : null;

    chart.timeScale().fitContent();

    return () => {
      chart.remove();
      chartRef.current = null;
      priceSeriesRef.current = null;
      liveLineRef.current = null;
    };
  }, [price, varLine, esLine, breaches, liveVar, liveEs]);

  // The spot price ticks every few seconds: move its line in place instead of
  // rebuilding the chart (which reset the user's zoom and scroll each time).
  useEffect(() => {
    livePriceRef.current = livePrice;
    const series = priceSeriesRef.current;
    if (!series) return;
    if (livePrice == null || !Number.isFinite(livePrice)) {
      if (liveLineRef.current) series.removePriceLine(liveLineRef.current);
      liveLineRef.current = null;
    } else if (liveLineRef.current) {
      liveLineRef.current.applyOptions({ price: livePrice });
    } else {
      liveLineRef.current = series.createPriceLine({
        price: livePrice, color: "#3ddc84", lineWidth: 1, lineStyle: 3, title: "live",
      });
    }
  }, [livePrice]);

  return <div ref={containerRef} className="h-[420px] w-full" />;
}
