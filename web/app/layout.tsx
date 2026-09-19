import type { Metadata } from "next";
import { Suspense } from "react";
import { JetBrains_Mono, Space_Grotesk } from "next/font/google";
import { Nav } from "@/components/Nav";
import { ApiStatusBanner } from "@/components/ApiStatusBanner";
import { TickerTape } from "@/components/TickerTape";
import "./globals.css";

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--jetbrains-mono",
  display: "swap",
});

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--space-grotesk",
  display: "swap",
});

export const metadata: Metadata = {
  title: "CUBO+ Risk Terminal",
  description:
    "Live VaR/ES, model comparison, portfolio, capital, and regimes for the cryptorisk study.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`h-full ${jetbrainsMono.variable} ${spaceGrotesk.variable}`}>
      <body className="flex min-h-full flex-col bg-bg text-text antialiased">
        <ApiStatusBanner />
        <TickerTape />
        <div className="flex min-h-0 flex-1 flex-col md:flex-row">
          <Suspense>
            <Nav />
          </Suspense>
          <main className="min-w-0 flex-1 overflow-x-hidden px-4 py-6 md:px-10 md:py-8">{children}</main>
        </div>
      </body>
    </html>
  );
}
