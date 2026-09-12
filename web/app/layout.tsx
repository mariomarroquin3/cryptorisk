import type { Metadata } from "next";
import { Nav } from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "CUBO+ Risk Terminal",
  description:
    "Live VaR/ES, model comparison, portfolio, capital, and regimes for the cryptorisk study.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full">
      <body className="flex min-h-full bg-bg text-text antialiased">
        <Nav />
        <main className="min-w-0 flex-1 overflow-x-hidden px-6 py-6">{children}</main>
      </body>
    </html>
  );
}
