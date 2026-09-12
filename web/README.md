# cryptorisk web

Next.js + TypeScript + Tailwind v4 front-end over `cryptorisk`'s FastAPI
(`../src/cryptorisk/api`). A third, independent surface over the same study
outputs as the Streamlit dashboard (`../src/cryptorisk/dashboard`) — richer
custom visuals, and a stack that's easy to host (e.g. Vercel).

## Run

```bash
# from the cryptorisk repo root: start the API first (separate process)
.venv/Scripts/python -m pip install -e ".[api]"
.venv/Scripts/python -m uvicorn cryptorisk.api.app:app --port 8000

# then, in this directory
npm install
cp .env.example .env.local   # NEXT_PUBLIC_API_BASE_URL, defaults to localhost:8000
npm run dev                   # http://localhost:3000
```

## Layout

```
app/
  page.tsx            Overview — live price + VaR/ES band (lightweight-charts)
  models/page.tsx     Model Comparison — FZ0/MCS, coverage, ES tests, GW-CPA
  portfolio/page.tsx  4-asset basket ranking + live composition
  capital/page.tsx    FRTB capital stack, limits, estimation risk, hedge
  regimes/page.tsx    MS-GARCH crisis probability vs. realized vol
components/           Nav, MetricCard, Badge, DataTable, Fz0BarChart, PriceChart
lib/
  api.ts              typed fetch client + response interfaces
  hooks.ts            SWR hooks (polling) built on api.ts
  format.ts           USD/percent/date formatting helpers
```

Client-rendered throughout (no server-side data fetching) — every page fetches
from the API via the SWR hooks in `lib/hooks.ts`, matching the `NEXT_PUBLIC_API_BASE_URL`
in `.env.local`. `npm run build` produces a fully static shell (no route needs
server data at build time); all data comes from the API at runtime in the browser.
