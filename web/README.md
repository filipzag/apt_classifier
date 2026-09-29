# APT Attribution Assistant — web front end

React 19 + TypeScript + Vite + Tailwind v4 single-page app served by the FastAPI backend in
`../api`. The API is the source of truth for every payload; `src/api/types.ts` mirrors
`api/serializers.py`.

```bash
make web-install   # npm ci
make web-dev       # Vite on http://127.0.0.1:5173 with /api proxied to the API on 8100
make api           # the API (start it first; it loads the embedder and Chroma on startup)
make web-build     # production bundle in web/dist, served by the API at /
make web-check     # tsc + oxlint
```

Layout: `src/pages/*` are the routed screens (dashboard, intake, evidence, technique review,
interview, assessment, canvas, diamond, live intelligence, AI review); `src/components/ui.tsx`
holds the design primitives (buttons, cards, badges, metrics, tables, slide-over, tabs);
`src/components/LayeredGraph.tsx` is the deterministic SVG layout shared by the canvas and the
diamond model; `src/api/hooks.ts` wraps TanStack Query with the cache-writing mutations.
