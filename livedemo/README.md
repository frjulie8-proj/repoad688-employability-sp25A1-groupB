# Live Demo (repo only — not published to the website)

This folder holds the interactive K-Means live demo. It is intentionally
**excluded from the website** (it is not in the `_quarto.yml` render list and
not in the navbar), so it lives in the codebase only.

`live_demo.qmd` contains:

- an interactive K-Means cluster scatter,
- the **multi-filter dashboard** with three independent dropdowns that combine
  freely — **segment**, **career stage**, and **gender** — recoloring a US
  choropleth (built with Observable JS), and
- a Plotly animated map cycling career stages.

## How to view it

The dashboard uses Observable JS, which needs to be **served** (not opened as a
raw file). From the repo root run:

```bash
quarto preview livedemo/live_demo.qmd
```

That renders it and opens it in your browser with the dropdowns live. To just
produce an HTML file instead:

```bash
quarto render livedemo/live_demo.qmd
```

It reads the same `data/processed/` outputs and `analysis/` helpers as the rest
of the project, so no extra setup is needed beyond the project environment.
