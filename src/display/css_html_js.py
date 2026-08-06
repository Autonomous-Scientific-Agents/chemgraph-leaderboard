custom_css = """

/* ============================================================
   ChemGraph Leaderboard — Clean & Modern Theme
   ============================================================ */

/* --- CSS Custom Properties (respects Gradio light/dark) --- */
:root, .light {
    --cg-primary: #2563eb;
    --cg-primary-light: #3b82f6;
    --cg-accent: #0d9488;
    --cg-accent-light: #14b8a6;
    --cg-surface: #ffffff;
    --cg-surface-alt: #f8fafc;
    --cg-surface-hover: #f1f5f9;
    --cg-border: #e2e8f0;
    --cg-border-light: #f1f5f9;
    --cg-text-primary: #0f172a;
    --cg-text-secondary: #475569;
    --cg-text-muted: #94a3b8;
    --cg-shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
    --cg-shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    --cg-shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
    --cg-radius: 12px;
    --cg-radius-sm: 8px;
    --cg-gradient: linear-gradient(135deg, #1e40af 0%, #0d9488 100%);
    --cg-gradient-subtle: linear-gradient(135deg, #eff6ff 0%, #f0fdfa 100%);
}

.dark {
    --cg-primary: #3b82f6;
    --cg-primary-light: #60a5fa;
    --cg-accent: #14b8a6;
    --cg-accent-light: #2dd4bf;
    --cg-surface: #1e293b;
    --cg-surface-alt: #0f172a;
    --cg-surface-hover: #334155;
    --cg-border: #334155;
    --cg-border-light: #1e293b;
    --cg-text-primary: #f1f5f9;
    --cg-text-secondary: #94a3b8;
    --cg-text-muted: #64748b;
    --cg-shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.2);
    --cg-shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -2px rgba(0, 0, 0, 0.2);
    --cg-shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -4px rgba(0, 0, 0, 0.25);
    --cg-gradient: linear-gradient(135deg, #1e3a8a 0%, #065f46 100%);
    --cg-gradient-subtle: linear-gradient(135deg, #1e293b 0%, #0f2027 100%);
}

/* ============================================================
   1. HEADER / TITLE BANNER
   ============================================================ */
#cg-title-banner {
    /* Bright blue hero: layered blue gradient + soft light blooms, with faint
       two molecular line-art fragments partially shown at the top-left and
       bottom-right (chemistry atmosphere). Glossy top highlight via inset. */
    background:
        url("data:image/svg+xml,%3Csvg%20xmlns%3D%27http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%27%20width%3D%27374%27%20height%3D%27274%27%20viewBox%3D%2744%2020%20374%20274%27%3E%3Cg%20fill%3D%27none%27%20stroke%3D%27rgba%28255%2C255%2C255%2C0.18%29%27%20stroke-width%3D%273%27%20stroke-linecap%3D%27round%27%3E%3Cpath%20d%3D%27M290%20150%20L250%20219%20L170%20219%20L130%20150%20L170%2081%20L250%2081%20Z%27%2F%3E%3Cpath%20d%3D%27M290%20150%20L360%20150%20L400%20108%27%2F%3E%3Cpath%20d%3D%27M250%2081%20L290%2034%27%2F%3E%3Cpath%20d%3D%27M170%20219%20L138%20278%27%2F%3E%3Cpath%20d%3D%27M130%20150%20L58%20150%27%2F%3E%3Cpath%20d%3D%27M243%2092%20L201%2092%27%2F%3E%3C%2Fg%3E%3Cg%20fill%3D%27rgba%28255%2C255%2C255%2C0.27%29%27%3E%3Ccircle%20cx%3D%27290%27%20cy%3D%27150%27%20r%3D%278%27%2F%3E%3Ccircle%20cx%3D%27250%27%20cy%3D%27219%27%20r%3D%278%27%2F%3E%3Ccircle%20cx%3D%27170%27%20cy%3D%27219%27%20r%3D%278%27%2F%3E%3Ccircle%20cx%3D%27130%27%20cy%3D%27150%27%20r%3D%278%27%2F%3E%3Ccircle%20cx%3D%27170%27%20cy%3D%2781%27%20r%3D%278%27%2F%3E%3Ccircle%20cx%3D%27250%27%20cy%3D%2781%27%20r%3D%278%27%2F%3E%3Ccircle%20cx%3D%27360%27%20cy%3D%27150%27%20r%3D%278%27%2F%3E%3Ccircle%20cx%3D%27400%27%20cy%3D%27108%27%20r%3D%278%27%2F%3E%3Ccircle%20cx%3D%27290%27%20cy%3D%2734%27%20r%3D%278%27%2F%3E%3Ccircle%20cx%3D%27138%27%20cy%3D%27278%27%20r%3D%278%27%2F%3E%3Ccircle%20cx%3D%2758%27%20cy%3D%27150%27%20r%3D%278%27%2F%3E%3C%2Fg%3E%3C%2Fsvg%3E") no-repeat left -85px top -55px / 260px auto,
        url("data:image/svg+xml,%3Csvg%20xmlns%3D%27http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%27%20width%3D%27374%27%20height%3D%27274%27%20viewBox%3D%2744%2020%20374%20274%27%3E%3Cg%20fill%3D%27none%27%20stroke%3D%27rgba%28255%2C255%2C255%2C0.18%29%27%20stroke-width%3D%273%27%20stroke-linecap%3D%27round%27%3E%3Cpath%20d%3D%27M290%20150%20L250%20219%20L170%20219%20L130%20150%20L170%2081%20L250%2081%20Z%27%2F%3E%3Cpath%20d%3D%27M290%20150%20L360%20150%20L400%20108%27%2F%3E%3Cpath%20d%3D%27M250%2081%20L290%2034%27%2F%3E%3Cpath%20d%3D%27M170%20219%20L138%20278%27%2F%3E%3Cpath%20d%3D%27M130%20150%20L58%20150%27%2F%3E%3Cpath%20d%3D%27M243%2092%20L201%2092%27%2F%3E%3C%2Fg%3E%3Cg%20fill%3D%27rgba%28255%2C255%2C255%2C0.27%29%27%3E%3Ccircle%20cx%3D%27290%27%20cy%3D%27150%27%20r%3D%278%27%2F%3E%3Ccircle%20cx%3D%27250%27%20cy%3D%27219%27%20r%3D%278%27%2F%3E%3Ccircle%20cx%3D%27170%27%20cy%3D%27219%27%20r%3D%278%27%2F%3E%3Ccircle%20cx%3D%27130%27%20cy%3D%27150%27%20r%3D%278%27%2F%3E%3Ccircle%20cx%3D%27170%27%20cy%3D%2781%27%20r%3D%278%27%2F%3E%3Ccircle%20cx%3D%27250%27%20cy%3D%2781%27%20r%3D%278%27%2F%3E%3Ccircle%20cx%3D%27360%27%20cy%3D%27150%27%20r%3D%278%27%2F%3E%3Ccircle%20cx%3D%27400%27%20cy%3D%27108%27%20r%3D%278%27%2F%3E%3Ccircle%20cx%3D%27290%27%20cy%3D%2734%27%20r%3D%278%27%2F%3E%3Ccircle%20cx%3D%27138%27%20cy%3D%27278%27%20r%3D%278%27%2F%3E%3Ccircle%20cx%3D%2758%27%20cy%3D%27150%27%20r%3D%278%27%2F%3E%3C%2Fg%3E%3C%2Fsvg%3E") no-repeat right -98px bottom -72px / 285px auto,
        radial-gradient(900px 440px at 86% -22%, rgba(147, 197, 253, 0.50), transparent 60%),
        radial-gradient(760px 420px at 6% 122%, rgba(37, 99, 235, 0.42), transparent 55%),
        linear-gradient(125deg, #1e40af 0%, #2563eb 52%, #3b82f6 100%);
    border-radius: var(--cg-radius);
    padding: 2rem 2.5rem 1.8rem;
    margin-bottom: 1rem;
    box-shadow: var(--cg-shadow-lg), inset 0 1px 0 0 rgba(255, 255, 255, 0.20);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.5rem;
    text-align: left;
    position: relative;
    overflow: hidden;
}

/* ChemGraph icon seated on a white disc for contrast on the blue banner. */
#cg-title-banner .cg-title-logo {
    flex-shrink: 0;
    width: 76px;
    height: 76px;
    border-radius: 50%;
    background: #ffffff;
    padding: 7px;
    box-sizing: border-box;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.18);
    position: relative;
    z-index: 1;
}

#cg-title-banner .cg-title-text {
    text-align: left;
    position: relative;
    z-index: 1;
}

#cg-title-banner h1 {
    color: #ffffff !important;
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    margin: 0 0 0.3rem 0 !important;
    letter-spacing: -0.02em;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
    position: relative;
    z-index: 1;
}

#cg-title-banner .cg-subtitle {
    color: rgba(255, 255, 255, 0.85);
    font-size: 1.05rem;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.01em;
    position: relative;
    z-index: 1;
}

#cg-title-banner .cg-badge-row {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-start;
    gap: 0.6rem;
    margin-top: 1rem;
    position: relative;
    z-index: 1;
}

#cg-title-banner .cg-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(4px);
    color: #ffffff;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* ============================================================
   2. INTRODUCTION TEXT (card-like)
   ============================================================ */
.markdown-text {
    font-size: 15px !important;
    line-height: 1.7 !important;
    color: var(--cg-text-primary) !important;
}

#cg-intro-block {
    background: var(--cg-surface) !important;
    border: 1px solid var(--cg-border) !important;
    border-radius: var(--cg-radius) !important;
    padding: 1.2rem 1.5rem !important;
    box-shadow: var(--cg-shadow-sm) !important;
    margin-bottom: 0.75rem !important;
}

#cg-intro-block table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: var(--cg-radius-sm);
    overflow: hidden;
    border: 1px solid var(--cg-border);
    margin: 1rem 0;
}

#cg-intro-block th {
    /* Gentle light-blue gradient (matches the banner family), white text. */
    background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.65rem 1rem !important;
    text-align: left !important;
    border: none !important;
}

#cg-intro-block td {
    padding: 0.55rem 1rem !important;
    border-bottom: 1px solid var(--cg-border-light) !important;
    font-size: 0.88rem !important;
    border-left: none !important;
    border-right: none !important;
}

#cg-intro-block tr:nth-child(even) td {
    background: var(--cg-surface-alt) !important;
}

#cg-intro-block tr:last-child td {
    border-bottom: none !important;
}

/* ============================================================
   3. TAB BUTTONS
   ============================================================ */
.tab-buttons button {
    font-size: 16px !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.4rem !important;
    border-radius: var(--cg-radius-sm) var(--cg-radius-sm) 0 0 !important;
    transition: all 0.2s ease !important;
    border: 1px solid transparent !important;
    border-bottom: none !important;
    color: var(--cg-text-secondary) !important;
    background: transparent !important;
}

.tab-buttons button:hover {
    color: var(--cg-primary) !important;
    background: var(--cg-surface-hover) !important;
}

.tab-buttons button.selected {
    color: var(--cg-primary) !important;
    font-weight: 600 !important;
    background: var(--cg-surface) !important;
    border-color: var(--cg-border) !important;
    border-bottom: 2px solid var(--cg-primary) !important;
    box-shadow: var(--cg-shadow-sm) !important;
}

/* ============================================================
   4. LEADERBOARD TABLE
   ============================================================ */
#leaderboard-table, #leaderboard-table-lite {
    margin-top: 12px;
}

/* Table wrapper — card appearance */
#leaderboard-table .table-wrap,
#leaderboard-table-lite .table-wrap {
    border-radius: var(--cg-radius) !important;
    border: 1px solid var(--cg-border) !important;
    box-shadow: var(--cg-shadow-md) !important;
    overflow: hidden !important;
}

/* Table headers */
#leaderboard-table table thead th,
#leaderboard-table-lite table thead th {
    background: var(--cg-surface-alt) !important;
    color: var(--cg-text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 0.7rem 0.6rem !important;
    border-bottom: 2px solid var(--cg-border) !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    white-space: nowrap;
    position: sticky;
    top: 0;
    z-index: 2;
}

/* Table body cells */
#leaderboard-table table tbody td,
#leaderboard-table-lite table tbody td {
    padding: 0.6rem 0.6rem !important;
    font-size: 0.88rem !important;
    border-bottom: 1px solid var(--cg-border-light) !important;
    transition: background 0.15s ease;
}

/* Alternating row stripes */
#leaderboard-table table tbody tr:nth-child(even),
#leaderboard-table-lite table tbody tr:nth-child(even) {
    background: var(--cg-surface-alt) !important;
}

/* Row hover highlight */
#leaderboard-table table tbody tr:hover td,
#leaderboard-table-lite table tbody tr:hover td {
    background: var(--cg-surface-hover) !important;
}

/* Model name column — prevent overflow */
#leaderboard-table td:nth-child(2),
#leaderboard-table th:nth-child(2) {
    max-width: 400px;
    overflow: auto;
    white-space: nowrap;
}

/* Search bar */
#search-bar-table-box > div:first-child {
    background: none;
    border: none;
}

#search-bar {
    padding: 0px;
}

/* ============================================================
   5. CONTROLS PANELS  (left = Data + Model family, right = Tasks)
   ============================================================
   Gradio's side-by-side layout is kept. JS redistributes labels:
     LEFT  (column-selector)    : Data + Model family
     RIGHT (model-family filter): Tasks
*/

/* Section header (Data / Model family / Tasks). */
[data-cg-role="column-selector"] .cg-group-header,
[data-cg-role="model-family-filter"] .cg-group-header {
    font-weight: 600;
    font-size: 0.78rem;
    color: var(--cg-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin: 0.55rem 0 0.25rem;
    padding-bottom: 0.25rem;
    border-bottom: 1px solid var(--cg-border-light);
    flex-basis: 100%;
    width: 100%;
}
[data-cg-role="column-selector"] > .cg-group-header-data,
[data-cg-role="model-family-filter"] > .cg-group-header-tasks {
    margin-top: 0;
}

/* ============================================================
   6. TRENDS TAB
   ============================================================ */
/* Trends controls — one bordered card with three zones (Data / View /
   Actions) separated by thin vertical dividers. */
.cg-controls-row {
    background: var(--cg-surface) !important;
    border: 1px solid var(--cg-border) !important;
    border-radius: var(--cg-radius-sm) !important;
    padding: 0.9rem 1rem !important;
    box-shadow: var(--cg-shadow-sm) !important;
    gap: 0 !important;
    align-items: stretch !important;
}

.cg-controls-row .cg-zone {
    padding: 0 1rem !important;
    border-left: 1px solid var(--cg-border-light);
    display: flex !important;
    flex-direction: column !important;
    gap: 0.55rem !important;
    min-width: 0 !important;
}
.cg-controls-row .cg-zone:first-child {
    border-left: 0;
    padding-left: 0 !important;
}
.cg-controls-row .cg-zone:last-child {
    padding-right: 0 !important;
}

/* Zone-internal blocks blend into the card. */
.cg-controls-row .cg-zone .block,
.cg-controls-row .cg-zone .form {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Uniform field-title size for every control in the Trends panel.
   Gradio uses different wrappers for different control types:
     - single-value Dropdown (Workflow): <div.container> > <span>
     - multiselect Dropdown (Models)   : <label.container> > <span>
     - Textbox (Last updated)          : <label.container> > <span>
     - Radio items (Past week/...)     : <label> > <span> (these are
       option labels, not the field title — left alone)
   The field-title span carries class svelte-g2oxp3 in all three
   wrapper variants, so target it directly to get one consistent size. */
.cg-controls-row .cg-zone .container > span.svelte-g2oxp3 {
    font-size: 0.875rem !important;   /* 14px @ default root */
    color: var(--cg-text-secondary) !important;
    font-weight: 500 !important;
    line-height: 1.2 !important;
}

/* The radio (Date range) — chip-row instead of stacked. */
.cg-controls-row .cg-zone-view .wrap {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 0.35rem !important;
}

/* From/To date pickers — compact native-date inputs side by side. */
#cg-trend-date-range,
#cg-family-date-range {
    gap: 0.5rem !important;
}
.cg-date-input input {
    font-size: 0.82rem !important;
    padding: 0.25rem 0.4rem !important;
}

/* Models multiselect — single-row pill that keeps the frame a few
   pixels taller than the chips inside so they don't touch the top
   or bottom border. Scoped to dropdowns that have .token children
   (the multiselect) so the single-value Workflow dropdown keeps its
   default compact height. */
.cg-controls-row .cg-zone-data .gradio-dropdown[data-testid="dropdown"] .wrap:has(.token) {
    min-height: 51px !important;
    padding: 0 0.4rem !important;
    overflow: hidden !important;
}
.cg-controls-row .cg-zone-data .wrap:has(.token):hover,
.cg-controls-row .cg-zone-data .wrap:has(.token):focus-within {
    overflow-x: auto !important;
}
.cg-controls-row .cg-zone-data .wrap:has(.token) .wrap-inner {
    flex-wrap: nowrap !important;
    overflow: visible !important;
    padding: 8px 6px !important;
    align-items: center !important;
}
.cg-controls-row .cg-zone-data .wrap:has(.token) .wrap-inner .token {
    flex-shrink: 0 !important;
}

/* Last-view / Hub data caption — muted small text under Refresh. */
.cg-controls-row .cg-zone-actions .cg-last-updated,
.cg-controls-row .cg-zone-actions .cg-last-updated p,
.cg-controls-row .cg-zone-actions .cg-last-updated small {
    color: var(--cg-text-muted) !important;
    font-size: 0.7rem !important;
    line-height: 1.2 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Refresh button — compact outline pill using the theme primary. */
.cg-controls-row .cg-zone-actions button {
    background: var(--cg-surface) !important;
    color: var(--cg-primary) !important;
    border: 1px solid var(--cg-primary) !important;
    border-radius: var(--cg-radius-sm) !important;
    padding: 0.3rem 0.85rem !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    line-height: 1.2 !important;
    min-height: 0 !important;
    width: auto !important;
    align-self: flex-start;
    transition: all 0.15s ease;
}
.cg-controls-row .cg-zone-actions button:hover {
    background: var(--cg-primary) !important;
    color: #fff !important;
    box-shadow: var(--cg-shadow-sm) !important;
}
.cg-controls-row .cg-zone-actions button:active {
    transform: translateY(1px);
}

#cg-trend-chart,
#cg-family-chart {
    border: 1px solid var(--cg-border) !important;
    border-radius: var(--cg-radius) !important;
    box-shadow: var(--cg-shadow-sm) !important;
    overflow: hidden !important;
    padding: 0.5rem !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.5rem !important;
    background: var(--cg-surface) !important;
}

#cg-trend-summary-label h3,
#cg-family-summary-label h3 {
    color: var(--cg-primary) !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
}

#cg-trend-summary,
#cg-family-summary {
    border: 1px solid var(--cg-border) !important;
    border-radius: var(--cg-radius) !important;
    box-shadow: var(--cg-shadow-sm) !important;
    overflow: hidden !important;
}

#cg-trend-summary table thead th,
#cg-family-summary table thead th {
    background: var(--cg-surface-alt) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
    padding: 0.65rem 0.8rem !important;
    border-bottom: 2px solid var(--cg-border) !important;
}

#cg-trend-summary table tbody tr:nth-child(even),
#cg-family-summary table tbody tr:nth-child(even) {
    background: var(--cg-surface-alt) !important;
}

#cg-trend-summary table tbody tr:hover td,
#cg-family-summary table tbody tr:hover td {
    background: var(--cg-surface-hover) !important;
}

/* Refresh button */
#cg-refresh-btn {
    border-radius: var(--cg-radius-sm) !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    border: 1px solid var(--cg-border) !important;
}

#cg-refresh-btn:hover {
    border-color: var(--cg-primary) !important;
    color: var(--cg-primary) !important;
    box-shadow: var(--cg-shadow-sm) !important;
}

/* ============================================================
   7. ABOUT TAB
   ============================================================ */
#cg-about-content {
    background: var(--cg-surface) !important;
    border: 1px solid var(--cg-border) !important;
    border-radius: var(--cg-radius) !important;
    padding: 1.5rem 2rem !important;
    box-shadow: var(--cg-shadow-sm) !important;
}

#cg-about-content h2 {
    color: var(--cg-primary) !important;
    font-weight: 700 !important;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid var(--cg-border);
    margin-bottom: 0.8rem !important;
}

#cg-about-content h3 {
    color: var(--cg-text-primary) !important;
    font-weight: 600 !important;
}

#cg-about-content code {
    background: var(--cg-surface-alt) !important;
    border: 1px solid var(--cg-border) !important;
    border-radius: 4px !important;
    padding: 0.15em 0.4em !important;
    font-size: 0.88em !important;
}

#cg-about-content pre {
    background: var(--cg-surface-alt) !important;
    border: 1px solid var(--cg-border) !important;
    border-radius: var(--cg-radius-sm) !important;
    padding: 1rem !important;
}

/* ============================================================
   8. SUBMIT TAB
   ============================================================ */
#cg-submit-heading {
    color: var(--cg-primary) !important;
}

#cg-submit-heading h1 {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: var(--cg-primary) !important;
}

/* Submission button */
#cg-submit-btn {
    background: var(--cg-gradient) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.7rem 2rem !important;
    border-radius: var(--cg-radius-sm) !important;
    border: none !important;
    box-shadow: var(--cg-shadow-md) !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

#cg-submit-btn:hover {
    box-shadow: var(--cg-shadow-lg) !important;
    transform: translateY(-1px) !important;
    filter: brightness(1.05) !important;
}

/* Accordion headers — eval queues */
.cg-queue-accordion {
    border-radius: var(--cg-radius-sm) !important;
    border: 1px solid var(--cg-border) !important;
    box-shadow: var(--cg-shadow-sm) !important;
    margin-bottom: 0.5rem !important;
    overflow: hidden !important;
}

/* Eval queue guidance text */
#cg-submit-guide {
    background: var(--cg-surface) !important;
    border: 1px solid var(--cg-border) !important;
    border-radius: var(--cg-radius) !important;
    padding: 1rem 1.5rem !important;
    box-shadow: var(--cg-shadow-sm) !important;
}

#cg-submit-guide h2 {
    color: var(--cg-primary) !important;
    font-weight: 700 !important;
}

#cg-submit-guide h3 {
    color: var(--cg-text-primary) !important;
    font-weight: 600 !important;
}

/* ============================================================
   9. CITATION ACCORDION
   ============================================================ */
#cg-citation-section {
    margin-top: 1rem !important;
}

#cg-citation-section .label-wrap {
    font-weight: 600 !important;
    font-size: 1rem !important;
}

#citation-button span {
    font-size: 15px !important;
}

#citation-button textarea {
    font-size: 14px !important;
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace !important;
    background: var(--cg-surface-alt) !important;
    border: 1px solid var(--cg-border) !important;
    border-radius: var(--cg-radius-sm) !important;
    padding: 1rem !important;
    line-height: 1.6 !important;
}

#citation-button > label > button {
    margin: 6px;
    transform: scale(1.2);
    transition: all 0.2s ease !important;
}

#citation-button > label > button:hover {
    transform: scale(1.35) !important;
    color: var(--cg-primary) !important;
}

/* ============================================================
   10. SCALE LOGO & MISC
   ============================================================ */
#models-to-add-text {
    font-size: 18px !important;
}

#scale-logo {
    border-style: none !important;
    box-shadow: none;
    display: block;
    margin-left: auto;
    margin-right: auto;
    max-width: 600px;
}

#scale-logo .download {
    display: none;
}

/* ============================================================
   11. GLOBAL ENHANCEMENTS
   ============================================================ */

/* Smoother inputs */
.gradio-container input[type="text"],
.gradio-container textarea,
.gradio-container select {
    border-radius: var(--cg-radius-sm) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.gradio-container input[type="text"]:focus,
.gradio-container textarea:focus {
    border-color: var(--cg-primary) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
}

/* Dropdowns */
.gradio-container .wrap .wrap-inner {
    border-radius: var(--cg-radius-sm) !important;
}

/* Smooth accordion transitions */
.gradio-container .accordion {
    border-radius: var(--cg-radius-sm) !important;
    border: 1px solid var(--cg-border) !important;
    overflow: hidden !important;
    transition: box-shadow 0.2s ease !important;
}

.gradio-container .accordion:hover {
    box-shadow: var(--cg-shadow-sm) !important;
}

/* Links in the leaderboard */
.gradio-container a {
    color: var(--cg-primary) !important;
    text-decoration: underline !important;
    text-decoration-style: dotted !important;
    transition: color 0.15s ease !important;
}

.gradio-container a:hover {
    color: var(--cg-accent) !important;
}

/* ============================================================
   11b. HIGHLIGHTS SUB-TAB (KPI strip + efficiency frontier + task difficulty)
   ============================================================ */

/* KPI strip — 4 headline cards, pure HTML (paints before any Plotly). */
.cg-kpi-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin: 0.25rem 0 0.9rem;
}
.cg-kpi-card {
    flex: 1 1 180px;
    min-width: 160px;
    background: var(--cg-surface);
    border: 1px solid var(--cg-border);
    border-radius: var(--cg-radius);
    box-shadow: var(--cg-shadow-sm);
    padding: 0.7rem 0.9rem;
}
.cg-kpi-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--cg-text-muted);
}
.cg-kpi-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--cg-primary);
    line-height: 1.15;
    margin: 0.15rem 0;
}
.cg-kpi-model {
    font-size: 0.85rem;
    color: var(--cg-text-primary);
    word-break: break-word;
}
.cg-kpi-sub {
    font-size: 0.7rem;
    color: var(--cg-text-muted);
    margin-top: 0.1rem;
}
/* Info glyph next to a KPI label — hover (or focus) shows the native tooltip
   from its title= attribute explaining what the metric means. */
.cg-kpi-info {
    display: inline-block;
    color: var(--cg-text-muted);
    opacity: 0.7;
    cursor: help;
    font-size: 0.72rem;
    vertical-align: super;
    line-height: 1;
    outline: none;
}
.cg-kpi-info:hover,
.cg-kpi-info:focus {
    opacity: 1;
    color: var(--cg-primary);
}
.cg-kpi-empty {
    color: var(--cg-text-muted);
    padding: 1rem;
}

/* Frontier + task-difficulty plot framing (both use .cg-frontier-plot). */
.cg-frontier-plot {
    border: 1px solid var(--cg-border) !important;
    border-radius: var(--cg-radius) !important;
    box-shadow: var(--cg-shadow-sm) !important;
    background: var(--cg-surface) !important;
    overflow: hidden !important;
    padding: 0.5rem !important;
    width: 100% !important;
    min-width: 0 !important;
}
.cg-frontier-plot .js-plotly-plot,
.cg-frontier-plot .plot-container {
    width: 100% !important;
}

/* Contribute-a-Task sections: a clean white bordered card (a Column, not a
   gr.Group — so fields keep their own borders and stay clearly separated,
   with no gray gap-fill). */
/* ONE consistent gap everywhere inside the form cards: the column, the merged
   ".form" wrapper, and explicit rows all use the same value so the vertical
   (and in-row) rhythm is uniform regardless of component type. */
.cg-form-group {
    background: var(--cg-surface) !important;
    border: 1px solid var(--cg-border) !important;
    border-radius: var(--cg-radius) !important;
    padding: 0.8rem 1rem !important;
    margin-bottom: 0.5rem !important;
    gap: 0.3rem !important;
}
/* Kill Gradio's merged-input ".form" wrapper (gray fill + extra border) and
   align its gap to the column's. */
.cg-form-group .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    gap: 0.3rem !important;
    overflow: visible !important;
}
/* Same gap for explicit rows (e.g. name / email / org). */
.cg-form-group .row {
    gap: 0.3rem !important;
}

/* Gradio textareas default to overflow-y:scroll → an always-on scrollbar that
   shows as a gray strip at the right edge of single-line boxes (esp. on macOS
   "always show scrollbars"). Only show it when content actually overflows. */
.cg-form-group textarea {
    overflow-y: auto !important;
}

/* Section title — sits OUTSIDE/above each form card, in brand blue. */
.cg-form-section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #000000;
    margin: 0.8rem 0 0.4rem;
}
/* Section title + inline action button (e.g. Submission status + Refresh). */
.cg-form-section-title-row {
    align-items: center !important;
    gap: 0.6rem !important;
    flex-wrap: nowrap !important;
}
.cg-form-section-title-row .cg-form-section-title {
    margin: 0.4rem 0;
}

/* Opening description — boxed (a .cg-form-group card) with a blue heading. */
.cg-contribute-intro h1,
.cg-contribute-intro h2 {
    color: var(--cg-primary) !important;
    margin-top: 0 !important;
}

/* "Tools used" — render the CheckboxGroup options as clickable colored pills.
   Selected state is only input:checked (label class unchanged) → use :has(). */
.cg-tool-tags {
    background: var(--cg-surface) !important;
    border: 1px solid var(--cg-border) !important;
    border-radius: var(--cg-radius-sm) !important;
    box-shadow: none !important;
    padding: 0.6rem 0.8rem !important;
}
.cg-tool-tags label {
    display: inline-flex !important;
    align-items: center;
    border: 1.5px solid var(--tag, #94a3b8) !important;
    color: var(--tag, #475569) !important;
    background: transparent !important;
    border-radius: 999px !important;
    padding: 0.24rem 0.75rem !important;
    margin: 0.2rem 0.4rem 0.2rem 0 !important;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer !important;
    transition: all 0.12s ease;
}
.cg-tool-tags label input[type="checkbox"] {
    display: none !important;
}
.cg-tool-tags label span {
    margin: 0 !important;
}
.cg-tool-tags label:hover {
    filter: brightness(0.97);
}
.cg-tool-tags label:has(input:checked) {
    background: var(--tag, #2563eb) !important;
    border-color: var(--tag, #2563eb) !important;
    color: #ffffff !important;
}
/* per-tag color, by position
   (RDKit · MACE · TBLite · NWChem · ORCA · UMA · AIMNet2 · gRASPA · XANES) */
.cg-tool-tags label:nth-of-type(1) { --tag: #2563eb; }  /* RDKit  — blue   */
.cg-tool-tags label:nth-of-type(2) { --tag: #0d9488; }  /* MACE   — teal   */
.cg-tool-tags label:nth-of-type(3) { --tag: #d97706; }  /* TBLite — amber  */
.cg-tool-tags label:nth-of-type(4) { --tag: #16a34a; }  /* NWChem — green  */
.cg-tool-tags label:nth-of-type(5) { --tag: #7c3aed; }  /* ORCA   — violet */
.cg-tool-tags label:nth-of-type(6) { --tag: #db2777; }  /* UMA    — pink   */
.cg-tool-tags label:nth-of-type(7) { --tag: #0891b2; }  /* AIMNet2— cyan   */
.cg-tool-tags label:nth-of-type(8) { --tag: #dc2626; }  /* gRASPA — red    */
.cg-tool-tags label:nth-of-type(9) { --tag: #65a30d; }  /* XANES  — lime   */

/* ---------------------------------------------------------------- */
/* Submission status board (Contribute tab)                          */
/* ---------------------------------------------------------------- */
.cg-sub-section {
    margin-bottom: 1.1rem;
}
.cg-sub-section-head {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--cg-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin: 0.2rem 0 0.5rem;
}
.cg-sub-count {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 1.3rem;
    height: 1.3rem;
    padding: 0 0.4rem;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--cg-text-muted);
    background: var(--cg-surface-alt);
    border: 1px solid var(--cg-border);
    border-radius: 999px;
}
.cg-sub-board {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}
.cg-sub-card {
    background: var(--cg-surface);
    border: 1px solid var(--cg-border);
    border-radius: var(--cg-radius-sm);
    padding: 0.6rem 0.85rem;
}
/* The identity/progress row (left | right). Charts (if any) sit below it. */
.cg-sub-row {
    display: flex;
    align-items: stretch;
    justify-content: space-between;
    gap: 1rem;
}
/* Left column: identity (name + tags + submitter/date/category). */
.cg-sub-main {
    flex: 1 1 auto;
    min-width: 0;  /* let long names wrap instead of pushing the aside off */
}
/* Right column: progress/result pinned top, link pinned bottom. */
.cg-sub-aside {
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    justify-content: space-between;
    gap: 0.4rem;
    text-align: right;
}
.cg-sub-meta {
    font-size: 0.75rem;
    color: var(--cg-text-muted);
    margin-top: 0.3rem;
    line-height: 1.4;
}
.cg-sub-link {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--cg-primary);
    text-decoration: none;
    white-space: nowrap;
}
.cg-sub-link:hover {
    text-decoration: underline;
}
.cg-sub-name {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--cg-text-primary);
    line-height: 1.3;
}
.cg-sub-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin: 0.35rem 0 0.1rem;
}
.cg-sub-tag {
    display: inline-flex;
    align-items: center;
    font-size: 0.72rem;
    font-weight: 600;
    line-height: 1.4;
    padding: 0.08rem 0.55rem;
    border-radius: 999px;
    border: 1.5px solid var(--tag, #94a3b8);
    color: var(--tag, #475569);
    background: transparent;
}
/* 4-step progress stepper: Review -> Validation -> Evaluation -> Done. */
.cg-stepper {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    justify-content: flex-end;
    margin-top: 0;  /* sits at the top of the right column now */
    font-size: 0.8rem;
}
.cg-step {
    display: inline-flex;
    align-items: center;
    gap: 0.28rem;
    white-space: nowrap;
}
.cg-step.done { color: #16a34a; }
.cg-step.current { color: var(--cg-primary); font-weight: 700; }
.cg-step.todo { color: var(--cg-text-muted); }
.cg-conn {
    width: 28px;
    height: 2px;
    background: var(--cg-border);
    margin: 0 0.55rem;
}
.cg-conn.done { background: #16a34a; }
/* Completed-task row: a done badge + the evaluation result chip. */
.cg-sub-result {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 0.5rem;
    margin-top: 0;  /* sits at the top of the right column now */
}
.cg-done-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.78rem;
    font-weight: 700;
    color: #16a34a;
    background: rgba(22, 163, 74, 0.1);
    border: 1px solid rgba(22, 163, 74, 0.35);
    border-radius: 999px;
    padding: 0.1rem 0.6rem;
}
.cg-result {
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--cg-primary);
    background: rgba(37, 99, 235, 0.08);
    border: 1px solid rgba(37, 99, 235, 0.3);
    border-radius: 999px;
    padding: 0.1rem 0.6rem;
}
.cg-result-pending {
    color: var(--cg-text-muted);
    background: var(--cg-surface-alt);
    border-color: var(--cg-border);
    font-weight: 600;
}
.cg-sub-empty {
    color: var(--cg-text-muted);
    font-size: 0.9rem;
    padding: 0.5rem 0;
}
/* Narrow screens: fall back to a single stacked column (left-aligned). */
@media (max-width: 640px) {
    .cg-sub-row {
        flex-direction: column;
        align-items: stretch;
    }
    .cg-sub-aside {
        align-items: flex-start;
        text-align: left;
    }
    .cg-stepper,
    .cg-sub-result {
        justify-content: flex-start;
    }
}

/* ---------------------------------------------------------------- */
/* Completed-task expandable charts panel                            */
/* ---------------------------------------------------------------- */
.cg-task-charts {
    margin-top: 0.6rem;
    border-top: 1px dashed var(--cg-border);
    padding-top: 0.1rem;
}
.cg-task-charts > summary {
    list-style: none;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--cg-primary);
    padding: 0.45rem 0 0.2rem;
    user-select: none;
}
.cg-task-charts > summary::-webkit-details-marker { display: none; }
.cg-task-charts > summary::before {
    content: "▸";
    font-size: 0.7rem;
    transition: transform 0.15s ease;
}
.cg-task-charts[open] > summary::before { content: "▾"; }
.cg-task-charts > summary:hover { text-decoration: underline; }
.cg-charts-body {
    padding: 0.4rem 0.1rem 0.2rem;
}
.cg-charts-note {
    font-size: 0.74rem;
    color: var(--cg-text-muted);
    margin-bottom: 0.7rem;
}
.cg-charts-note b { color: var(--cg-text-secondary); font-weight: 600; }
.cg-chart-block { margin-top: 1rem; }
/* Bar + pie side by side (wraps to a column on narrow screens). */
.cg-charts-row {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    gap: 0.6rem 1.75rem;
    margin-top: 1rem;
}
.cg-charts-row .cg-chart-block { margin-top: 0; }
.cg-chart-bar { flex: 1.6 1 340px; min-width: 0; }
/* Right column holding the two stacked donuts (time split + per-tool). */
.cg-chart-pie-col {
    flex: 1 1 300px;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 1rem;
}
.cg-chart-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--cg-text-primary);
    margin-bottom: 0.5rem;
}
.cg-chart-empty {
    font-size: 0.82rem;
    color: var(--cg-text-muted);
    padding: 0.4rem 0;
}
/* shared legend (token bar) + swatches */
.cg-chart-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem 1rem;
    font-size: 0.76rem;
    color: var(--cg-text-secondary);
    margin-bottom: 0.5rem;
}
.cg-lg { display: inline-flex; align-items: center; gap: 0.4rem; }
.cg-sw {
    display: inline-block;
    width: 0.8rem;
    height: 0.8rem;
    border-radius: 3px;
    flex: 0 0 auto;
}
.cg-sw-hatch {
    background-image: repeating-linear-gradient(
        45deg, transparent, transparent 2px,
        rgba(255, 255, 255, 0.6) 2px, rgba(255, 255, 255, 0.6) 3.5px) !important;
    background-blend-mode: normal;
}
.cg-token-svg { display: block; max-width: 760px; }
/* donut + legend (compact — two of these stack in the right column) */
.cg-pie-wrap {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.3rem 0.9rem;
}
.cg-pie-svg { flex: 0 0 auto; }
.cg-pie-legend {
    list-style: none;
    margin: 0;
    padding: 0;
    flex: 1 1 130px;
    min-width: 120px;
}
.cg-pie-li {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.73rem;
    padding: 0.11rem 0;
    color: var(--cg-text-secondary);
}
.cg-pie-name { flex: 1 1 auto; }
.cg-pie-val { color: var(--cg-text-muted); font-variant-numeric: tabular-nums; }

/* Bold dark chart title (黑体). */
.cg-section-label {
    font-size: 1rem;
    font-weight: 700;
    color: var(--cg-text-primary);
    margin: 1rem 0 0.1rem;
    line-height: 1.35;
}
/* Lighter one-line description under the title. */
.cg-section-sub {
    font-size: 0.74rem;
    font-weight: 400;
    color: var(--cg-text-muted);
    margin: 0 0 0.4rem;
    line-height: 1.4;
}

/* View all → button, same outline-pill look as the Refresh button. */
.cg-viewall-btn {
    background: var(--cg-surface) !important;
    color: var(--cg-primary) !important;
    border: 1px solid var(--cg-primary) !important;
    border-radius: var(--cg-radius-sm) !important;
    padding: 0.3rem 0.85rem !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    line-height: 1.2 !important;
    min-height: 0 !important;
    width: auto !important;
    align-self: flex-end;
    margin: 0.3rem 0 0.6rem auto !important;
    transition: all 0.15s ease;
}
.cg-viewall-btn:hover {
    background: var(--cg-primary) !important;
    color: #fff !important;
    box-shadow: var(--cg-shadow-sm) !important;
}

/* ============================================================
   12. RESPONSIVE ADJUSTMENTS
   ============================================================ */
@media (max-width: 768px) {
    #cg-title-banner {
        padding: 1.5rem 1rem 1.3rem;
        flex-direction: column;
        text-align: center;
        gap: 0.9rem;
    }

    #cg-title-banner .cg-title-logo {
        width: 60px;
        height: 60px;
    }

    #cg-title-banner h1 {
        font-size: 1.6rem !important;
    }

    #cg-title-banner .cg-subtitle {
        font-size: 0.9rem;
    }

    #cg-title-banner .cg-badge-row {
        flex-wrap: wrap;
        justify-content: center;
    }

    .tab-buttons button {
        font-size: 14px !important;
        padding: 0.5rem 0.8rem !important;
    }
}

/* ============================================================
   LOG-DETAIL DRAWER (Full table per-cell click)
   Slides in from the right when a task-accuracy cell is clicked;
   wired by css_html_js.py:wireCellClicks() + app.py hidden textbox.
   ============================================================ */
/* Visually-hidden bridge textbox: kept in the DOM (Gradio 5 drops
   visible=False components entirely) but removed from view/layout so the
   JS bridge can set its value and fire its `input` event. */
.cg-vh {
    position: absolute !important;
    width: 1px !important;
    height: 1px !important;
    padding: 0 !important;
    margin: -1px !important;
    overflow: hidden !important;
    clip: rect(0, 0, 0, 0) !important;
    white-space: nowrap !important;
    border: 0 !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
#cg-logpanel-scrim {
    position: fixed;
    inset: 0;
    background: rgba(15, 23, 42, 0.45);
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.25s ease;
    z-index: 1000;
}
#cg-logpanel-scrim.cg-open {
    opacity: 1;
    pointer-events: auto;
}

#cg-logpanel-drawer {
    position: fixed;
    top: 0;
    right: 0;
    height: 100vh;
    width: min(600px, 94vw);
    background: var(--cg-surface);
    border-left: 1px solid var(--cg-border);
    box-shadow: var(--cg-shadow-lg);
    transform: translateX(101%);
    transition: transform 0.28s cubic-bezier(0.4, 0, 0.2, 1);
    z-index: 1001;
    display: flex !important;
    flex-direction: column;
    gap: 0 !important;
    padding: 0 !important;
    overflow: hidden;
}
#cg-logpanel-drawer.cg-open {
    transform: translateX(0);
}

.cg-drawer-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 18px;
    font-weight: 600;
    font-size: 15px;
    color: #ffffff;
    background: var(--cg-gradient);
}
.cg-drawer-close {
    background: rgba(255, 255, 255, 0.18);
    border: none;
    color: #ffffff;
    width: 28px;
    height: 28px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 15px;
    line-height: 1;
    transition: background 0.15s ease;
}
.cg-drawer-close:hover { background: rgba(255, 255, 255, 0.35); }

#cg-logpanel-body {
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
    padding: 16px 18px 48px;
}

.cg-log-empty {
    color: var(--cg-text-muted);
    font-size: 14px;
    text-align: center;
    padding: 40px 20px;
    line-height: 1.6;
}

.cg-log-head { margin-bottom: 14px; }
.cg-log-model {
    font-size: 16px;
    font-weight: 700;
    color: var(--cg-text-primary);
    word-break: break-all;
}
.cg-log-sub {
    font-size: 12.5px;
    color: var(--cg-text-secondary);
    margin-top: 2px;
}

.cg-log-list { display: flex; flex-direction: column; gap: 8px; }

details.cg-q {
    border: 1px solid var(--cg-border);
    border-radius: var(--cg-radius-sm);
    background: var(--cg-surface-alt);
    overflow: hidden;
}
.cg-q-summary {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 9px 12px;
    cursor: pointer;
    font-size: 13px;
    list-style: none;
    user-select: none;
}
.cg-q-summary::-webkit-details-marker { display: none; }
.cg-q-summary:hover { background: var(--cg-surface-hover); }
.cg-qbadge {
    flex: 0 0 auto;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 700;
    color: #ffffff;
}
.cg-qbadge-pass { background: var(--cg-accent); }
.cg-qbadge-fail { background: #dc2626; }
.cg-q-id {
    flex: 0 0 auto;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11px;
    color: var(--cg-text-muted);
}
.cg-q-text {
    flex: 1 1 auto;
    color: var(--cg-text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.cg-q-body {
    padding: 10px 12px 12px;
    border-top: 1px solid var(--cg-border);
}
.cg-q-field { margin-bottom: 10px; }
.cg-q-field > b {
    display: block;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: var(--cg-text-muted);
    margin-bottom: 3px;
}
.cg-q-field pre,
.cg-msg-text {
    white-space: pre-wrap;
    word-break: break-word;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
    line-height: 1.45;
    background: var(--cg-surface);
    border: 1px solid var(--cg-border-light);
    border-radius: 6px;
    padding: 7px 9px;
    margin: 0;
    color: var(--cg-text-primary);
}
.cg-q-meta {
    font-size: 11.5px;
    color: var(--cg-text-secondary);
    margin-bottom: 10px;
}
.cg-chip {
    display: inline-block;
    font-size: 11px;
    padding: 1px 7px;
    border-radius: 10px;
    margin: 2px 4px 2px 0;
}
.cg-chip-ok { background: rgba(13, 148, 136, 0.14); color: var(--cg-accent); }
.cg-chip-no { background: rgba(220, 38, 38, 0.12); color: #dc2626; }

.cg-transcript { display: flex; flex-direction: column; gap: 6px; }
.cg-msg {
    border-left: 3px solid var(--cg-border);
    padding: 4px 0 4px 10px;
}
.cg-msg-human { border-left-color: var(--cg-primary-light); }
.cg-msg-ai { border-left-color: var(--cg-accent); }
.cg-msg-tool { border-left-color: var(--cg-text-muted); }
.cg-msg-role {
    display: block;
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 600;
    color: var(--cg-text-secondary);
    margin-bottom: 3px;
}
.cg-msg-body { display: flex; flex-direction: column; gap: 4px; }
.cg-tool-call { font-size: 12px; color: var(--cg-text-secondary); }
.cg-msg-empty { font-size: 12px; color: var(--cg-text-muted); font-style: italic; }

"""

get_window_url_params = """
    function(url_params) {
        const params = new URLSearchParams(window.location.search);
        url_params = Object.fromEntries(params);
        return url_params;
    }
    """


# Inline <script> injected via gr.Blocks(head=...). Step 1 only:
# inside the column-selector CheckboxGroup, sort labels into Data
# (Average / T / Model / trend cols) and Tasks (the 12 categories)
# and insert a header before each group. Does NOT touch the parent
# layout, the model-family filter, ancestor styling, or anything
# else. Each prior attempt to do those things produced regressions
# (gray seam, half-width card, leak across tabs).
group_columns_head = r"""
<script>
(function () {
  const TASK_COLS = new Set([
    "SMILES Lookup", "Opt (Name)", "Opt (SMILES)",
    "Vib (Name)", "Vib (SMILES)",
    "Thermo (Name)", "Thermo (SMILES)",
    "Dipole (Name)", "Dipole (SMILES)",
    "Energy (Name)", "Energy (SMILES)",
    "Reaction Energy",
  ]);

  const labelOf = el => (el.innerText || el.textContent || "").trim();

  // Find the column-selector CheckboxGroup. Seed on an "Average"
  // label and walk up until the wrapper contains task labels but
  // not unrelated UI labels (Search, org chips).
  function findColumnSelectorContainers() {
    const seeds = Array.from(document.querySelectorAll("label"))
      .filter(l => labelOf(l).startsWith("Average"));
    const out = new Set();
    for (const seed of seeds) {
      let cur = seed.parentElement;
      for (let i = 0; i < 4 && cur; i++) {
        const texts = Array.from(cur.children)
          .filter(c => c.tagName === "LABEL").map(labelOf);
        const hasTask = texts.some(t => TASK_COLS.has(t));
        const polluted = texts.includes("Search")
          || texts.includes("anthropic") || texts.includes("openai");
        if (hasTask && !polluted) { out.add(cur); break; }
        cur = cur.parentElement;
      }
    }
    return Array.from(out);
  }

  // Find the model-family filter: a small CheckboxGroup whose direct
  // <label> children are org names.
  function findModelFamilyContainers() {
    const seeds = Array.from(document.querySelectorAll("label"))
      .filter(l => { const t = labelOf(l); return t === "anthropic" || t === "openai"; });
    const out = new Set();
    for (const seed of seeds) {
      const parent = seed.parentElement;
      if (!parent) continue;
      const texts = Array.from(parent.children)
        .filter(c => c.tagName === "LABEL").map(labelOf);
      if (texts.length === 0 || texts.length > 12) continue;
      if (texts.some(t => TASK_COLS.has(t))) continue;
      out.add(parent);
    }
    return Array.from(out);
  }

  // Hide a Gradio-rendered section label ("Columns to display" /
  // "Model family") sitting just above the given content container.
  function hideSiblingSectionLabel(container, needle) {
    let cur = container.parentElement;
    for (let i = 0; i < 5 && cur; i++) {
      const candidates = cur.querySelectorAll(
        ":scope > label, :scope > span, :scope > .label");
      for (const el of candidates) {
        if (labelOf(el).toLowerCase().includes(needle) && !el.dataset.cgHidden) {
          el.style.display = "none";
          el.dataset.cgHidden = "1";
          return;
        }
      }
      cur = cur.parentElement;
    }
  }

  // Pair each column-selector with the model-family filter whose
  // nearest common ancestor is shortest — they live in the same tab.
  function pairColAndMF(cols, fils) {
    const pairs = [];
    cols.forEach(c => {
      let best = null, bestDepth = Infinity;
      fils.forEach(f => {
        let cur = c.parentElement, d = 0;
        while (cur && d < 12) {
          if (cur.contains(f)) {
            if (d < bestDepth) { best = f; bestDepth = d; }
            break;
          }
          cur = cur.parentElement; d++;
        }
      });
      pairs.push([c, best]);
    });
    return pairs;
  }

  // Lay out a panel: optionally hide its Gradio section label, then
  // sort its direct-child labels into Data + Model-family groups
  // (left) or Tasks-only (right) with sticky group headers.
  function reshape(colSel, mfFilter) {
    colSel.dataset.cgRole = "column-selector";
    if (mfFilter) mfFilter.dataset.cgRole = "model-family-filter";

    hideSiblingSectionLabel(colSel, "columns to display");
    if (mfFilter) hideSiblingSectionLabel(mfFilter, "model family");

    if (colSel.dataset.cgGrouped === "1") return;

    // 1. Snapshot the column-selector's current labels and split.
    const colLabels = Array.from(colSel.children).filter(c => c.tagName === "LABEL");
    const dataLabels = colLabels.filter(l => !TASK_COLS.has(labelOf(l)));
    const taskLabels = colLabels.filter(l => TASK_COLS.has(labelOf(l)));

    // 2. Remove any prior headers we inserted in either panel.
    colSel.querySelectorAll(":scope > .cg-group-header").forEach(h => h.remove());
    if (mfFilter) {
      mfFilter.querySelectorAll(":scope > .cg-group-header").forEach(h => h.remove());
    }

    // 3. LEFT panel (column-selector): Data section, then Model family
    //    section. We move the MF chips here AFTER Data; the model-
    //    family panel on the right becomes the home for Tasks.
    if (dataLabels.length) {
      const h = document.createElement("div");
      h.className = "cg-group-header cg-group-header-data";
      h.textContent = "Data";
      colSel.appendChild(h);
      dataLabels.forEach(l => colSel.appendChild(l));
    }

    if (mfFilter) {
      const mfLabels = Array.from(mfFilter.children).filter(c => c.tagName === "LABEL");
      if (mfLabels.length) {
        const h = document.createElement("div");
        h.className = "cg-group-header cg-group-header-family";
        h.textContent = "Model family";
        colSel.appendChild(h);
        mfLabels.forEach(l => colSel.appendChild(l));
      }

      // 4. RIGHT panel (was model-family): now hosts Tasks.
      if (taskLabels.length) {
        const h = document.createElement("div");
        h.className = "cg-group-header cg-group-header-tasks";
        h.textContent = "Tasks";
        mfFilter.appendChild(h);
        taskLabels.forEach(l => mfFilter.appendChild(l));
      }
    } else if (taskLabels.length) {
      // No right panel found — keep Tasks in the left panel under a
      // header so they aren't lost.
      const h = document.createElement("div");
      h.className = "cg-group-header cg-group-header-tasks";
      h.textContent = "Tasks";
      colSel.appendChild(h);
      taskLabels.forEach(l => colSel.appendChild(l));
    }

    colSel.dataset.cgGrouped = "1";
  }

  // Turn the From/To textboxes into native date pickers, constrained
  // to the dataset's [min, max] eval_date range (carried in a hidden
  // span). Gradio doesn't ship a DatePicker; mutating the input's
  // type is enough because the value still flows through the textbox
  // 'input' event, so Python-side change handlers fire normally.
  function upgradeDateInputs() {
    const pairs = [
      { ids: ["cg-trend-from", "cg-trend-to"], boundsId: "cg-trend-date-bounds" },
      { ids: ["cg-fam-from",   "cg-fam-to"  ], boundsId: "cg-fam-date-bounds"   },
    ];
    pairs.forEach(({ ids, boundsId }) => {
      const bounds = document.getElementById(boundsId);
      const minDate = bounds ? bounds.dataset.min : "";
      const maxDate = bounds ? bounds.dataset.max : "";
      ids.forEach(id => {
        const wrap = document.getElementById(id);
        if (!wrap || wrap.dataset.cgDated === "1") return;
        const input = wrap.querySelector("input, textarea");
        if (!input) return;
        let target = input;
        if (input.tagName === "TEXTAREA") {
          target = document.createElement("input");
          target.value = input.value;
          for (const a of input.attributes) {
            if (a.name === "rows") continue;
            target.setAttribute(a.name, a.value);
          }
          input.replaceWith(target);
        }
        target.type = "date";
        if (minDate) target.min = minDate;
        if (maxDate) target.max = maxDate;
        wrap.dataset.cgDated = "1";
      });
    });
  }

  // Gradio 5.50's PlotlyPlot.svelte does NOT trigger Plotly.Plots.resize
  // on the display:none → display:flex transition that happens when
  // the user switches into Multi-Agent or Trends for the first time.
  // ResizeObserver only fires on real box-size changes, and the parent
  // already had its layout width before the tab was shown.
  //
  // Without a nudge, the chart stays at its initial (700px fallback)
  // width forever. Earlier attempts to fix this combined a resize
  // nudge with a visibility:hidden + reveal-poll, which introduced a
  // "blank → narrow → correct" three-step flash. Lesson: do the
  // resize, do NOT hide the host. The result is a brief sub-perceptual
  // narrow→correct paint (Gradio PR #8740 / Plotly #2769) which is
  // less bad than a permanently-narrow chart.
  function nudgeVisiblePlotlyCharts() {
    if (!window.Plotly) return;
    document.querySelectorAll('.js-plotly-plot').forEach(chart => {
      // offsetParent is null for display:none — skip; resize would do nothing.
      if (!chart.offsetParent) return;
      try { window.Plotly.Plots.resize(chart); } catch (e) {}
    });
  }

  function wireTabClickResize() {
    if (window.__cgTabHooked) return;
    window.__cgTabHooked = true;
    document.addEventListener('click', e => {
      if (!e.target.closest('[role="tab"], .tab-buttons button')) return;
      // Resize twice: once after Gradio flips display, once after the
      // browser has actually laid out the newly-visible plot's parent.
      requestAnimationFrame(() => requestAnimationFrame(nudgeVisiblePlotlyCharts));
      setTimeout(nudgeVisiblePlotlyCharts, 200);
    }, true);
  }

  // --- Log-detail drawer: bridge a Full-table task-cell click to Python -----
  // Clicking a per-task accuracy number opens a right-hand drawer with that
  // (model, workflow, category)'s per-query logs. We compute the column at
  // click time (sort/filter/hide safe) and, if it's a task column, push
  // "{workflow}|||{full_model}|||{colName}" into a hidden Gradio textbox whose
  // .input() handler (app.py) renders the panel HTML.
  function setNativeValue(el, value) {
    // Svelte/Gradio only reacts to the value setter + a bubbling 'input'
    // event, not to a plain el.value = ... assignment.
    const proto = el.tagName === "TEXTAREA"
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, "value").set;
    setter.call(el, value);
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function openLogDrawer() {
    const d = document.getElementById("cg-logpanel-drawer");
    const s = document.getElementById("cg-logpanel-scrim");
    if (d) d.classList.add("cg-open");
    if (s) s.classList.add("cg-open");
  }
  function closeLogDrawer() {
    const d = document.getElementById("cg-logpanel-drawer");
    const s = document.getElementById("cg-logpanel-scrim");
    if (d) d.classList.remove("cg-open");
    if (s) s.classList.remove("cg-open");
  }

  function wireCellClicks() {
    if (window.__cgCellHooked) return;
    window.__cgCellHooked = true;

    document.addEventListener("click", e => {
      // Close controls first.
      if (e.target.closest("#cg-logpanel-close") ||
          e.target.id === "cg-logpanel-scrim") {
        closeLogDrawer();
        return;
      }

      const td = e.target.closest("td");
      if (!td) return;
      const scope = td.closest(".cg-leaderboard");
      if (!scope) return;                       // only the Full-table leaderboard
      const table = td.closest("table");
      const row = td.closest("tr");
      if (!table || !row) return;

      const colIdx = Array.from(row.children).indexOf(td);
      if (colIdx < 0) return;
      const heads = Array.from(table.querySelectorAll("thead th"));
      const colName = heads[colIdx] ? labelOf(heads[colIdx]) : "";
      if (!TASK_COLS.has(colName)) return;      // ignore Model / rank / avg / etc.

      // Model = the cell under the "Model" header; take its link target.
      const modelIdx = heads.findIndex(h => labelOf(h) === "Model");
      if (modelIdx < 0) return;
      const modelCell = row.children[modelIdx];
      if (!modelCell) return;
      const a = modelCell.querySelector("a[href]");
      let fullModel = a
        ? a.getAttribute("href").replace(/^https?:\/\/huggingface\.co\//, "")
        : labelOf(modelCell);
      fullModel = (fullModel || "").trim();
      if (!fullModel) return;

      // Workflow from the leaderboard's elem_id (cg-single* / cg-multi*).
      const wf = scope.closest('[id^="cg-single"]') ? "single_agent"
               : scope.closest('[id^="cg-multi"]')  ? "multi_agent"
               : (scope.id && scope.id.indexOf("cg-multi") === 0 ? "multi_agent"
                  : scope.id && scope.id.indexOf("cg-single") === 0 ? "single_agent"
                  : null);
      if (!wf) return;

      const input = document.querySelector(
        "#cg-logpanel-input textarea, #cg-logpanel-input input");
      if (!input) return;
      setNativeValue(input, wf + "|||" + fullModel + "|||" + colName);
      openLogDrawer();
    }, true);

    document.addEventListener("keydown", e => {
      if (e.key === "Escape") closeLogDrawer();
    });
  }

  function pass() {
    const cols = findColumnSelectorContainers();
    const fils = findModelFamilyContainers();
    pairColAndMF(cols, fils).forEach(([c, f]) => reshape(c, f));
    upgradeDateInputs();
    wireTabClickResize();
    wireCellClicks();
  }

  let pending = false;
  function schedule() {
    if (pending) return;
    pending = true;
    setTimeout(() => {
      pending = false;
      try { pass(); } catch (e) { console.warn("[cg-group]", e); }
    }, 100);
  }

  function start() {
    pass();
    new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
</script>
"""
