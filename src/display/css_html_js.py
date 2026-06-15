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
    background: var(--cg-gradient);
    border-radius: var(--cg-radius);
    padding: 2rem 2.5rem 1.8rem;
    margin-bottom: 1rem;
    box-shadow: var(--cg-shadow-lg);
    text-align: center;
    position: relative;
    overflow: hidden;
}

#cg-title-banner::before {
    content: "";
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}

#cg-title-banner::after {
    content: "";
    position: absolute;
    bottom: -40%;
    left: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
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
    justify-content: center;
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
    background: var(--cg-gradient) !important;
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
#cg-trends-header {
    background: var(--cg-surface) !important;
    border: 1px solid var(--cg-border) !important;
    border-radius: var(--cg-radius) !important;
    padding: 1rem 1.5rem !important;
    box-shadow: var(--cg-shadow-sm) !important;
    margin-bottom: 0.5rem !important;
}

#cg-trends-header h3 {
    color: var(--cg-primary) !important;
    font-weight: 600 !important;
    margin-bottom: 0.3rem !important;
}

/* Trends controls — one bordered card with three zones (Data / View /
   Actions) separated by thin vertical dividers. */
#cg-trend-controls {
    background: var(--cg-surface) !important;
    border: 1px solid var(--cg-border) !important;
    border-radius: var(--cg-radius-sm) !important;
    padding: 0.9rem 1rem !important;
    box-shadow: var(--cg-shadow-sm) !important;
    gap: 0 !important;
    align-items: stretch !important;
}

#cg-trend-controls .cg-zone {
    padding: 0 1rem !important;
    border-left: 1px solid var(--cg-border-light);
    display: flex !important;
    flex-direction: column !important;
    gap: 0.55rem !important;
    min-width: 0 !important;
}
#cg-trend-controls .cg-zone:first-child {
    border-left: 0;
    padding-left: 0 !important;
}
#cg-trend-controls .cg-zone:last-child {
    padding-right: 0 !important;
}

/* Zone-internal blocks blend into the card. */
#cg-trend-controls .cg-zone .block,
#cg-trend-controls .cg-zone .form {
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
#cg-trend-controls .cg-zone .container > span.svelte-g2oxp3 {
    font-size: 0.875rem !important;   /* 14px @ default root */
    color: var(--cg-text-secondary) !important;
    font-weight: 500 !important;
    line-height: 1.2 !important;
}

/* The radio (Date range) — chip-row instead of stacked. */
#cg-trend-controls .cg-zone-view .wrap {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 0.35rem !important;
}

/* Models multiselect — single-row pill that keeps the frame a few
   pixels taller than the chips inside so they don't touch the top
   or bottom border. Scoped to dropdowns that have .token children
   (the multiselect) so the single-value Workflow dropdown keeps its
   default compact height. */
#cg-trend-controls .cg-zone-data .gradio-dropdown[data-testid="dropdown"] .wrap:has(.token) {
    min-height: 51px !important;
    padding: 0 0.4rem !important;
    overflow: hidden !important;
}
#cg-trend-controls .cg-zone-data .wrap:has(.token):hover,
#cg-trend-controls .cg-zone-data .wrap:has(.token):focus-within {
    overflow-x: auto !important;
}
#cg-trend-controls .cg-zone-data .wrap:has(.token) .wrap-inner {
    flex-wrap: nowrap !important;
    overflow: visible !important;
    padding: 8px 6px !important;
    align-items: center !important;
}
#cg-trend-controls .cg-zone-data .wrap:has(.token) .wrap-inner .token {
    flex-shrink: 0 !important;
}

/* Refresh button — compact outline pill using the theme primary. */
#cg-trend-controls .cg-zone-actions button {
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
#cg-trend-controls .cg-zone-actions button:hover {
    background: var(--cg-primary) !important;
    color: #fff !important;
    box-shadow: var(--cg-shadow-sm) !important;
}
#cg-trend-controls .cg-zone-actions button:active {
    transform: translateY(1px);
}

#cg-trend-chart {
    border: 1px solid var(--cg-border) !important;
    border-radius: var(--cg-radius) !important;
    box-shadow: var(--cg-shadow-sm) !important;
    overflow: hidden !important;
    padding: 0.5rem !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.5rem !important;
    background: var(--cg-surface) !important;
}

#cg-trend-summary-label h3 {
    color: var(--cg-primary) !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
}

#cg-trend-summary {
    border: 1px solid var(--cg-border) !important;
    border-radius: var(--cg-radius) !important;
    box-shadow: var(--cg-shadow-sm) !important;
    overflow: hidden !important;
}

#cg-trend-summary table thead th {
    background: var(--cg-surface-alt) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
    padding: 0.65rem 0.8rem !important;
    border-bottom: 2px solid var(--cg-border) !important;
}

#cg-trend-summary table tbody tr:nth-child(even) {
    background: var(--cg-surface-alt) !important;
}

#cg-trend-summary table tbody tr:hover td {
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
   12. RESPONSIVE ADJUSTMENTS
   ============================================================ */
@media (max-width: 768px) {
    #cg-title-banner {
        padding: 1.5rem 1rem 1.3rem;
    }

    #cg-title-banner h1 {
        font-size: 1.6rem !important;
    }

    #cg-title-banner .cg-subtitle {
        font-size: 0.9rem;
    }

    #cg-title-banner .cg-badge-row {
        flex-wrap: wrap;
    }

    .tab-buttons button {
        font-size: 14px !important;
        padding: 0.5rem 0.8rem !important;
    }
}

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

  function pass() {
    const cols = findColumnSelectorContainers();
    const fils = findModelFamilyContainers();
    pairColAndMF(cols, fils).forEach(([c, f]) => reshape(c, f));
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
