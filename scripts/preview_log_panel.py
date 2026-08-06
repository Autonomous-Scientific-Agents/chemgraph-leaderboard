#!/usr/bin/env python
"""Render the log panel to a standalone HTML file for eyeballing.

Booting the whole Gradio app just to look at the panel is slow and pulls the eval
datasets from HuggingFace. This writes the real panel markup wrapped in the real
``custom_css`` instead, so the modal can be inspected offline in any browser.

The fixed-position/hidden modal state is neutralised inline so the panel sits in
normal page flow and the whole thing is scrollable::

    HF_HUB_OFFLINE=1 python scripts/preview_log_panel.py \\
        "multi_agent|||argo:gpt-5.6-terra|||Reaction Energy" -o /tmp/panel.html

Pass ``--dark`` to check the dark-mode tokens.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.display.css_html_js import custom_css  # noqa: E402
from src.leaderboard.logs import render_log_panel  # noqa: E402

try:  # only present once the head constant lands
    from src.leaderboard.logs import LOG_PANEL_HEAD_HTML
except ImportError:  # pragma: no cover - transitional
    LOG_PANEL_HEAD_HTML = (
        '<div class="cg-drawer-head"><span class="cg-drawer-title">Log details</span>'
        '<button id="cg-logpanel-close" class="cg-drawer-close">&#10005;</button></div>'
    )

# Lay the modal out in page flow instead of fixed+hidden, so the whole panel is
# visible and scrollable in a plain browser tab.
_UNPIN = (
    "position:static;transform:none;opacity:1;visibility:visible;"
    "max-height:none;width:min(1200px,98vw);margin:20px auto;pointer-events:auto"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("payload", nargs="+", help='e.g. "single_agent|||openai/gpt-4.1|||SMILES Lookup"')
    ap.add_argument("-o", "--out", default="panel.html")
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args()

    blocks = []
    for payload in args.payload:
        t0 = time.time()
        body = render_log_panel(payload)
        ms = (time.time() - t0) * 1000
        print(f"{len(body):>9,} chars  {ms:6.1f} ms  {payload}")
        blocks.append(
            f'<div id="cg-logpanel-drawer" class="cg-drawer cg-open" style="{_UNPIN}">'
            f'{LOG_PANEL_HEAD_HTML}<div id="cg-logpanel-body">{body}</div></div>'
        )

    theme = "dark" if args.dark else "light"
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{custom_css}</style>"
        f"<style>body{{margin:0;background:{'#0f172a' if args.dark else '#eef2f7'};"
        "font-family:ui-sans-serif,system-ui,sans-serif}</style>"
        f"</head><body class='{theme}'>{''.join(blocks)}</body></html>"
    )
    out = Path(args.out)
    out.write_text(page)
    print(f"-> {out.resolve()}  ({len(page):,} chars, {theme})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
