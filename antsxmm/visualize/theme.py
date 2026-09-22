"""Theme and HTML template engine for antsxmm visual reporting.

Provides a self-contained, publication-grade CSS3 dark/light variable system,
KPI metric cards, accessible tabs, and an offline HTML document bundler with
zero external CDN dependencies.
"""

from __future__ import annotations

import html
import uuid
from typing import Any, Sequence


def get_theme_css() -> str:
    """Return the complete inline CSS3 stylesheet for antsxmm reports."""
    return """
:root {
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 16px;
  --transition-speed: 0.2s;
}

/* Light Theme (Default fallback) */
:root, :root[data-theme="light"] {
  --bg-primary: #f8fafc;
  --bg-secondary: #ffffff;
  --bg-card: #ffffff;
  --bg-card-hover: #f1f5f9;
  --bg-subtle: #f1f5f9;
  --text-primary: #0f172a;
  --text-secondary: #334155;
  --text-muted: #64748b;
  --border-color: #e2e8f0;
  --border-hover: #cbd5e1;
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
  --accent-primary: #0284c7;
  --accent-hover: #0369a1;
  --accent-subtle: #e0f2fe;
  --tab-active-border: #0284c7;

  /* Status Colors */
  --status-normal-bg: #ecfdf5;
  --status-normal-text: #065f46;
  --status-normal-border: #a7f3d0;
  --status-warning-bg: #fffbeb;
  --status-warning-text: #92400e;
  --status-warning-border: #fde68a;
  --status-danger-bg: #fff1f2;
  --status-danger-text: #9f1239;
  --status-danger-border: #fecdd3;
  --status-info-bg: #eff6ff;
  --status-info-text: #1e40af;
  --status-info-border: #bfdbfe;
}

/* Dark Clinical Mode */
:root[data-theme="dark"] {
  --bg-primary: #0b0f19;
  --bg-secondary: #111827;
  --bg-card: #162032;
  --bg-card-hover: #1e293b;
  --bg-subtle: #1e293b;
  --text-primary: #f8fafc;
  --text-secondary: #cbd5e1;
  --text-muted: #94a3b8;
  --border-color: #27354a;
  --border-hover: #3b4d66;
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.4);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.5), 0 2px 4px -2px rgba(0, 0, 0, 0.4);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.6), 0 4px 6px -4px rgba(0, 0, 0, 0.4);
  --accent-primary: #38bdf8;
  --accent-hover: #7dd3fc;
  --accent-subtle: #0c4a6e;
  --tab-active-border: #38bdf8;

  /* Status Colors - Clinical High Contrast */
  --status-normal-bg: #064e3b;
  --status-normal-text: #6ee7b7;
  --status-normal-border: #059669;
  --status-warning-bg: #78350f;
  --status-warning-text: #fcd34d;
  --status-warning-border: #d97706;
  --status-danger-bg: #881337;
  --status-danger-text: #fda4af;
  --status-danger-border: #e11d48;
  --status-info-bg: #1e3a8a;
  --status-info-text: #93c5fd;
  --status-info-border: #2563eb;
}

/* Reset & Base */
*, *::before, *::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: var(--font-sans);
  font-size: 14px;
  line-height: 1.5;
  color: var(--text-primary);
  background-color: var(--bg-primary);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  padding-bottom: 60px;
  transition: background-color var(--transition-speed) ease, color var(--transition-speed) ease;
}

/* Layout */
.report-container {
  max-width: 1440px;
  margin: 0 auto;
  padding: 24px 32px;
}

/* Header */
.report-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 20px;
  margin-bottom: 24px;
  border-bottom: 1px solid var(--border-color);
  flex-wrap: wrap;
  gap: 16px;
}

.report-header-titles {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.report-title {
  font-size: 26px;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--text-primary);
}

.report-subtitle {
  font-size: 14px;
  color: var(--text-muted);
}

.report-controls {
  display: flex;
  align-items: center;
  gap: 12px;
}

/* Theme Toggle Button */
.theme-toggle-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background-color: var(--bg-card);
  color: var(--text-secondary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 8px 14px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all var(--transition-speed) ease;
  user-select: none;
}

.theme-toggle-btn:hover {
  background-color: var(--bg-card-hover);
  border-color: var(--border-hover);
  color: var(--text-primary);
}

.theme-toggle-icon {
  width: 16px;
  height: 16px;
  fill: currentColor;
}

/* KPI Metrics Grid */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}

.kpi-card {
  background-color: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 16px 20px;
  box-shadow: var(--shadow-sm);
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  position: relative;
  transition: transform var(--transition-speed) ease, box-shadow var(--transition-speed) ease, border-color var(--transition-speed) ease;
}

.kpi-card:hover {
  box-shadow: var(--shadow-md);
  border-color: var(--border-hover);
}

.kpi-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 8px;
  gap: 8px;
}

.kpi-label {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
}

.kpi-value-container {
  display: flex;
  align-items: baseline;
  gap: 6px;
}

.kpi-value {
  font-family: var(--font-mono);
  font-size: 26px;
  font-weight: 700;
  color: var(--text-primary);
  line-height: 1.1;
}

.kpi-unit {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-muted);
}

/* Status Badges */
.badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 600;
  border-radius: 9999px;
  line-height: 1.4;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  border: 1px solid transparent;
}

.badge-normal {
  background-color: var(--status-normal-bg);
  color: var(--status-normal-text);
  border-color: var(--status-normal-border);
}

.badge-warning {
  background-color: var(--status-warning-bg);
  color: var(--status-warning-text);
  border-color: var(--status-warning-border);
}

.badge-danger {
  background-color: var(--status-danger-bg);
  color: var(--status-danger-text);
  border-color: var(--status-danger-border);
}

.badge-info {
  background-color: var(--status-info-bg);
  color: var(--status-info-text);
  border-color: var(--status-info-border);
}

.badge-neutral {
  background-color: var(--bg-subtle);
  color: var(--text-secondary);
  border-color: var(--border-color);
}

/* Card Component */
.card {
  background-color: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-sm);
  margin-bottom: 24px;
  overflow: hidden;
  transition: border-color var(--transition-speed) ease;
}

.card-header {
  padding: 16px 20px;
  border-bottom: 1px solid var(--border-color);
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: var(--bg-card);
  gap: 12px;
}

.card-title-group {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.card-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
}

.card-subtitle {
  font-size: 12px;
  color: var(--text-muted);
}

.card-body {
  padding: 20px;
}

/* Tabs System */
.tabs-container {
  margin-bottom: 24px;
}

.tab-nav {
  display: flex;
  gap: 4px;
  border-bottom: 2px solid var(--border-color);
  margin-bottom: 20px;
  overflow-x: auto;
  white-space: nowrap;
  scrollbar-width: thin;
}

.tab-btn {
  background: none;
  border: none;
  color: var(--text-muted);
  font-family: var(--font-sans);
  font-size: 14px;
  font-weight: 600;
  padding: 10px 18px;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  margin-bottom: -2px;
  transition: all var(--transition-speed) ease;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  user-select: none;
}

.tab-btn:hover {
  color: var(--text-primary);
  background-color: var(--bg-card-hover);
  border-top-left-radius: var(--radius-sm);
  border-top-right-radius: var(--radius-sm);
}

.tab-btn:focus-visible {
  outline: 2px solid var(--accent-primary);
  outline-offset: -2px;
}

.tab-btn.active {
  color: var(--accent-primary);
  border-bottom-color: var(--tab-active-border);
  background: transparent;
}

.tab-panel {
  display: none;
}

.tab-panel.active {
  display: block;
  animation: fadeIn 0.2s ease-in-out;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(2px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Image & Plot Containers */
.img-container {
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: var(--bg-primary);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-color);
  padding: 12px;
  overflow: hidden;
  box-sizing: border-box;
}

.img-responsive {
  max-width: 100%;
  height: auto;
  display: block;
  border-radius: var(--radius-sm);
}

/* Tooltip helper */
[data-tooltip] {
  position: relative;
  cursor: help;
}

[data-tooltip]::after {
  content: attr(data-tooltip);
  position: absolute;
  bottom: 125%;
  left: 50%;
  transform: translateX(-50%);
  background: var(--bg-secondary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
  padding: 4px 8px;
  font-size: 11px;
  border-radius: var(--radius-sm);
  white-space: nowrap;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.15s ease;
  box-shadow: var(--shadow-md);
  z-index: 100;
}

[data-tooltip]:hover::after {
  opacity: 1;
}

/* Responsive Grid Helpers */
.grid-2 {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
  gap: 20px;
}

.grid-3 {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 20px;
}

@media (max-width: 768px) {
  .report-container {
    padding: 16px;
  }
  .grid-2, .grid-3 {
    grid-template-columns: 1fr;
  }
  .report-header {
    flex-direction: column;
    align-items: flex-start;
  }
}
"""


def get_theme_js() -> str:
    """Return inline JavaScript for theme toggling and accessible tab navigation."""
    return """
(function() {
  // Theme Switching Logic with LocalStorage and prefers-color-scheme
  const THEME_KEY = 'antsxmm-theme';

  function getPreferredTheme() {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === 'dark' || saved === 'light') {
      return saved;
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    updateThemeToggleButtons(theme);
  }

  function updateThemeToggleButtons(theme) {
    document.querySelectorAll('.theme-toggle-btn').forEach(btn => {
      const label = btn.querySelector('.theme-toggle-label');
      if (label) {
        label.textContent = theme === 'dark' ? 'Light Mode' : 'Dark Mode';
      }
      btn.setAttribute('aria-label', 'Switch to ' + (theme === 'dark' ? 'light' : 'dark') + ' mode');
    });
  }

  window.toggleTheme = function() {
    const current = document.documentElement.getAttribute('data-theme') || getPreferredTheme();
    const next = current === 'dark' ? 'light' : 'dark';
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
  };

  // Listen for OS theme changes if user hasn't overridden
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
    if (!localStorage.getItem(THEME_KEY)) {
      applyTheme(e.matches ? 'dark' : 'light');
    }
  });

  // Tab Navigation with Keyboard Accessibility (ARIA)
  function initTabs() {
    document.querySelectorAll('.tabs-container').forEach(container => {
      const tablist = container.querySelector('[role="tablist"]');
      if (!tablist) return;

      const tabs = Array.from(tablist.querySelectorAll('[role="tab"]'));
      const panels = Array.from(container.querySelectorAll('[role="tabpanel"]'));

      function activateTab(tab, focus = true) {
        tabs.forEach(t => {
          t.classList.remove('active');
          t.setAttribute('aria-selected', 'false');
          t.setAttribute('tabindex', '-1');
        });
        panels.forEach(p => {
          p.classList.remove('active');
          p.hidden = true;
        });

        tab.classList.add('active');
        tab.setAttribute('aria-selected', 'true');
        tab.setAttribute('tabindex', '0');
        if (focus) tab.focus();

        const panelId = tab.getAttribute('aria-controls');
        const targetPanel = container.querySelector('#' + panelId);
        if (targetPanel) {
          targetPanel.classList.add('active');
          targetPanel.hidden = false;
        }
      }

      tabs.forEach((tab, index) => {
        tab.addEventListener('click', () => activateTab(tab, false));

        tab.addEventListener('keydown', e => {
          let targetIndex = null;
          if (e.key === 'ArrowRight') {
            targetIndex = (index + 1) % tabs.length;
          } else if (e.key === 'ArrowLeft') {
            targetIndex = (index - 1 + tabs.length) % tabs.length;
          } else if (e.key === 'Home') {
            targetIndex = 0;
          } else if (e.key === 'End') {
            targetIndex = tabs.length - 1;
          }

          if (targetIndex !== null) {
            e.preventDefault();
            activateTab(tabs[targetIndex], true);
          }
        });
      });
    });
  }

  // Initialize on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      applyTheme(getPreferredTheme());
      initTabs();
    });
  } else {
    applyTheme(getPreferredTheme());
    initTabs();
  }
})();
"""


def render_badge(text: str, status: str = "normal") -> str:
    """Render an accessible status badge.

    Parameters
    ----------
    text : str
        Badge label text.
    status : str
        Status class: "normal", "warning", "danger", "info", "neutral".
    """
    valid_status = status.lower() if status.lower() in {"normal", "warning", "danger", "info", "neutral"} else "neutral"
    clean_text = html.escape(str(text))
    return f'<span class="badge badge-{valid_status}">{clean_text}</span>'


def render_kpi_card(
    label: str,
    value: Any,
    unit: str = "",
    status: str = "normal",
    tooltip: str = "",
) -> str:
    """Render a publication-grade KPI metric card.

    Parameters
    ----------
    label : str
        Metric description or label.
    value : Any
        Numerical or text metric value.
    unit : str, optional
        Unit of measurement (e.g. "mm", "cm3", "%").
    status : str, optional
        Status: "normal", "warning", "danger", "info", "neutral".
    tooltip : str, optional
        Informational tooltip text for clinical hover context.
    """
    if value is None:
        val_str = "N/A"
    elif isinstance(value, float):
        val_str = f"{value:.3g}" if (abs(value) < 0.01 or abs(value) >= 10000) else f"{value:.2f}"
    else:
        val_str = str(value)

    label_esc = html.escape(str(label))
    val_esc = html.escape(val_str)
    unit_esc = html.escape(str(unit))
    tooltip_attr = f' data-tooltip="{html.escape(tooltip)}"' if tooltip else ""
    badge_html = render_badge(status.capitalize(), status=status)

    unit_html = f'<span class="kpi-unit">{unit_esc}</span>' if unit_esc else ""

    return f"""<div class="kpi-card"{tooltip_attr}>
  <div class="kpi-header">
    <span class="kpi-label">{label_esc}</span>
    {badge_html}
  </div>
  <div class="kpi-value-container">
    <span class="kpi-value">{val_esc}</span>
    {unit_html}
  </div>
</div>"""


def render_card(
    title: str,
    content_html: str,
    subtitle: str = "",
    badge: str = "",
    card_id: str | None = None,
    extra_class: str = "",
) -> str:
    """Render a container card with header, subtitle, optional badge, and body.

    Parameters
    ----------
    title : str
        Card header title.
    content_html : str
        HTML content inside card body.
    subtitle : str, optional
        Secondary explanatory text.
    badge : str, optional
        Badge text or status.
    card_id : str, optional
        HTML element id attribute.
    extra_class : str, optional
        Additional CSS classes.
    """
    id_attr = f' id="{html.escape(card_id)}"' if card_id else ""
    cls_attr = f"card {extra_class}".strip()
    title_esc = html.escape(title)
    subtitle_html = f'<span class="card-subtitle">{html.escape(subtitle)}</span>' if subtitle else ""
    badge_html = f'<div class="card-badge">{render_badge(badge, "info")}</div>' if badge else ""

    return f"""<div class="{cls_attr}"{id_attr}>
  <div class="card-header">
    <div class="card-title-group">
      <h3 class="card-title">{title_esc}</h3>
      {subtitle_html}
    </div>
    {badge_html}
  </div>
  <div class="card-body">
    {content_html}
  </div>
</div>"""


def render_tabs(
    tabs: Sequence[tuple[str, str, str]] | Sequence[dict[str, str]],
    active_tab: str | None = None,
    tab_group_id: str | None = None,
) -> str:
    """Render an accessible, tabbed interface container.

    Parameters
    ----------
    tabs : list of tuple or dict
        Tabs list. If tuples, each element is (tab_id, label, content_html).
        If dicts, keys are 'id', 'label', 'content'.
    active_tab : str, optional
        ID of tab to activate by default. If None, first tab is active.
    tab_group_id : str, optional
        Unique ID for tab group container.
    """
    parsed_tabs: list[tuple[str, str, str]] = []
    for item in tabs:
        if isinstance(item, dict):
            parsed_tabs.append((str(item["id"]), str(item["label"]), str(item.get("content", ""))))
        else:
            parsed_tabs.append((str(item[0]), str(item[1]), str(item[2])))

    if not parsed_tabs:
        return ""

    group_id = tab_group_id or f"tabgroup-{uuid.uuid4().hex[:8]}"

    if active_tab is None or not any(t[0] == active_tab for t in parsed_tabs):
        active_id = parsed_tabs[0][0]
    else:
        active_id = active_tab

    buttons_html: list[str] = []
    panels_html: list[str] = []

    for tab_id, label, content in parsed_tabs:
        is_active = tab_id == active_id
        active_cls = " active" if is_active else ""
        selected = "true" if is_active else "false"
        tabindex = "0" if is_active else "-1"
        hidden_attr = "" if is_active else " hidden"

        tab_btn_id = f"tab-{group_id}-{tab_id}"
        panel_id = f"panel-{group_id}-{tab_id}"

        buttons_html.append(
            f'<button type="button" role="tab" class="tab-btn{active_cls}" '
            f'id="{tab_btn_id}" aria-controls="{panel_id}" '
            f'aria-selected="{selected}" tabindex="{tabindex}">{html.escape(label)}</button>'
        )

        panels_html.append(
            f'<div role="tabpanel" id="{panel_id}" aria-labelledby="{tab_btn_id}" '
            f'class="tab-panel{active_cls}"{hidden_attr}>\n{content}\n</div>'
        )

    nav_bar = f'<div class="tab-nav" role="tablist" aria-label="Sections">\n{"".join(buttons_html)}\n</div>'
    panels_body = f'<div class="tab-panels">\n{"".join(panels_html)}\n</div>'

    return f'<div class="tabs-container" id="{group_id}">\n{nav_bar}\n{panels_body}\n</div>'


def build_html_document(
    title: str,
    body_html: str,
    theme: str = "auto",
    extra_css: str = "",
    extra_js: str = "",
    header_title: str | None = None,
    subtitle: str | None = None,
) -> str:
    """Build a complete, standalone, self-contained HTML document.

    Guarantees ZERO external CDN dependencies (offline/air-gapped compatible).
    Includes embedded CSS3 variables, theme switcher, and accessible tab system.

    Parameters
    ----------
    title : str
        Document <title> tag.
    body_html : str
        Inner HTML content.
    theme : str, optional
        "auto" (OS preference + localStorage), "dark", or "light".
    extra_css : str, optional
        Additional inline CSS styles.
    extra_js : str, optional
        Additional inline JavaScript.
    header_title : str, optional
        Title in page header banner. Defaults to title.
    subtitle : str, optional
        Secondary subtitle in header banner.
    """
    disp_title = header_title or title
    theme_attr = f' data-theme="{theme}"' if theme in {"dark", "light"} else ""

    theme_css = get_theme_css()
    theme_js = get_theme_js()

    subtitle_html = (
        f'<div class="report-subtitle">{html.escape(subtitle)}</div>'
        if subtitle
        else ""
    )

    # Sun/Moon SVG Icon for Theme Switcher (Zero external assets)
    toggle_icon_svg = (
        '<svg class="theme-toggle-icon" viewBox="0 0 24 24" aria-hidden="true">'
        '<path d="M12 3a9 9 0 1 0 9 9c0-.46-.04-.92-.1-1.36a5.389 5.389 0 0 1-4.4 2.26 '
        '9 9 0 0 1-9-9c0-1.81.54-3.5 1.46-4.9A8.96 8.96 0 0 0 12 3z"/>'
        '</svg>'
    )

    return f"""<!DOCTYPE html>
<html lang="en"{theme_attr}>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
{theme_css}
{extra_css}
  </style>
</head>
<body>
  <div class="report-container">
    <header class="report-header">
      <div class="report-header-titles">
        <h1 class="report-title">{html.escape(disp_title)}</h1>
        {subtitle_html}
      </div>
      <div class="report-controls">
        <button type="button" class="theme-toggle-btn" onclick="window.toggleTheme()" aria-label="Toggle dark/light theme">
          {toggle_icon_svg}
          <span class="theme-toggle-label">Theme</span>
        </button>
      </div>
    </header>
    <main>
{body_html}
    </main>
  </div>
  <script>
{theme_js}
{extra_js}
  </script>
</body>
</html>"""
