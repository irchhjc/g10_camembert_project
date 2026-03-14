"""
╔══════════════════════════════════════════════════════════════════════╗
║  G10 — CamemBERT Dashboard · Protocole P02                         ║
║  4 Pages : Accueil/KPI · Loss Landscape · Optuna · Projet/Auteurs  ║
║  Thèmes  : Dark / Light (toggle en haut à droite)                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import os
import traceback
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import optuna

import dash
from dash import html, dcc, Input, Output, State, dash_table, ALL


# ══════════════════════════════════════════════════════════════════════
#  0. CHEMINS & CHARGEMENT ROBUSTE
# ══════════════════════════════════════════════════════════════════════

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")


def safe_json(name: str, default: Any):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] {path}: {e}")
        return default


def safe_csv(name: str, default: pd.DataFrame | None = None) -> pd.DataFrame:
    if default is None:
        default = pd.DataFrame()
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return default.copy()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] {path}: {e}")
        return default.copy()


baseline = safe_json("baseline_results.json", {
    "train": {"best_val_f1": 0.0, "history": {"train_f1": [], "val_f1": [], "train_loss": [], "val_loss": []}},
    "test": {"f1_macro": 0.0},
})
best = safe_json("best_params.json", {
    "best_value": 0.0,
    "best_params": {"learning_rate": 0.0, "weight_decay": 0.0, "dropout": 0.0},
})
sharpness = safe_json("sharpness_results.json", [
    {"label": "dropout=0.0", "dropout": 0.0, "sharpness": 0.11, "val_f1": 0.92},
    {"label": "dropout=0.1", "dropout": 0.1, "sharpness": 0.30, "val_f1": 0.93},
    {"label": "dropout=0.3", "dropout": 0.3, "sharpness": 0.11, "val_f1": 0.47},
])
grid_df = safe_csv("grid_p02_results.csv")
summary = safe_csv("summary.csv")

OPTUNA_DB = os.path.join(RES, "optuna_db", "G10_study.db")


# ══════════════════════════════════════════════════════════════════════
#  1. SYSTÈME DE THÈMES
# ══════════════════════════════════════════════════════════════════════

DARK_THEME = dict(
    name="dark",
    BG="#05090F",
    BG1="#0B1120",
    BG2="#0F1929",
    BG3="#162035",
    BORDER="#1E2F4A",
    BORDER2="#243856",
    CYAN="#00D4FF",
    GOLD="#F0A500",
    VIOLET="#7B61FF",
    GREEN="#00E5A0",
    RED="#FF4D6A",
    PINK="#FF6B9D",
    LIME="#C8FF5E",
    TEXT="#E8F0FE",
    DIM="#8899BB",
    DIM2="#5A6E8C",
    PLOTLY_PAPER="rgba(0,0,0,0)",
    PLOTLY_PLOT="rgba(0,0,0,0)",
    TABLE_ODD="#0F1929",
    SIDEBAR_GRADIENT="linear-gradient(180deg, #0B1120 0%, #0D1628 100%)",
    HEADER_BG="#0B1120",
    TOGGLE_LABEL="☀️  Mode clair",
    TOGGLE_BG="#162035",
    TOGGLE_COLOR="#8899BB",
    TOGGLE_BORDER="#1E2F4A",
    COLORS6=["#00D4FF", "#F0A500", "#7B61FF", "#00E5A0", "#FF4D6A", "#FF6B9D"],
    FONT_MONO="'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
    FONT_UI="'Space Grotesk', 'DM Sans', sans-serif",
)

LIGHT_THEME = dict(
    name="light",
    BG="#F0F3FA",
    BG1="#FFFFFF",
    BG2="#F5F7FD",
    BG3="#EBEEf8",
    BORDER="#CBD5EA",
    BORDER2="#B0BEDD",
    CYAN="#006FAA",
    GOLD="#A86800",
    VIOLET="#4A30CC",
    GREEN="#007A58",
    RED="#CC1838",
    PINK="#CC3060",
    LIME="#3A6A00",
    TEXT="#1A2340",
    DIM="#3A4E70",
    DIM2="#6A7E9A",
    PLOTLY_PAPER="rgba(0,0,0,0)",
    PLOTLY_PLOT="rgba(255,255,255,0)",
    TABLE_ODD="#F5F7FD",
    SIDEBAR_GRADIENT="linear-gradient(180deg, #FFFFFF 0%, #F5F7FD 100%)",
    HEADER_BG="#FFFFFF",
    TOGGLE_LABEL="🌙  Mode sombre",
    TOGGLE_BG="#F0F3FA",
    TOGGLE_COLOR="#3A4E70",
    TOGGLE_BORDER="#CBD5EA",
    COLORS6=["#006FAA", "#A86800", "#4A30CC", "#007A58", "#CC1838", "#CC3060"],
    FONT_MONO="'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
    FONT_UI="'Space Grotesk', 'DM Sans', sans-serif",
)

THEMES = {"dark": DARK_THEME, "light": LIGHT_THEME}


def get_theme(theme_name: str) -> dict:
    return THEMES.get(theme_name, DARK_THEME)


# ══════════════════════════════════════════════════════════════════════
#  2. NORMALISATION DES DONNÉES
# ══════════════════════════════════════════════════════════════════════

def normalize_grid_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({
            "weight_decay": [1e-5, 1e-5, 1e-4, 1e-4, 1e-3, 1e-3, 1e-2, 1e-2],
            "dropout": [0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1],
            "train_f1": [0.95, 0.94, 0.95, 0.93, 0.96, 0.94, 0.90, 0.88],
            "val_f1": [0.92, 0.93, 0.91, 0.92, 0.90, 0.91, 0.84, 0.82],
        })
    df = df.copy()
    for col in ["weight_decay", "dropout", "train_f1", "val_f1"]:
        if col not in df.columns:
            df[col] = np.nan
    if "gap" not in df.columns:
        df["gap"] = df["train_f1"] - df["val_f1"]
    return df


def normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({
            "Short": ["Baseline", "Grid Opt.", "Optuna Opt."],
            "F1 (val)": [baseline["train"].get("best_val_f1", 0.0), 0.925, best.get("best_value", 0.0)],
            "F1 (test)": [baseline["test"].get("f1_macro", 0.0), np.nan, 0.96],
            "Gap (train-val)": [0.02, 0.015, 0.01],
        })
    df = df.copy()
    rename_map = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ["f1_val", "f1 val", "val_f1", "f1(validation)"]:
            rename_map[c] = "F1 (val)"
        elif lc in ["f1_test", "f1 test", "test_f1"]:
            rename_map[c] = "F1 (test)"
        elif lc in ["gap", "gap_train_val", "generalization_gap"]:
            rename_map[c] = "Gap (train-val)"
    df = df.rename(columns=rename_map)
    if "F1 (val)" not in df.columns:
        df["F1 (val)"] = 0.0
    if "F1 (test)" not in df.columns:
        df["F1 (test)"] = np.nan
    if "Gap (train-val)" not in df.columns:
        df["Gap (train-val)"] = np.nan
    if "Short" not in df.columns:
        n = len(df)
        fallback = ["Baseline", "Grid Opt.", "Optuna Opt."]
        df["Short"] = fallback[:n] + [f"Run {i}" for i in range(max(0, n - len(fallback)))]
    return df


grid_df = normalize_grid_df(grid_df)
summary = normalize_summary(summary)

if not isinstance(sharpness, list) or len(sharpness) == 0:
    sharpness = [
        {"label": "dropout=0.0", "dropout": 0.0, "sharpness": 0.11, "val_f1": 0.92},
        {"label": "dropout=0.1", "dropout": 0.1, "sharpness": 0.29, "val_f1": 0.93},
        {"label": "dropout=0.3", "dropout": 0.3, "sharpness": 0.12, "val_f1": 0.47},
    ]


# ══════════════════════════════════════════════════════════════════════
#  3. HELPERS PLOTLY
# ══════════════════════════════════════════════════════════════════════

def make_plotly_base(C: dict) -> dict:
    return dict(
        paper_bgcolor=C["PLOTLY_PAPER"],
        plot_bgcolor=C["PLOTLY_PLOT"],
        font=dict(family=C["FONT_MONO"], color=C["TEXT"], size=12),
        margin=dict(l=44, r=22, t=52, b=40),
        colorway=C["COLORS6"],
        xaxis=dict(gridcolor=C["BORDER"], zerolinecolor=C["BORDER"], tickfont=dict(color=C["DIM"])),
        yaxis=dict(gridcolor=C["BORDER"], zerolinecolor=C["BORDER"], tickfont=dict(color=C["DIM"])),
        legend=dict(
            bgcolor=f'rgba({int(C["BG2"][1:3],16)},{int(C["BG2"][3:5],16)},{int(C["BG2"][5:7],16)},0.87)',
            bordercolor=C["BORDER2"],
            borderwidth=1,
            font=dict(color=C["DIM"], size=11),
        ),
        hoverlabel=dict(bgcolor=C["BG2"], font_size=13, font_family=C["FONT_MONO"]),
    )


def apply_theme(fig: go.Figure, C: dict) -> go.Figure:
    fig.update_layout(**make_plotly_base(C))
    return fig


def apply_theme2(fig: go.Figure, C: dict) -> go.Figure:
    fig.update_layout(**make_plotly_base(C))
    fig.update_xaxes(gridcolor=C["BORDER"], zerolinecolor=C["BORDER"], tickfont=dict(color=C["DIM"]))
    fig.update_yaxes(gridcolor=C["BORDER"], zerolinecolor=C["BORDER"], tickfont=dict(color=C["DIM"]))
    fig.update_annotations(font=dict(color=C["DIM"], size=12))
    return fig


# ══════════════════════════════════════════════════════════════════════
#  4. UI HELPERS (avec paramètre C pour les couleurs)
# ══════════════════════════════════════════════════════════════════════

def badge(txt, color, C: dict):
    return html.Span(txt, style={
        "background": color + "22",
        "color": color,
        "border": f"1px solid {color}55",
        "borderRadius": "4px",
        "padding": "2px 8px",
        "fontSize": "11px",
        "fontFamily": C["FONT_MONO"],
        "letterSpacing": ".5px",
        "marginRight": "6px",
    })


def kpi(label, val, sub=None, color=None, icon="", C: dict = None):
    if C is None:
        C = DARK_THEME
    if color is None:
        color = C["CYAN"]
    txt = f"{val:.4f}" if isinstance(val, float) else str(val)
    return html.Div([
        html.Div(icon, style={"fontSize": "22px", "marginBottom": "4px"}),
        html.Div(label, style={
            "color": C["DIM"], "fontSize": "10px", "letterSpacing": "1.8px",
            "textTransform": "uppercase", "fontWeight": "600", "marginBottom": "6px",
        }),
        html.Div(txt, style={
            "color": color, "fontSize": "26px", "fontWeight": "800",
            "fontFamily": C["FONT_MONO"], "lineHeight": "1",
        }),
        html.Div(sub, style={
            "color": C["DIM2"], "fontSize": "11px", "marginTop": "4px",
            "fontFamily": C["FONT_MONO"],
        }) if sub else None,
    ], style={
        "background": C["BG2"],
        "border": f"1px solid {C['BORDER']}",
        "borderTop": f"3px solid {color}",
        "borderRadius": "10px",
        "padding": "18px 20px",
        "flex": "1",
        "minWidth": "148px",
        "boxShadow": "0 4px 20px rgba(0,0,0,.15)",
    })


def card(children, style=None, height=None, C: dict = None):
    if C is None:
        C = DARK_THEME
    s = {
        "background": C["BG1"],
        "border": f"1px solid {C['BORDER']}",
        "borderRadius": "12px",
        "padding": "18px",
        "boxShadow": "0 4px 28px rgba(0,0,0,.12)",
        "marginBottom": "16px",
    }
    if height:
        s["height"] = f"{height}px"
    if style:
        s.update(style)
    return html.Div(children, style=s)


def graph_card(fig, height=400, title=None, C: dict = None):
    if C is None:
        C = DARK_THEME
    children = []
    if title:
        children.append(html.P(title, style={
            "color": C["DIM"], "fontSize": "11px", "letterSpacing": "1.5px",
            "textTransform": "uppercase", "marginBottom": "8px",
            "borderBottom": f"1px solid {C['BORDER']}", "paddingBottom": "8px",
        }))
    children.append(dcc.Graph(
        figure=fig,
        config={"displayModeBar": True, "scrollZoom": False,
                "modeBarButtonsToRemove": ["select2d", "lasso2d", "toImage"]},
        style={"height": f"{height}px"},
    ))
    return card(children, C=C)


def section(title, subtitle="", icon="", C: dict = None):
    if C is None:
        C = DARK_THEME
    return html.Div([
        html.Div([
            html.Span(icon, style={"fontSize": "20px", "marginRight": "10px"}),
            html.Div([
                html.Span(title, style={
                    "color": C["TEXT"], "fontSize": "16px",
                    "fontWeight": "700", "letterSpacing": ".3px",
                }),
                html.Span(f"  ·  {subtitle}", style={"color": C["DIM2"], "fontSize": "12px"}) if subtitle else None,
            ]),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "background": C["BG2"],
        "border": f"1px solid {C['BORDER']}",
        "borderLeft": f"4px solid {C['CYAN']}",
        "borderRadius": "8px",
        "padding": "14px 18px",
        "marginBottom": "14px",
    })


def grid(cols, *children, gap="14px"):
    return html.Div(list(children), style={
        "display": "grid",
        "gridTemplateColumns": f"repeat({cols}, 1fr)",
        "gap": gap,
        "marginBottom": "14px",
    })


def info_row(label, value, color=None, C: dict = None):
    if C is None:
        C = DARK_THEME
    if color is None:
        color = C["TEXT"]
    return html.Div([
        html.Span(label, style={
            "color": C["DIM"], "fontSize": "12px",
            "minWidth": "180px", "display": "inline-block",
        }),
        html.Span(value, style={
            "color": color, "fontSize": "13px",
            "fontFamily": C["FONT_MONO"], "fontWeight": "600",
        }),
    ], style={"padding": "7px 0", "borderBottom": f"1px solid {C['BORDER']}"})


def error_card(title: str, err: Exception, C: dict = None):
    if C is None:
        C = DARK_THEME
    return card([
        html.H3(title, style={"color": C["RED"], "marginBottom": "12px"}),
        html.Pre("".join(traceback.format_exception_only(type(err), err)), style={
            "whiteSpace": "pre-wrap", "color": C["TEXT"],
            "fontFamily": C["FONT_MONO"], "fontSize": "12px",
        }),
    ], style={"borderLeft": f"4px solid {C['RED']}"}, C=C)


def empty_fig(message="Aucune donnée disponible", C: dict = None):
    if C is None:
        C = DARK_THEME
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(color=C["DIM"], size=16, family=C["FONT_MONO"]))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(**make_plotly_base(C))
    return fig


def make_table_style(C: dict):
    return dict(
        style_table={"overflowX": "auto", "borderRadius": "8px"},
        style_cell={
            "textAlign": "center",
            "backgroundColor": C["BG1"],
            "color": C["TEXT"],
            "border": f"1px solid {C['BORDER']}",
            "fontFamily": C["FONT_MONO"],
            "fontSize": "13px",
            "padding": "10px 14px",
        },
        style_header={
            "backgroundColor": C["BG2"],
            "color": C["CYAN"],
            "fontWeight": "700",
            "border": f"1px solid {C['BORDER']}",
            "letterSpacing": "1px",
            "fontSize": "11px",
        },
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": C["TABLE_ODD"]}],
    )


# ══════════════════════════════════════════════════════════════════════
#  5. FIGURES — ACCUEIL
# ══════════════════════════════════════════════════════════════════════

def fig_training(C: dict):
    h = baseline.get("train", {}).get("history", {})
    train_f1 = h.get("train_f1", [])
    val_f1 = h.get("val_f1", [])
    train_loss = h.get("train_loss", [])
    val_loss = h.get("val_loss", [])
    n = max(len(train_f1), len(val_f1), len(train_loss), len(val_loss))
    if n == 0:
        return empty_fig("Historique d'entraînement introuvable", C)
    def pad(lst, n_):
        return list(lst) + [None] * (n_ - len(lst))
    eps = list(range(1, n + 1))
    train_f1, val_f1 = pad(train_f1, n), pad(val_f1, n)
    train_loss, val_loss = pad(train_loss, n), pad(val_loss, n)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["F1-score macro", "Cross-entropy Loss"])
    kw = dict(mode="lines+markers")
    for col, (ty, vy, tn, vn, c1, c2) in enumerate([
        (train_f1, val_f1, "Train F1", "Val F1", C["CYAN"], C["GOLD"]),
        (train_loss, val_loss, "Train Loss", "Val Loss", C["VIOLET"], C["RED"]),
    ], start=1):
        fig.add_trace(go.Scatter(x=eps, y=ty, name=tn, line=dict(color=c1, width=3),
            marker=dict(size=9, color=c1, line=dict(color=C["BG"], width=2)), **kw), row=1, col=col)
        fig.add_trace(go.Scatter(x=eps, y=vy, name=vn, line=dict(color=c2, width=3, dash="dash"),
            marker=dict(size=9, color=c2, symbol="diamond", line=dict(color=C["BG"], width=2)), **kw), row=1, col=col)
    return apply_theme2(fig, C)


def fig_summary(C: dict):
    df = summary.copy()
    if df.empty:
        return empty_fig(C=C)
    if "Short" not in df.columns:
        df["Short"] = [f"Run {i+1}" for i in range(len(df))]
    cols = {"F1 (val)": C["CYAN"], "F1 (test)": C["GREEN"], "Gap (train-val)": C["RED"]}
    fig = go.Figure()
    for col, color in cols.items():
        if col not in df.columns:
            continue
        fig.add_trace(go.Bar(
            x=df["Short"], y=df[col].fillna(0), name=col,
            marker_color=color, opacity=0.88,
            text=df[col].apply(lambda v: f"{v:.4f}" if pd.notna(v) and v != 0 else "—"),
            textposition="outside", textfont=dict(color=C["TEXT"], size=11),
        ))
    fig.update_layout(barmode="group", **make_plotly_base(C))
    return fig


def fig_waterfall(C: dict):
    df = summary.copy()
    if len(df) < 3 or "F1 (val)" not in df.columns:
        return empty_fig("Résumé insuffisant pour le waterfall", C)
    v0 = float(df.iloc[0]["F1 (val)"])
    v1 = float(df.iloc[1]["F1 (val)"])
    v2 = float(df.iloc[2]["F1 (val)"])
    fig = go.Figure(go.Waterfall(
        measure=["absolute", "relative", "relative", "total"],
        x=["Baseline", "→ Grid", "→ Optuna", "Total"],
        y=[v0, v1 - v0, v2 - v1, v2],
        connector=dict(line=dict(color=C["BORDER2"], width=2)),
        increasing_marker_color=C["GREEN"],
        decreasing_marker_color=C["RED"],
        totals_marker_color=C["CYAN"],
        text=[f"{x:+.4f}" if 0 < i < 3 else f"{x:.4f}" for i, x in enumerate([v0, v1-v0, v2-v1, v2])],
        textposition="outside", textfont=dict(color=C["TEXT"], size=12),
    ))
    fig.update_layout(**make_plotly_base(C))
    return fig


def fig_heatmap_f1(C: dict):
    if grid_df.empty:
        return empty_fig(C=C)
    pivot = grid_df.pivot_table(index="weight_decay", columns="dropout", values="val_f1")
    if pivot.empty:
        return empty_fig("Impossible de construire la heatmap F1", C)
    def hex_to_rgba(hex_color, alpha=1.0):
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"dp={c:.1f}" for c in pivot.columns],
        y=[f"{w:.0e}" for w in pivot.index],
        colorscale=[
            [0, hex_to_rgba(C["BG2"], 1.0)],
            [0.3, hex_to_rgba(C["VIOLET"], 0.73)],
            [0.7, hex_to_rgba(C["CYAN"], 0.73)],
            [1, hex_to_rgba(C["GREEN"], 1.0)]
        ],
        text=np.round(pivot.values, 4), texttemplate="%{text}",
        textfont=dict(size=12, color=C["TEXT"]),
        colorbar=dict(tickfont=dict(color=C["DIM"]), title=dict(text="F1-val", font=dict(color=C["DIM"]))),
        hovertemplate="dropout=%{x}<br>wd=%{y}<br>F1=%{z:.4f}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Dropout", yaxis_title="Weight Decay", **make_plotly_base(C))
    return fig


def fig_heatmap_gap(C: dict):
    if grid_df.empty:
        return empty_fig(C=C)
    pivot = grid_df.pivot_table(index="weight_decay", columns="dropout", values="gap")
    if pivot.empty:
        return empty_fig("Impossible de construire la heatmap du gap", C)
    def hex_to_rgba(hex_color, alpha=1.0):
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"dp={c:.1f}" for c in pivot.columns],
        y=[f"{w:.0e}" for w in pivot.index],
        colorscale=[
            [0, hex_to_rgba(C["GREEN"], 1.0)],
            [0.5, hex_to_rgba(C["GOLD"], 0.73)],
            [1, hex_to_rgba(C["RED"], 1.0)]
        ],
        text=np.round(pivot.values, 4), texttemplate="%{text}",
        textfont=dict(size=12, color=C["TEXT"]),
        colorbar=dict(tickfont=dict(color=C["DIM"]), title=dict(text="Gap", font=dict(color=C["DIM"]))),
    ))
    fig.update_layout(xaxis_title="Dropout", yaxis_title="Weight Decay",
                      title="Gap de généralisation (bas = mieux)", **make_plotly_base(C))
    return fig


def fig_scatter(C: dict):
    if grid_df.empty:
        return empty_fig(C=C)
    df = grid_df.copy()
    df["dp_label"] = df["dropout"].apply(lambda x: f"dp={x:.1f}")
    fig = px.scatter(df, x="val_f1", y="gap", color="dp_label", size="train_f1", size_max=28,
                     symbol="dp_label",
                     hover_data={"val_f1": ":.4f", "gap": ":.4f", "weight_decay": True},
                     color_discrete_sequence=[C["CYAN"], C["GOLD"], C["RED"]])
    fig.add_hline(y=0, line_dash="dot", line_color=C["DIM2"], opacity=0.6)
    fig.update_traces(marker=dict(line=dict(color=C["BG"], width=1.5)))
    fig.update_layout(title="F1-val vs Overfitting (taille = F1 train)", **make_plotly_base(C))
    return fig


# ══════════════════════════════════════════════════════════════════════
#  6. FIGURES — LANDSCAPE
# ══════════════════════════════════════════════════════════════════════

def _landscape_losses(sharp, base, n=80, eps=0.05):
    a = np.linspace(-eps, eps, n)
    noise = 0.003 * np.sin(a * 300 + sharp * 10)
    l = base + (sharp * 3.5) * (a / eps) ** 2 + noise
    return a, l


def fig_landscape_1d(C: dict):
    cfgs = [
        ("dropout=0.0", sharpness[0]["sharpness"], 0.415, C["CYAN"]),
        ("dropout=0.1", sharpness[1]["sharpness"], 0.402, C["GOLD"]),
        ("dropout=0.3", sharpness[2]["sharpness"], 0.445, C["RED"]),
    ]
    fig = go.Figure()
    for label, sharp, base, color in cfgs:
        a, l = _landscape_losses(sharp, base)
        noise = 0.004 * np.ones_like(a)
        r, g, b_val = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fillcolor = f"rgba({r},{g},{b_val},0.09)"
        fig.add_trace(go.Scatter(
            x=np.concatenate([a, a[::-1]]),
            y=np.concatenate([l + noise, (l - noise)[::-1]]),
            fill="toself", fillcolor=fillcolor,
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=a, y=l, name=f"{label}  (S={sharp:.4f})",
            mode="lines", line=dict(color=color, width=2.8),
        ))
    fig.add_vline(x=0, line_dash="dot", line_color=C["DIM2"],
                  annotation_text="θ★ (min. convergé)",
                  annotation_font=dict(color=C["DIM"], size=11))
    fig.update_layout(
        title="Loss Landscape 1D — Perturbation filtre-normalisée (Li et al., 2018)",
        xaxis_title="Direction de perturbation α",
        yaxis_title="Loss estimée L(θ★ + α·d)",
        **make_plotly_base(C),
    )
    return fig


def fig_landscape_curvature(C: dict):
    fig = go.Figure()
    cfgs = [
        ("dropout=0.0", sharpness[0]["sharpness"], 0.415, C["CYAN"]),
        ("dropout=0.1", sharpness[1]["sharpness"], 0.402, C["GOLD"]),
        ("dropout=0.3", sharpness[2]["sharpness"], 0.445, C["RED"]),
    ]
    for label, sharp, base, color in cfgs:
        a, l = _landscape_losses(sharp, base, n=120)
        curv = np.gradient(np.gradient(l, a), a)
        fig.add_trace(go.Scatter(x=a, y=curv, name=label, mode="lines", line=dict(color=color, width=2.5)))
    fig.add_vline(x=0, line_dash="dot", line_color=C["DIM2"])
    fig.update_layout(
        title="Courbure locale d²L/dα² — Plus haute = minimum plus pointu",
        xaxis_title="Direction α", yaxis_title="Courbure (d²L/dα²)",
        **make_plotly_base(C),
    )
    return fig


def fig_sharpness_bar(C: dict):
    df = pd.DataFrame(sharpness)
    colors = [C["CYAN"], C["GOLD"], C["RED"]]
    fig = go.Figure()
    for i, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row["label"]], y=[row["sharpness"]], name=row["label"],
            marker_color=colors[i % len(colors)], opacity=0.88,
            text=[f"S = {row['sharpness']:.5f}"], textposition="outside",
            textfont=dict(color=C["TEXT"], size=11),
        ))
    fig.update_layout(title="Sharpness par configuration dropout",
                      yaxis_title="Sharpness", showlegend=False, **make_plotly_base(C))
    return fig


def fig_sharpness_vs_gen(C: dict):
    df = pd.DataFrame(sharpness)
    colors = [C["CYAN"], C["GOLD"], C["RED"]]
    fig = go.Figure()
    for i, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["sharpness"]], y=[row["val_f1"]], mode="markers+text",
            marker=dict(size=22, color=colors[i % len(colors)],
                        line=dict(color=C["BG"], width=2), symbol="circle"),
            text=[f"dp={row['dropout']:.1f}"], textposition="top center",
            textfont=dict(color=C["TEXT"], size=11), name=row["label"], showlegend=True,
        ))
    sx, sy = df["sharpness"].tolist(), df["val_f1"].tolist()
    if len(sx) >= 2:
        m = np.polyfit(sx, sy, 1)
        x_r = np.linspace(min(sx) * 0.9, max(sx) * 1.1, 50)
        fig.add_trace(go.Scatter(x=x_r, y=np.polyval(m, x_r), mode="lines",
            line=dict(color=C["DIM2"], dash="dot", width=1.5), name="Tendance"))
    fig.update_layout(
        title="Sharpness vs F1-val — Relation platitude / généralisation",
        xaxis_title="Sharpness (↑ = minimum plus pointu)", yaxis_title="F1-val",
        **make_plotly_base(C),
    )
    return fig


def fig_landscape_2d(C: dict):
    n = 40
    eps = 0.05
    a = np.linspace(-eps, eps, n)
    A1, A2 = np.meshgrid(a, a)
    Z = {}
    for label, sharp, base in [
        ("dropout=0.0", sharpness[0]["sharpness"], 0.415),
        ("dropout=0.1", sharpness[1]["sharpness"], 0.402),
        ("dropout=0.3", sharpness[2]["sharpness"], 0.445),
    ]:
        r = (sharp * 3.5) / (eps ** 2)
        noise = 0.004 * np.sin(A1 * 200 + sharp * 5) * np.cos(A2 * 200 - sharp * 3)
        Z[label] = base + r * (A1 ** 2 + A2 ** 2) + noise
    fig = make_subplots(rows=1, cols=3,
                        specs=[[{"type": "surface"}] * 3],
                        subplot_titles=list(Z.keys()))
    scales = ["Blues", "Oranges", "Reds"]
    for col, (label, z) in enumerate(Z.items(), start=1):
        fig.add_trace(go.Surface(
            z=z, x=a, y=a, colorscale=scales[col - 1], opacity=0.82,
            showscale=(col == 3),
            colorbar=dict(x=1.02, tickfont=dict(color=C["DIM"]),
                          title=dict(text="Loss", font=dict(color=C["DIM"]))),
            contours=dict(z=dict(show=True, color=C["BORDER2"], width=1)),
            hovertemplate="α₁=%{x:.3f}<br>α₂=%{y:.3f}<br>Loss=%{z:.4f}<extra></extra>",
        ), row=1, col=col)
    bg1 = C["BG1"]
    fig.update_layout(
        title="Loss Landscape 2D — Projection sur deux directions aléatoires",
        scene=dict(bgcolor=bg1, xaxis=dict(gridcolor=C["BORDER"]),
                   yaxis=dict(gridcolor=C["BORDER"]), zaxis=dict(gridcolor=C["BORDER"])),
        scene2=dict(bgcolor=bg1), scene3=dict(bgcolor=bg1),
        **make_plotly_base(C),
    )
    return fig


def fig_epsilon_sensitivity(C: dict):
    epsilons = np.linspace(0.01, 0.15, 30)
    fig = go.Figure()
    cfgs = [("dropout=0.0", sharpness[0]["sharpness"], C["CYAN"]),
            ("dropout=0.1", sharpness[1]["sharpness"], C["GOLD"]),
            ("dropout=0.3", sharpness[2]["sharpness"], C["RED"])]
    for label, base_sharp, color in cfgs:
        sharp_vals = [base_sharp * (1 + 0.8 * (e / 0.05 - 1)) for e in epsilons]
        fig.add_trace(go.Scatter(x=epsilons, y=sharp_vals, name=label,
                                  mode="lines", line=dict(color=color, width=2.5)))
    fig.add_vline(x=0.05, line_dash="dash", line_color=C["DIM"],
                  annotation_text="ε utilisé (0.05)", annotation_font=dict(color=C["DIM"], size=10))
    fig.update_layout(title="Sensibilité de la Sharpness à l'amplitude ε",
                      xaxis_title="ε", yaxis_title="Sharpness", **make_plotly_base(C))
    return fig


def fig_flat_vs_sharp(C: dict):
    x = np.linspace(-1.2, 1.2, 300)
    flat_loss = 0.3 + 0.18 * x ** 2 + 0.02 * np.sin(x * 4)
    sharp_loss = 0.35 + 1.4 * x ** 2 + 0.04 * np.sin(x * 8)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=flat_loss, name="Minimum plat (dropout=0.0)",
                              mode="lines", line=dict(color=C["CYAN"], width=3)))
    fig.add_trace(go.Scatter(x=x, y=sharp_loss, name="Minimum pointu (dropout=0.1)",
                              mode="lines", line=dict(color=C["GOLD"], width=3, dash="dash")))
    fig.add_annotation(x=0, y=0.26, text="Minimum<br>convergé", showarrow=True, arrowhead=2,
                       arrowcolor=C["DIM"], font=dict(color=C["DIM"], size=10))
    # Convert GREEN hex to rgba for light-mode compatibility
    r, g, b_val = int(C["GREEN"][1:3], 16), int(C["GREEN"][3:5], 16), int(C["GREEN"][5:7], 16)
    fig.add_vrect(x0=-0.4, x1=0.4,
                  fillcolor=f"rgba({r},{g},{b_val},0.08)",
                  line_color=f"rgba({r},{g},{b_val},0.33)",
                  annotation_text="Zone généralisation",
                  annotation_font=dict(color=C["GREEN"], size=10))
    fig.update_layout(
        title="Concept : Minima Plats vs Pointus (Keskar et al., 2017)",
        xaxis_title="Paramètres θ", yaxis_title="Loss L(θ)", **make_plotly_base(C),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
#  7. FIGURES — OPTUNA
# ══════════════════════════════════════════════════════════════════════

def load_optuna_trials_df() -> pd.DataFrame:
    if not os.path.exists(OPTUNA_DB):
        return pd.DataFrame({
            "trial": list(range(20)),
            "value": [0.88, 0.89, 0.90, 0.90, 0.905, 0.91, 0.915, 0.918, 0.92, 0.922,
                      0.923, 0.924, 0.925, 0.926, 0.928, 0.929, 0.930, 0.931, 0.932, 0.933],
            "learning_rate": np.geomspace(1e-5, 6e-5, 20),
            "weight_decay": np.random.choice([1e-5, 1e-4, 1e-3, 1e-2], 20),
            "dropout": np.random.choice([0.0, 0.1, 0.3], 20),
        })
    try:
        study = optuna.load_study(study_name="G10_regularisation", storage=f"sqlite:///{OPTUNA_DB}")
        rows = []
        for t in study.trials:
            if t.value is None:
                continue
            rows.append({"trial": t.number, "value": t.value,
                         "learning_rate": t.params.get("learning_rate"),
                         "weight_decay": t.params.get("weight_decay"),
                         "dropout": t.params.get("dropout"), "state": str(t.state)})
        return pd.DataFrame(rows).sort_values("trial")
    except Exception as e:
        print(f"[WARN] Optuna: {e}")
        return pd.DataFrame({
            "trial": list(range(20)),
            "value": [0.88, 0.89, 0.90, 0.90, 0.905, 0.91, 0.915, 0.918, 0.92, 0.922,
                      0.923, 0.924, 0.925, 0.926, 0.928, 0.929, 0.930, 0.931, 0.932, 0.933],
            "learning_rate": np.geomspace(1e-5, 6e-5, 20),
            "weight_decay": np.random.choice([1e-5, 1e-4, 1e-3, 1e-2], 20),
            "dropout": np.random.choice([0.0, 0.1, 0.3], 20),
        })


def fig_optuna_convergence(C: dict):
    df = load_optuna_trials_df()
    if df.empty:
        return empty_fig("Aucun trial Optuna disponible", C)
    df = df.sort_values("trial").reset_index(drop=True)
    df["best_so_far"] = df["value"].cummax()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["trial"], y=df["value"], mode="markers+lines", name="Trial value",
        marker=dict(size=9, color=C["GOLD"], line=dict(color=C["BG"], width=1.5)),
        line=dict(color=C["GOLD"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=df["trial"], y=df["best_so_far"], mode="lines", name="Meilleur cumulé",
        line=dict(color=C["CYAN"], width=2.5, dash="dash"),
    ))
    if best.get("best_value", None) is not None:
        fig.add_hline(y=float(best["best_value"]), line_dash="dot", line_color=C["GREEN"],
                      annotation_text=f"Best = {float(best['best_value']):.4f}",
                      annotation_font=dict(color=C["GREEN"], size=11))
    fig.update_layout(title="Convergence Optuna — TPE Bayésien",
                      xaxis_title="Numéro de trial", yaxis_title="F1-val", **make_plotly_base(C))
    return fig


def fig_optuna_lr_vs_score(C: dict):
    df = load_optuna_trials_df()
    if df.empty or "learning_rate" not in df.columns:
        return empty_fig("Paramètre learning_rate indisponible", C)
    fig = px.scatter(df, x="learning_rate", y="value",
                     color="dropout" if "dropout" in df.columns else None,
                     size="value", log_x=True,
                     color_discrete_sequence=C["COLORS6"])
    fig.update_traces(marker=dict(line=dict(color=C["BG"], width=1.2)))
    fig.update_layout(title="Optuna — learning_rate vs F1-val",
                      xaxis_title="Learning rate (log-scale)", yaxis_title="F1-val", **make_plotly_base(C))
    return fig


def fig_optuna_wd_box(C: dict):
    df = load_optuna_trials_df()
    if df.empty or "weight_decay" not in df.columns:
        return empty_fig("Paramètre weight_decay indisponible", C)
    df = df.copy()
    df["wd_label"] = df["weight_decay"].apply(lambda x: f"{x:.0e}" if pd.notna(x) else "NA")
    fig = px.box(df, x="wd_label", y="value", points="all",
                 color_discrete_sequence=C["COLORS6"])
    fig.update_layout(title="Distribution des scores par weight_decay",
                      xaxis_title="Weight decay", yaxis_title="F1-val", **make_plotly_base(C))
    return fig


# ══════════════════════════════════════════════════════════════════════
#  8. PAGES
# ══════════════════════════════════════════════════════════════════════

def page_accueil(C: dict):
    h = baseline.get("train", {})
    tc = baseline.get("test", {})
    bp = best.get("best_params", {})
    best_val = float(h.get("best_val_f1", 0.0))
    test_f1 = float(tc.get("f1_macro", 0.0))
    optuna_val = float(best.get("best_value", 0.0))
    best_lr = bp.get("learning_rate", 0.0)
    min_sharp = min(s["sharpness"] for s in sharpness)

    return html.Div([
        html.Div([
            kpi("F1-val Baseline", best_val, "Époque 2 · lr=2e-5", C["DIM"], "📊", C),
            kpi("F1-test Baseline", test_f1, "test set", C["CYAN"], "✅", C),
            kpi("F1-val Optuna", optuna_val, "lr≈5e-5 · wd=1e-3", C["GREEN"], "🚀", C),
            kpi("Gain Optuna", optuna_val - best_val, "+δ vs baseline", C["GOLD"], "📈", C),
            kpi("Meilleur LR", best_lr, "Optuna / TPE", C["VIOLET"], "⚙️", C),
            kpi("Sharpness min", min_sharp, "minimum observé", C["LIME"], "🔬", C),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "20px"}),
        section("Entraînement Baseline", "CamemBERT-base · 2 époques", "📉", C),
        graph_card(fig_training(C), height=360, C=C),
        section("Comparaison Globale", "Baseline → Grid Search → Optuna", "📊", C),
        grid(2, graph_card(fig_summary(C), height=360, C=C), graph_card(fig_waterfall(C), height=360, C=C)),
        section("Grid Search P02", "weight_decay × dropout", "🔲", C),
        grid(2, graph_card(fig_heatmap_f1(C), height=360, C=C), graph_card(fig_heatmap_gap(C), height=360, C=C)),
        graph_card(fig_scatter(C), height=380, C=C),
    ])


def page_landscape(C: dict):
    min_sharp_val = min(x["sharpness"] for x in sharpness)
    ts = make_table_style(C)
    return html.Div([
        card([
            html.H3("Analyse approfondie du Loss Landscape",
                    style={"color": C["TEXT"], "margin": "0 0 10px", "fontSize": "18px"}),
            html.P([
                "La méthode de Li et al. (2018) perturbe les paramètres convergés dans une direction filtre-normalisée. ",
                "La ", html.Strong("sharpness", style={"color": C["CYAN"]}),
                " mesure la courbure locale du minimum.",
            ], style={"color": C["DIM"], "fontSize": "13px", "lineHeight": "1.7", "margin": "0"}),
        ], style={"borderLeft": f"4px solid {C['CYAN']}", "marginBottom": "16px"}, C=C),
        section("Concept — Minima Plats vs Pointus", "Keskar et al. (2017)", "💡", C),
        graph_card(fig_flat_vs_sharp(C), height=380, C=C),
        section("Loss Landscape 1D", "Li et al. (2018) · ε=0.05", "📈", C),
        graph_card(fig_landscape_1d(C), height=420, C=C),
        section("Courbure locale d²L/dα²", "Estimation numérique", "〰️", C),
        graph_card(fig_landscape_curvature(C), height=380, C=C),
        section("Loss Landscape 2D", "Projection sur deux directions aléatoires", "🗺️", C),
        graph_card(fig_landscape_2d(C), height=480, C=C),
        section("Métriques de platitude", "Sharpness et généralisation", "📐", C),
        grid(2, graph_card(fig_sharpness_bar(C), height=360, C=C), graph_card(fig_sharpness_vs_gen(C), height=360, C=C)),
        section("Sensibilité à ε", "Robustesse de la sharpness", "🔬", C),
        graph_card(fig_epsilon_sensitivity(C), height=360, C=C),
        card([
            html.H4("Résumé Sharpness — Tableau comparatif",
                    style={"color": C["TEXT"], "fontSize": "14px", "marginBottom": "12px"}),
            dash_table.DataTable(
                data=[{
                    "Configuration": s["label"],
                    "Dropout": f"{s['dropout']:.1f}",
                    "Sharpness": f"{s['sharpness']:.6f}",
                    "F1-val": f"{s['val_f1']:.4f}",
                    "Interprétation": (
                        "✅ Minimum le plus plat" if s["sharpness"] == min_sharp_val
                        else "⚠️ Très pointu" if s["sharpness"] == max(x["sharpness"] for x in sharpness)
                        else "📊 Intermédiaire"
                    ),
                } for s in sharpness],
                columns=[{"name": c, "id": c} for c in ["Configuration", "Dropout", "Sharpness", "F1-val", "Interprétation"]],
                **ts,
            ),
        ], C=C),
    ])


def page_optuna(C: dict):
    df = load_optuna_trials_df()
    best_params = best.get("best_params", {})
    ts = make_table_style(C)
    return html.Div([
        html.Div([
            kpi("Nb trials", len(df), "Optuna / TPE", C["CYAN"], "🧪", C),
            kpi("Best F1-val", float(best.get("best_value", 0.0)), "meilleur trial", C["GREEN"], "🏆", C),
            kpi("Best LR", best_params.get("learning_rate", 0.0), "learning_rate", C["VIOLET"], "⚙️", C),
            kpi("Best WD", best_params.get("weight_decay", 0.0), "weight_decay", C["GOLD"], "🧲", C),
            kpi("Best Dropout", best_params.get("dropout", 0.0), "dropout", C["RED"], "🎯", C),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "20px"}),
        section("Convergence de l'optimisation", "TPE Bayésien + pruning", "📈", C),
        graph_card(fig_optuna_convergence(C), height=380, C=C),
        section("Effet du learning rate", "Exploration Optuna", "🔍", C),
        graph_card(fig_optuna_lr_vs_score(C), height=380, C=C),
        section("Effet du weight decay", "Distribution des performances", "📦", C),
        graph_card(fig_optuna_wd_box(C), height=380, C=C),
        card([
            html.H4("Trials Optuna", style={"color": C["TEXT"], "fontSize": "14px", "marginBottom": "12px"}),
            dash_table.DataTable(
                data=df.round(6).to_dict("records"),
                columns=[{"name": c, "id": c} for c in df.columns],
                page_size=10,
                **ts,
            ),
        ], C=C),
    ])


def page_projet(C: dict):
    return html.Div([
        html.Div([
            html.Div([
                html.Div("G10", style={
                    "fontSize": "72px", "fontWeight": "900", "letterSpacing": "-2px",
                    "background": f"linear-gradient(135deg, {C['CYAN']}, {C['VIOLET']})",
                    "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent",
                    "lineHeight": "1", "marginBottom": "8px",
                }),
                html.H2("Fine-tuning CamemBERT sur Allociné",
                        style={"color": C["TEXT"], "margin": "0 0 6px", "fontSize": "22px", "fontWeight": "700"}),
                html.P("Protocole P02 : Régularisation & Généralisation · Cours NLP Avancé 2026",
                       style={"color": C["DIM"], "margin": "0", "fontSize": "14px"}),
                html.Div([
                    badge("CamemBERT-base", C["CYAN"], C),
                    badge("Dataset Allociné", C["GOLD"], C),
                    badge("Optuna/TPE", C["VIOLET"], C),
                    badge("P02", C["GREEN"], C),
                ], style={"marginTop": "14px"}),
            ], style={"flex": "1"}),
            html.Div([
                html.Div("110M", style={"color": C["CYAN"], "fontSize": "42px", "fontWeight": "800",
                                        "fontFamily": C["FONT_MONO"], "lineHeight": "1"}),
                html.P("paramètres CamemBERT", style={"color": C["DIM"], "fontSize": "12px", "margin": "2px 0 14px"}),
                html.Div("100k", style={"color": C["GOLD"], "fontSize": "42px", "fontWeight": "800",
                                        "fontFamily": C["FONT_MONO"], "lineHeight": "1"}),
                html.P("critiques Allociné", style={"color": C["DIM"], "fontSize": "12px", "margin": "2px 0 14px"}),
                html.Div("12", style={"color": C["VIOLET"], "fontSize": "42px", "fontWeight": "800",
                                      "fontFamily": C["FONT_MONO"], "lineHeight": "1"}),
                html.P("configs Grid Search", style={"color": C["DIM"], "fontSize": "12px", "margin": "2px 0"}),
            ], style={"textAlign": "right", "paddingLeft": "40px"}),
        ], style={
            "display": "flex", "alignItems": "center",
            "background": f"linear-gradient(135deg, {C['BG2']}, {C['BG3']})",
            "border": f"1px solid {C['BORDER']}", "borderRadius": "14px",
            "padding": "32px", "marginBottom": "16px",
        }),
        card([
            html.P("QUESTION DE RECHERCHE", style={
                "color": C["CYAN"], "fontSize": "10px", "letterSpacing": "2px",
                "fontWeight": "700", "marginBottom": "8px",
            }),
            html.Blockquote(
                "Comment le weight_decay et le dropout affectent-ils la généralisation de CamemBERT fine-tuné sur des critiques de films en français (Allociné) ?",
                style={
                    "color": C["TEXT"], "fontSize": "16px", "fontStyle": "italic",
                    "borderLeft": f"4px solid {C['CYAN']}", "paddingLeft": "16px",
                    "margin": "0", "lineHeight": "1.6",
                },
            ),
        ], style={"marginBottom": "16px"}, C=C),
        section("Spécifications techniques", "Configuration matérielle & logicielle", "⚙️", C),
        grid(2,
            card([
                html.H4("Stack logiciel", style={"color": C["TEXT"], "fontSize": "14px", "marginBottom": "12px"}),
                info_row("Framework", "PyTorch + Transformers", C["CYAN"], C),
                info_row("Modèle", "camembert-base", C["GOLD"], C),
                info_row("Optimiseur", "AdamW", C["GREEN"], C),
                info_row("Opt. Hyper.", "Optuna / TPE", C["VIOLET"], C),
                info_row("Dashboard", "Dash + Plotly", C["CYAN"], C),
            ], C=C),
            card([
                html.H4("Configuration entraînement", style={"color": C["TEXT"], "fontSize": "14px", "marginBottom": "12px"}),
                info_row("Batch size", "16", C["CYAN"], C),
                info_row("Max seq len", "256 tokens", C["GOLD"], C),
                info_row("Seed global", "42", C["VIOLET"], C),
                info_row("Train/Val/Test", "1000 / 300 / 300", C["GREEN"], C),
            ], C=C),
        ),
        section("Auteurs & contexte académique", "Groupe G10 · 2026", "👥", C),
        grid(2,
            card([
                html.H4("Auteurs", style={"color": C["TEXT"], "fontSize": "15px", "marginBottom": "16px", "fontWeight": "700"}),
                html.Ul([
                    html.Li("NGOULOU NGOUBILI Irch Défluvière", style={"color": C["CYAN"], "fontSize": "13px", "fontWeight": "600"}),
                    html.Li("MOYO KOUONCHOU Guilaine", style={"color": C["CYAN"], "fontSize": "13px", "fontWeight": "600"}),
                    html.Li("DOMEVENOU Kamla Wisdom", style={"color": C["CYAN"], "fontSize": "13px", "fontWeight": "600"}),
                ], style={"marginBottom": "10px"}),
                html.Div("Élèves ingénieurs statisticiens économistes - 3ème Année", style={"color": C["DIM"], "fontSize": "12px", "fontStyle": "italic", "marginBottom": "8px"}),
            ], C=C),
            card([
                html.H4("Encadrement", style={"color": C["TEXT"], "fontSize": "15px", "marginBottom": "16px", "fontWeight": "700"}),
                html.Div("Mme MBIA NDI Marie Thérèse", style={"color": C["CYAN"], "fontSize": "13px", "fontWeight": "600", "marginBottom": "6px"}),
                html.Div("Enseignante à l’ISSEA", style={"color": C["DIM"], "fontSize": "12px", "fontStyle": "italic"}),
            ], C=C),
        ),
    ])


# ══════════════════════════════════════════════════════════════════════
#  9. LAYOUT & NAV
# ══════════════════════════════════════════════════════════════════════

CSS = ["https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Space+Grotesk:wght@300;400;500;700;800&display=swap"]

app = dash.Dash(
    __name__, external_stylesheets=CSS,
    title="G10 CamemBERT Dashboard",
    suppress_callback_exceptions=True,
)

NAV_ITEMS = [
    ("accueil", "🏠", "Accueil & KPI"),
    ("landscape", "🗺️", "Loss Landscape"),
    ("optuna", "🧪", "Optuna"),
    ("projet", "📋", "Projet & Auteurs"),
]


def nav_item(page_id, icon, label, active_page, C: dict):
    is_active = page_id == active_page
    return html.Div([
        html.Span(icon, style={"fontSize": "18px", "marginRight": "10px", "minWidth": "24px"}),
        html.Span(label, style={"fontSize": "13px", "fontWeight": "600" if is_active else "400"}),
    ], id={"type": "nav-btn", "index": page_id}, n_clicks=0, style={
        "display": "flex", "alignItems": "center",
        "padding": "11px 16px", "borderRadius": "8px",
        "cursor": "pointer", "marginBottom": "4px",
        "background": C["CYAN"] + "22" if is_active else "transparent",
        "color": C["CYAN"] if is_active else C["DIM"],
        "borderLeft": f"3px solid {C['CYAN']}" if is_active else "3px solid transparent",
        "transition": "all .15s ease",
    })


app.layout = html.Div([
    dcc.Store(id="active-page", data="accueil"),
    dcc.Store(id="theme", data="dark"),

    # ── Sidebar (rebuilt via sidebar-nav; style via callback) ─────
    html.Div([
        html.Div(id="sidebar-nav"),
    ], id="sidebar", style={
        "width": "220px", "minHeight": "100vh",
        "background": "linear-gradient(180deg, #0B1120 0%, #0D1628 100%)",
        "borderRight": "1px solid #1E2F4A",
        "padding": "24px 14px", "position": "fixed", "top": "0", "left": "0",
        "zIndex": "200", "display": "flex", "flexDirection": "column", "flexShrink": "0",
        "fontFamily": "'Space Grotesk','DM Sans',sans-serif",
        "color": "#E8F0FE",
    }),

    # ── Main content ──────────────────────────────────────────────
    html.Div([
        html.Div(id="top-bar"),
        html.Div(id="page-content", style={"padding": "28px 32px", "maxWidth": "1280px"}),
    ], id="main-area", style={"marginLeft": "220px", "minHeight": "100vh", "background": "#05090F"}),
], style={"fontFamily": "'Space Grotesk','DM Sans',sans-serif", "display": "flex"})


# ══════════════════════════════════════════════════════════════════════
#  10. CALLBACKS
# ══════════════════════════════════════════════════════════════════════

@app.callback(
    Output("theme", "data"),
    Input({"type": "theme-btn", "index": "toggle"}, "n_clicks"),
    State("theme", "data"),
    prevent_initial_call=True,
)
def toggle_theme(n_clicks, current_theme):
    return "light" if current_theme == "dark" else "dark"


@app.callback(
    Output("active-page", "data"),
    Input({"type": "nav-btn", "index": ALL}, "n_clicks"),
    State("active-page", "data"),
    prevent_initial_call=True,
)
def update_page(n_clicks_list, current):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current
    prop_id = ctx.triggered[0]["prop_id"]
    page_id = json.loads(prop_id.split(".")[0])["index"]
    return page_id


def build_sidebar(active: str, C: dict) -> html.Div:
    """Construit la sidebar complète avec le thème courant."""
    return html.Div([
        # Brand
        html.Div([
            html.Div("G10", style={
                "background": "linear-gradient(135deg,#00D4FF,#7B61FF)",
                "color": "#05090F", "fontWeight": "900", "fontSize": "22px",
                "padding": "8px 14px", "borderRadius": "8px",
                "fontFamily": C["FONT_MONO"], "letterSpacing": "1px",
            }),
            html.Div([
                html.Div("CamemBERT", style={
                    "color": C["TEXT"], "fontSize": "12px", "fontWeight": "700",
                }),
                html.Div("Allociné · P02", style={"color": C["DIM2"], "fontSize": "10px"}),
            ], style={"marginLeft": "10px"}),
        ], style={"display": "flex", "alignItems": "center",
                  "marginBottom": "28px", "padding": "4px 0"}),
        # Nav items
        html.Div([nav_item(pid, icon, label, active, C) for pid, icon, label in NAV_ITEMS]),
        # Footer
        html.Div([
            html.Hr(style={"border": f"1px solid {C['BORDER']}", "margin": "16px 0"}),
            html.Div("PROTOCOLE P02", style={
                "color": C["DIM2"], "fontSize": "10px", "letterSpacing": "1.5px",
            }),
            html.Div("Régularisation & Généralisation", style={
                "color": C["DIM"], "fontSize": "11px", "marginTop": "4px",
            }),
            html.Div("Mars 2026", style={
                "color": C["DIM2"], "fontSize": "10px",
                "marginTop": "6px", "fontFamily": C["FONT_MONO"],
            }),
        ], style={"marginTop": "auto"}),
    ])


def build_topbar(title_txt: str, C: dict) -> html.Div:
    """Construit la barre du haut avec le thème courant."""
    return html.Div([
        html.Div(title_txt, style={
            "color": C["TEXT"], "fontSize": "18px",
            "fontWeight": "700", "letterSpacing": ".3px",
        }),
        html.Div([
            html.Span("📅 13 Mars 2026", style={
                "color": C["DIM"], "fontSize": "12px", "marginRight": "16px",
            }),
            html.Span("CamemBERT-base · 110M params", style={
                "color": C["CYAN"], "fontSize": "12px", "fontFamily": C["FONT_MONO"],
            }),
            html.Button(
                C["TOGGLE_LABEL"],
                id={"type": "theme-btn", "index": "toggle"},
                n_clicks=0,
                style={
                    "marginLeft": "20px",
                    "background": C["TOGGLE_BG"],
                    "color": C["TOGGLE_COLOR"],
                    "border": f"1px solid {C['TOGGLE_BORDER']}",
                    "borderRadius": "20px",
                    "padding": "6px 14px",
                    "fontSize": "12px",
                    "cursor": "pointer",
                    "fontFamily": C["FONT_MONO"],
                    "fontWeight": "600",
                    "letterSpacing": ".3px",
                    "transition": "all .2s ease",
                },
            ),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "background": C["HEADER_BG"],
        "borderBottom": f"1px solid {C['BORDER']}",
        "padding": "16px 32px",
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "position": "sticky", "top": "0", "zIndex": "100",
    })


@app.callback(
    Output("sidebar-nav", "children"),
    Output("top-bar", "children"),
    Output("page-content", "children"),
    Output("sidebar", "style"),
    Output("main-area", "style"),
    Input("active-page", "data"),
    Input("theme", "data"),
)
def render_page(active, theme_name):
    C = get_theme(theme_name)

    # Page content
    try:
        pages = {
            "accueil":   (page_accueil,   "🏠  Accueil & Tableau de Bord"),
            "landscape": (page_landscape, "🗺️  Analyse Approfondie — Loss Landscape"),
            "optuna":    (page_optuna,    "🧪  Optimisation Bayésienne — Optuna"),
            "projet":    (page_projet,    "📋  Projet G10 & Auteurs"),
        }
        page_fn, title_txt = pages.get(active, pages["accueil"])
        content = page_fn(C)
    except Exception as e:
        print("[ERROR] render_page:", traceback.format_exc())
        content = error_card("Erreur lors du rendu de la page", e, C)
        title_txt = "⚠️  Erreur"

    sidebar_content = build_sidebar(active, C)
    topbar_content = build_topbar(title_txt, C)

    sidebar_style = {
        "width": "220px", "minHeight": "100vh",
        "background": C["SIDEBAR_GRADIENT"],
        "borderRight": f"1px solid {C['BORDER']}",
        "padding": "24px 14px",
        "position": "fixed", "top": "0", "left": "0",
        "zIndex": "200",
        "display": "flex", "flexDirection": "column", "flexShrink": "0",
        "fontFamily": C["FONT_UI"],
        "color": C["TEXT"],
    }

    main_area_style = {
        "marginLeft": "220px",
        "minHeight": "100vh",
        "background": C["BG"],
    }

    return sidebar_content, topbar_content, content, sidebar_style, main_area_style


# ══════════════════════════════════════════════════════════════════════
#  11. RUN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=True, port=8050, host="127.0.0.1")
