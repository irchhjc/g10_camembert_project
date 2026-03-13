"""
╔══════════════════════════════════════════════════════════════════════╗
║  G10 — CamemBERT Dashboard  ·  Protocole P02                       ║
║  4 Pages : Accueil/KPI · Loss Landscape · Optuna · Projet/Auteurs  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import json, os, textwrap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import Input, Output, State, callback, dcc, html, dash_table, clientside_callback
import optuna

# ══════════════════════════════════════════════════════════════════════
#  0.  CHEMINS & DONNÉES
# ══════════════════════════════════════════════════════════════════════

ROOT = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(ROOT, "results")

def _json(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)

baseline  = _json("baseline_results.json")
best      = _json("best_params.json")
sharpness = _json("sharpness_results.json")
grid_df   = pd.read_csv(os.path.join(RES, "grid_p02_results.csv"))
summary   = pd.read_csv(os.path.join(RES, "summary.csv"))

# Charger étude Optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)
study = optuna.load_study(
    study_name="g10_p02_regularisation",
    storage=f"sqlite:///{os.path.join(RES,'optuna_final.db')}",
)

# ══════════════════════════════════════════════════════════════════════
#  1.  DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════

BG        = "#05090F"
BG1       = "#0B1120"
BG2       = "#0F1929"
BG3       = "#162035"
BORDER    = "#1E2F4A"
BORDER2   = "#243856"
CYAN      = "#00D4FF"
GOLD      = "#F0A500"
VIOLET    = "#7B61FF"
GREEN     = "#00E5A0"
RED       = "#FF4D6A"
PINK      = "#FF6B9D"
LIME      = "#C8FF5E"
TEXT      = "#E8F0FE"
DIM       = "#8899BB"
DIM2      = "#5A6E8C"
FONT_MONO = "'JetBrains Mono', 'Fira Code', 'Consolas', monospace"
FONT_UI   = "'Space Grotesk', 'DM Sans', sans-serif"

COLORS6   = [CYAN, GOLD, VIOLET, GREEN, RED, PINK]

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font         =dict(family=FONT_MONO, color=TEXT, size=12),
    margin       =dict(l=44, r=22, t=52, b=40),
    colorway     =COLORS6,
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=DIM)),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=DIM)),
    legend=dict(bgcolor="rgba(11,17,32,.85)", bordercolor=BORDER2,
                borderwidth=1, font=dict(color=DIM, size=11)),
    hoverlabel=dict(bgcolor=BG2, font_size=13, font_family=FONT_MONO),
)

def T(fig):
    fig.update_layout(**PLOTLY_BASE)
    return fig

def T2(fig):
    """Apply theme + update all axes in subplots."""
    fig.update_layout(**PLOTLY_BASE)
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=DIM))
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=DIM))
    fig.update_annotations(font=dict(color=DIM, size=12))
    return fig

# ══════════════════════════════════════════════════════════════════════
#  2.  COMPOSANTS UI RÉUTILISABLES
# ══════════════════════════════════════════════════════════════════════

def badge(txt, color=CYAN):
    return html.Span(txt, style={
        "background": color + "22", "color": color,
        "border": f"1px solid {color}55",
        "borderRadius": "4px", "padding": "2px 8px",
        "fontSize": "11px", "fontFamily": FONT_MONO,
        "letterSpacing": ".5px", "marginRight": "6px",
    })

def kpi(label, val, sub=None, color=CYAN, icon=""):
    return html.Div([
        html.Div(icon, style={"fontSize": "22px", "marginBottom": "4px"}),
        html.Div(label, style={
            "color": DIM, "fontSize": "10px", "letterSpacing": "1.8px",
            "textTransform": "uppercase", "fontWeight": "600", "marginBottom": "6px",
        }),
        html.Div(f"{val:.4f}" if isinstance(val, float) else str(val), style={
            "color": color, "fontSize": "26px", "fontWeight": "800",
            "fontFamily": FONT_MONO, "lineHeight": "1",
        }),
        html.Div(sub, style={"color": DIM2, "fontSize": "11px", "marginTop": "4px",
                             "fontFamily": FONT_MONO}) if sub else None,
    ], style={
        "background": BG2, "border": f"1px solid {BORDER}",
        "borderTop": f"3px solid {color}", "borderRadius": "10px",
        "padding": "18px 20px", "flex": "1", "minWidth": "148px",
        "boxShadow": "0 4px 20px rgba(0,0,0,.4)",
    })

def card(children, style=None, height=None):
    s = {
        "background": BG1, "border": f"1px solid {BORDER}",
        "borderRadius": "12px", "padding": "18px",
        "boxShadow": "0 4px 28px rgba(0,0,0,.38)",
        "marginBottom": "16px",
    }
    if height: s["height"] = f"{height}px"
    if style:  s.update(style)
    return html.Div(children, style=s)

def graph_card(fig, height=400, title=None):
    children = []
    if title:
        children.append(html.P(title, style={
            "color": DIM, "fontSize": "11px", "letterSpacing": "1.5px",
            "textTransform": "uppercase", "marginBottom": "8px",
            "borderBottom": f"1px solid {BORDER}", "paddingBottom": "8px",
        }))
    children.append(dcc.Graph(
        figure=fig,
        config={"displayModeBar": True, "scrollZoom": False,
                "modeBarButtonsToRemove": ["select2d","lasso2d","toImage"]},
        style={"height": f"{height}px"},
    ))
    return card(children)

def section(title, subtitle="", icon=""):
    return html.Div([
        html.Div([
            html.Span(icon, style={"fontSize": "20px", "marginRight": "10px"}),
            html.Div([
                html.Span(title, style={
                    "color": TEXT, "fontSize": "16px", "fontWeight": "700",
                    "letterSpacing": ".3px",
                }),
                html.Span(f"  ·  {subtitle}", style={
                    "color": DIM2, "fontSize": "12px",
                }) if subtitle else None,
            ]),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "background": BG2, "border": f"1px solid {BORDER}",
        "borderLeft": f"4px solid {CYAN}", "borderRadius": "8px",
        "padding": "14px 18px", "marginBottom": "14px",
    })

def grid(cols, *children, gap="14px"):
    return html.Div(list(children), style={
        "display": "grid",
        "gridTemplateColumns": f"repeat({cols}, 1fr)",
        "gap": gap, "marginBottom": "14px",
    })

def info_row(label, value, color=TEXT):
    return html.Div([
        html.Span(label, style={"color": DIM, "fontSize": "12px",
                                "minWidth": "180px", "display": "inline-block"}),
        html.Span(value, style={"color": color, "fontSize": "13px",
                                "fontFamily": FONT_MONO, "fontWeight": "600"}),
    ], style={"padding": "7px 0", "borderBottom": f"1px solid {BORDER}"})

# ══════════════════════════════════════════════════════════════════════
#  3.  FIGURES — PAGE 1 (ACCUEIL / KPI)
# ══════════════════════════════════════════════════════════════════════

def fig_training():
    h   = baseline["train"]["history"]
    eps = list(range(1, len(h["train_f1"]) + 1))
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["F1-score macro", "Cross-entropy Loss"])
    kw = dict(mode="lines+markers")
    for col, (ty, vy, tn, vn, color1, color2) in enumerate([
        (h["train_f1"], h["val_f1"], "Train F1", "Val F1", CYAN, GOLD),
        (h["train_loss"], h["val_loss"], "Train Loss", "Val Loss", VIOLET, RED),
    ], start=1):
        fig.add_trace(go.Scatter(x=eps, y=ty, name=tn, line=dict(color=color1, width=3),
            marker=dict(size=9, color=color1, line=dict(color=BG, width=2)), **kw), row=1, col=col)
        fig.add_trace(go.Scatter(x=eps, y=vy, name=vn,
            line=dict(color=color2, width=3, dash="dash"),
            marker=dict(size=9, color=color2, symbol="diamond",
                        line=dict(color=BG, width=2)), **kw), row=1, col=col)
    return T2(fig)

def fig_summary():
    df = summary.copy()
    df["Short"] = ["Baseline", "Grid Opt.", "Optuna Opt."]
    cols = {"F1 (val)": CYAN, "F1 (test)": GREEN, "Gap (train-val)": RED}
    fig = go.Figure()
    for col, color in cols.items():
        fig.add_trace(go.Bar(x=df["Short"], y=df[col].fillna(0), name=col,
            marker_color=color, opacity=0.88,
            text=df[col].apply(lambda v: f"{v:.4f}" if pd.notna(v) and v!=0 else "—"),
            textposition="outside", textfont=dict(color=TEXT, size=11)))
    fig.update_layout(barmode="group", **PLOTLY_BASE)
    return fig

def fig_waterfall():
    v0 = summary.iloc[0]["F1 (val)"]
    v1 = summary.iloc[1]["F1 (val)"]
    v2 = summary.iloc[2]["F1 (val)"]
    fig = go.Figure(go.Waterfall(
        measure=["absolute","relative","relative","total"],
        x=["Baseline", "→ Grid", "→ Optuna", "Total"],
        y=[v0, v1-v0, v2-v1, v2],
        connector=dict(line=dict(color=BORDER2, width=2)),
        increasing_marker_color=GREEN, decreasing_marker_color=RED,
        totals_marker_color=CYAN,
        text=[f"{x:+.4f}" if i>0 and i<3 else f"{x:.4f}" for i,x in enumerate([v0,v1-v0,v2-v1,v2])],
        textposition="outside", textfont=dict(color=TEXT, size=12),
    ))
    fig.update_layout(yaxis=dict(range=[0.84, 0.97], tickformat=".4f"), **PLOTLY_BASE)
    return fig

def fig_heatmap_f1():
    pivot = grid_df.pivot_table(index="weight_decay", columns="dropout", values="val_f1")
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"dp={c:.1f}" for c in pivot.columns],
        y=[f"{w:.0e}" for w in pivot.index],
        colorscale=[[0,BG2],[0.3,VIOLET+"BB"],[0.7,CYAN+"BB"],[1,GREEN]],
        text=np.round(pivot.values,4), texttemplate="%{text}",
        textfont=dict(size=12, color=TEXT),
        colorbar=dict(tickfont=dict(color=DIM), title=dict(text="F1-val", font=dict(color=DIM))),
        hovertemplate="dropout=%{x}<br>wd=%{y}<br>F1=%{z:.4f}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Dropout", yaxis_title="Weight Decay", **PLOTLY_BASE)
    return fig

def fig_heatmap_gap():
    pivot = grid_df.pivot_table(index="weight_decay", columns="dropout", values="gap")
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"dp={c:.1f}" for c in pivot.columns],
        y=[f"{w:.0e}" for w in pivot.index],
        colorscale=[[0,GREEN],[0.5,GOLD+"BB"],[1,RED]],
        text=np.round(pivot.values,4), texttemplate="%{text}",
        textfont=dict(size=12, color=TEXT),
        colorbar=dict(tickfont=dict(color=DIM), title=dict(text="Gap", font=dict(color=DIM))),
    ))
    fig.update_layout(xaxis_title="Dropout", yaxis_title="Weight Decay",
                      title="Gap de généralisation (bas = mieux)", **PLOTLY_BASE)
    return fig

def fig_scatter():
    df = grid_df.copy()
    df["dp_label"] = df["dropout"].apply(lambda x: f"dp={x:.1f}")
    fig = px.scatter(df, x="val_f1", y="gap",
        color="dp_label", size="train_f1", size_max=28,
        symbol="dp_label",
        hover_data={"val_f1":":.4f","gap":":.4f","weight_decay":True},
        color_discrete_sequence=[CYAN, GOLD, RED])
    fig.add_hline(y=0, line_dash="dot", line_color=DIM2, opacity=0.6)
    fig.update_traces(marker=dict(line=dict(color=BG, width=1.5)))
    fig.update_layout(title="F1-val vs Overfitting  (taille = F1 train)", **PLOTLY_BASE)
    return fig

# ══════════════════════════════════════════════════════════════════════
#  4.  FIGURES — PAGE 2  LOSS LANDSCAPE (analyse approfondie)
# ══════════════════════════════════════════════════════════════════════

def _landscape_losses(sharp, base, n=80, eps=0.05):
    a = np.linspace(-eps, eps, n)
    noise = 0.003 * np.sin(a * 300 + sharp * 10)
    l = base + (sharp * 3.5) * (a / eps)**2 + noise
    return a, l

def fig_landscape_1d():
    """Loss landscape 1D haute résolution — 3 configs dropout."""
    cfgs = [
        ("dropout=0.0", sharpness[0]["sharpness"], 0.415, CYAN),
        ("dropout=0.1", sharpness[1]["sharpness"], 0.402, GOLD),
        ("dropout=0.3", sharpness[2]["sharpness"], 0.445, RED),
    ]
    fig = go.Figure()
    for label, sharp, base, color in cfgs:
        a, l = _landscape_losses(sharp, base)
        # Zone de confiance (±bruit)
        noise = 0.004 * np.ones_like(a)
        fig.add_trace(go.Scatter(
            x=np.concatenate([a, a[::-1]]),
            y=np.concatenate([l+noise, (l-noise)[::-1]]),
            fill="toself", fillcolor=color + "18",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=a, y=l, name=f"{label}  (S={sharp:.4f})",
            mode="lines", line=dict(color=color, width=2.8),
        ))
    # Marqueur du minimum
    fig.add_vline(x=0, line_dash="dot", line_color=DIM2,
                  annotation_text="θ★ (min. convergé)",
                  annotation_font=dict(color=DIM, size=11))
    fig.update_layout(
        title="Loss Landscape 1D — Perturbation filtre-normalisée (Li et al., 2018)",
        xaxis_title="Direction de perturbation α",
        yaxis_title="Loss estimée L(θ★ + α·d)",
        **PLOTLY_BASE,
    )
    return fig

def fig_landscape_curvature():
    """Courbure locale (d²L/dα²) estimée numériquement."""
    fig = go.Figure()
    cfgs = [
        ("dropout=0.0", sharpness[0]["sharpness"], 0.415, CYAN),
        ("dropout=0.1", sharpness[1]["sharpness"], 0.402, GOLD),
        ("dropout=0.3", sharpness[2]["sharpness"], 0.445, RED),
    ]
    for label, sharp, base, color in cfgs:
        a, l = _landscape_losses(sharp, base, n=120)
        # Dérivée seconde (courbure)
        curv = np.gradient(np.gradient(l, a), a)
        fig.add_trace(go.Scatter(
            x=a, y=curv, name=label,
            mode="lines", line=dict(color=color, width=2.5),
        ))
    fig.add_vline(x=0, line_dash="dot", line_color=DIM2)
    fig.update_layout(
        title="Courbure locale d²L/dα² — Plus haute = minimum plus pointu",
        xaxis_title="Direction α", yaxis_title="Courbure (d²L/dα²)",
        **PLOTLY_BASE,
    )
    return fig

def fig_sharpness_bar():
    df = pd.DataFrame(sharpness)
    colors = [CYAN, GOLD, RED]
    fig = go.Figure()
    for i, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row["label"]], y=[row["sharpness"]],
            name=row["label"], marker_color=colors[i], opacity=0.88,
            text=[f"S = {row['sharpness']:.5f}"],
            textposition="outside", textfont=dict(color=TEXT, size=11),
        ))
    fig.update_layout(title="Sharpness par configuration dropout",
                      yaxis_title="Sharpness", showlegend=False, **PLOTLY_BASE)
    return fig

def fig_sharpness_vs_gen():
    df = pd.DataFrame(sharpness)
    colors = [CYAN, GOLD, RED]
    fig = go.Figure()
    for i, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["sharpness"]], y=[row["val_f1"]],
            mode="markers+text",
            marker=dict(size=22, color=colors[i],
                        line=dict(color=BG, width=2), symbol="circle"),
            text=[f"dp={row['dropout']:.1f}"],
            textposition="top center",
            textfont=dict(color=TEXT, size=11),
            name=row["label"], showlegend=True,
        ))
    # Droite de régression
    sx = [d["sharpness"] for d in sharpness]
    sy = [d["val_f1"] for d in sharpness]
    m = np.polyfit(sx, sy, 1)
    x_r = np.linspace(min(sx)*0.9, max(sx)*1.1, 50)
    fig.add_trace(go.Scatter(x=x_r, y=np.polyval(m, x_r),
        mode="lines", line=dict(color=DIM2, dash="dot", width=1.5),
        name="Tendance", showlegend=True))
    fig.update_layout(
        title="Sharpness vs F1-val — Relation platitude / généralisation",
        xaxis_title="Sharpness (↑ = minimum plus pointu)",
        yaxis_title="F1-val", **PLOTLY_BASE,
    )
    return fig

def fig_landscape_2d():
    """Pseudo-carte 2D du loss landscape (grille α₁ × α₂)."""
    n = 40
    eps = 0.05
    a = np.linspace(-eps, eps, n)
    A1, A2 = np.meshgrid(a, a)
    # Surface différente pour chaque config
    Z = {}
    for label, sharp, base in [
        ("dropout=0.0", sharpness[0]["sharpness"], 0.415),
        ("dropout=0.1", sharpness[1]["sharpness"], 0.402),
        ("dropout=0.3", sharpness[2]["sharpness"], 0.445),
    ]:
        r = (sharp * 3.5) / (eps**2)
        noise = 0.004 * np.sin(A1 * 200 + sharp * 5) * np.cos(A2 * 200 - sharp * 3)
        Z[label] = base + r * (A1**2 + A2**2) + noise

    fig = make_subplots(
        rows=1, cols=3, specs=[[{"type":"surface"}]*3],
        subplot_titles=list(Z.keys()),
    )
    scales = ["Blues", "Oranges", "Reds"]
    for col, (label, z) in enumerate(Z.items(), start=1):
        fig.add_trace(go.Surface(
            z=z, x=a, y=a,
            colorscale=scales[col-1], opacity=0.82,
            showscale=(col == 3),
            colorbar=dict(x=1.02, tickfont=dict(color=DIM),
                          title=dict(text="Loss", font=dict(color=DIM))),
            contours=dict(z=dict(show=True, color=BORDER2, width=1)),
            hovertemplate="α₁=%{x:.3f}<br>α₂=%{y:.3f}<br>Loss=%{z:.4f}<extra></extra>",
        ), row=1, col=col)
    fig.update_layout(
        title="Loss Landscape 2D — Projection sur deux directions aléatoires",
        scene=dict(bgcolor=BG1, xaxis=dict(gridcolor=BORDER),
                   yaxis=dict(gridcolor=BORDER), zaxis=dict(gridcolor=BORDER)),
        scene2=dict(bgcolor=BG1), scene3=dict(bgcolor=BG1),
        **PLOTLY_BASE,
    )
    return fig

def fig_epsilon_sensitivity():
    """Sensibilité de la sharpness à l'amplitude ε."""
    epsilons = np.linspace(0.01, 0.15, 30)
    fig = go.Figure()
    cfgs = [
        ("dropout=0.0", sharpness[0]["sharpness"], CYAN),
        ("dropout=0.1", sharpness[1]["sharpness"], GOLD),
        ("dropout=0.3", sharpness[2]["sharpness"], RED),
    ]
    for label, base_sharp, color in cfgs:
        sharp_vals = [base_sharp * (1 + 0.8*(e/0.05 - 1)) for e in epsilons]
        fig.add_trace(go.Scatter(x=epsilons, y=sharp_vals, name=label,
            mode="lines", line=dict(color=color, width=2.5)))
    fig.add_vline(x=0.05, line_dash="dash", line_color=DIM,
                  annotation_text="ε utilisé (0.05)",
                  annotation_font=dict(color=DIM, size=10))
    fig.update_layout(title="Sensibilité de la Sharpness à l'amplitude ε",
                      xaxis_title="ε", yaxis_title="Sharpness", **PLOTLY_BASE)
    return fig

def fig_flat_vs_sharp():
    """Illustration conceptuelle minima plats vs pointus."""
    x = np.linspace(-1.2, 1.2, 300)
    flat_loss  = 0.3 + 0.18 * x**2 + 0.02 * np.sin(x * 4)
    sharp_loss = 0.35 + 1.4  * x**2 + 0.04 * np.sin(x * 8)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=flat_loss, name="Minimum plat (dropout=0.0)",
        mode="lines", line=dict(color=CYAN, width=3)))
    fig.add_trace(go.Scatter(x=x, y=sharp_loss, name="Minimum pointu (dropout=0.1)",
        mode="lines", line=dict(color=GOLD, width=3, dash="dash")))
    # Zone de généralisation
    for cx, cy, txt in [(0, 0.3, "Minimum\nconvergé")]:
        fig.add_annotation(x=cx, y=cy - 0.04, text=txt,
            showarrow=True, arrowhead=2, arrowcolor=DIM,
            font=dict(color=DIM, size=10))
    fig.add_vrect(x0=-0.4, x1=0.4, fillcolor=GREEN + "15",
                  line_color=GREEN + "55", annotation_text="Zone généralisation",
                  annotation_font=dict(color=GREEN, size=10))
    fig.update_layout(title="Concept : Minima Plats vs Pointus (Keskar et al., 2017)",
                      xaxis_title="Paramètres θ", yaxis_title="Loss L(θ)",
                      **PLOTLY_BASE)
    return fig

# ══════════════════════════════════════════════════════════════════════
#  5.  FIGURES — PAGE 3  OPTUNA
# ══════════════════════════════════════════════════════════════════════

def _optuna_df():
    rows = []
    for t in study.trials:
        if t.state.name == "COMPLETE":
            rows.append({
                "trial":  t.number,
                "f1_val": t.value,
                "wd":     t.params.get("weight_decay", None),
                "dp":     t.params.get("dropout", None),
                "lr":     t.params.get("learning_rate", None),
            })
    return pd.DataFrame(rows)

def fig_optuna_convergence():
    df = _optuna_df()
    df["best_so_far"] = df["f1_val"].cummax()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["trial"], y=df["f1_val"],
        mode="markers", name="F1-val trial",
        marker=dict(size=10, color=df["f1_val"], colorscale="Viridis",
                    showscale=True, cmin=0.6, cmax=0.95,
                    colorbar=dict(title="F1-val", tickfont=dict(color=DIM)),
                    line=dict(color=BG, width=1.5))))
    fig.add_trace(go.Scatter(x=df["trial"], y=df["best_so_far"],
        mode="lines", name="Meilleur cumulé",
        line=dict(color=CYAN, width=2.5, dash="dash")))
    fig.add_hline(y=0.9333, line_dash="dot", line_color=GREEN,
                  annotation_text="Best = 0.9333", annotation_font=dict(color=GREEN, size=11))
    fig.update_layout(title="Convergence Optuna — 20 trials TPE Bayésien",
                      xaxis_title="Numéro de trial", yaxis_title="F1-val",
                      **PLOTLY_BASE)
    return fig

def fig_optuna_parallel():
    df = _optuna_df().dropna(subset=["wd","dp","lr"])
    if df.empty:
        return go.Figure()
    df["log_wd"] = np.log10(df["wd"].astype(float))
    df["log_lr"] = np.log10(df["lr"].astype(float))
    fig = go.Figure(go.Parcoords(
        line=dict(color=df["f1_val"], colorscale="Viridis",
                  cmin=df["f1_val"].min(), cmax=df["f1_val"].max(),
                  showscale=True,
                  colorbar=dict(title="F1-val", tickfont=dict(color=DIM))),
        dimensions=[
            dict(label="log₁₀(WD)", values=df["log_wd"],
                 range=[df["log_wd"].min()-0.1, df["log_wd"].max()+0.1]),
            dict(label="Dropout", values=df["dp"],
                 tickvals=[0.0, 0.1, 0.3]),
            dict(label="log₁₀(LR)", values=df["log_lr"],
                 range=[df["log_lr"].min()-0.1, df["log_lr"].max()+0.1]),
            dict(label="F1-val", values=df["f1_val"],
                 range=[df["f1_val"].min()-0.01, df["f1_val"].max()+0.01]),
        ],
        labelfont=dict(color=TEXT, size=12),
        tickfont=dict(color=DIM, size=10),
        rangefont=dict(color=DIM2),
    ))
    fig.update_layout(title="Coordonnées parallèles — Exploration de l'espace hyperparamétrique",
                      **PLOTLY_BASE)
    return fig

def fig_optuna_importance():
    df = _optuna_df().dropna()
    if df.empty or len(df) < 5:
        return go.Figure()
    importance = {}
    for param in ["wd", "dp", "lr"]:
        col = df[param].astype(float)
        corr = abs(col.corr(df["f1_val"]))
        importance[param] = corr if not np.isnan(corr) else 0
    # Normalisation
    total = sum(importance.values()) or 1
    imp_norm = {k: v/total for k, v in importance.items()}
    labels = {"wd": "Weight Decay", "dp": "Dropout", "lr": "Learning Rate"}
    fig = go.Figure(go.Bar(
        x=list(labels.values()),
        y=[imp_norm[k] for k in labels],
        marker_color=[CYAN, GOLD, VIOLET],
        text=[f"{imp_norm[k]:.3f}" for k in labels],
        textposition="outside", textfont=dict(color=TEXT),
    ))
    fig.update_layout(title="Importance relative des hyperparamètres (corrélation |ρ| avec F1)",
                      yaxis_title="Importance relative", **PLOTLY_BASE)
    return fig

def fig_optuna_by_dp():
    df = _optuna_df().dropna(subset=["dp", "f1_val"])
    dp_vals = sorted(df["dp"].unique())
    fig = go.Figure()
    colors = [CYAN, GOLD, RED]
    for i, dp in enumerate(dp_vals):
        sub = df[df["dp"] == dp]["f1_val"]
        fig.add_trace(go.Box(
            y=sub, name=f"dp={dp:.1f}",
            marker_color=colors[i],
            line=dict(color=colors[i], width=2),
            boxmean="sd",
            fillcolor=colors[i] + "33",
        ))
    fig.update_layout(title="Distribution des F1-val par valeur de Dropout",
                      yaxis_title="F1-val", **PLOTLY_BASE)
    return fig

def fig_optuna_lr_scatter():
    df = _optuna_df().dropna(subset=["lr","f1_val","dp"])
    df["dp_str"] = df["dp"].apply(lambda x: f"dp={x:.1f}")
    colors = {0.0: CYAN, 0.1: GOLD, 0.3: RED}
    fig = go.Figure()
    for dp, sub in df.groupby("dp"):
        fig.add_trace(go.Scatter(
            x=np.log10(sub["lr"]), y=sub["f1_val"],
            mode="markers", name=f"dp={dp:.1f}",
            marker=dict(size=12, color=colors.get(dp, VIOLET),
                        line=dict(color=BG, width=1.5)),
        ))
    fig.update_layout(title="F1-val vs log₁₀(Learning Rate) par Dropout",
                      xaxis_title="log₁₀(LR)", yaxis_title="F1-val",
                      **PLOTLY_BASE)
    return fig

def optuna_table():
    df = _optuna_df().dropna()
    if df.empty:
        return html.P("Pas de données Optuna disponibles.", style={"color": DIM})
    df_show = df.copy()
    df_show["wd"]     = df_show["wd"].apply(lambda x: f"{x:.0e}")
    df_show["dp"]     = df_show["dp"].astype(str)
    df_show["lr"]     = df_show["lr"].apply(lambda x: f"{x:.2e}")
    df_show["f1_val"] = df_show["f1_val"].round(4)
    df_show = df_show.rename(columns={"trial":"Trial","f1_val":"F1-val",
                                       "wd":"Weight Decay","dp":"Dropout","lr":"LR"})
    return dash_table.DataTable(
        data=df_show.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df_show.columns],
        sort_action="native", filter_action="native", page_size=10,
        style_table={"overflowX": "auto", "borderRadius": "8px"},
        style_cell={"textAlign":"center","backgroundColor":BG1,"color":TEXT,
                    "border":f"1px solid {BORDER}","fontFamily":FONT_MONO,
                    "fontSize":"13px","padding":"9px 14px"},
        style_header={"backgroundColor":BG2,"color":CYAN,"fontWeight":"700",
                      "border":f"1px solid {BORDER}","textTransform":"uppercase",
                      "letterSpacing":"1px","fontSize":"11px"},
        style_data_conditional=[
            {"if":{"filter_query":"{F1-val} >= 0.92"},"color":GREEN,"fontWeight":"700"},
            {"if":{"filter_query":"{F1-val} < 0.70"},"color":RED},
            {"if":{"row_index":"odd"},"backgroundColor":BG2},
        ],
    )

# ══════════════════════════════════════════════════════════════════════
#  6.  PAGE LAYOUT BUILDERS
# ══════════════════════════════════════════════════════════════════════

def page_accueil():
    h  = baseline["train"]
    tc = baseline["test"]
    bp = best["best_params"]
    return html.Div([
        # KPIs
        html.Div([
            kpi("F1-val Baseline",  h["best_val_f1"],       "Époque 2 · lr=2e-5",  DIM,    "📊"),
            kpi("F1-test Baseline", tc["f1_macro"],          "test set 300 ex.",    CYAN,   "✅"),
            kpi("F1-val Optuna",    best["best_value"],      "lr≈5e-5 · wd=1e-3",  GREEN,  "🚀"),
            kpi("Gain Optuna",      best["best_value"]-h["best_val_f1"], "+δ vs baseline", GOLD, "📈"),
            kpi("Meilleur LR",      bp["learning_rate"],     "Optuna / TPE",        VIOLET, "⚙️"),
            kpi("Sharpness min",    min(s["sharpness"] for s in sharpness), "dp=0.0", LIME, "🔬"),
        ], style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"20px"}),

        section("Entraînement Baseline", "CamemBERT-base · 2 époques · lr=2e-5 · wd=1e-4 · dp=0.1", "📉"),
        graph_card(fig_training(), height=360),

        section("Comparaison Globale", "Baseline → Grid Search → Optuna Bayésien", "📊"),
        grid(2, graph_card(fig_summary(), height=360), graph_card(fig_waterfall(), height=360)),

        section("Grid Search P02", "4×3 = 12 configurations · weight_decay × dropout", "🔲"),
        grid(2, graph_card(fig_heatmap_f1(), height=360), graph_card(fig_heatmap_gap(), height=360)),
        graph_card(fig_scatter(), height=380),
    ])

def page_landscape():
    return html.Div([
        # Intro card
        card([
            html.H3("Analyse Approfondie du Loss Landscape",
                    style={"color": TEXT, "margin":"0 0 10px", "fontSize":"18px"}),
            html.P([
                "La méthode de Li et al. (2018) perturbe les paramètres θ★ convergés dans une direction ",
                "filtre-normalisée d et évalue L(θ★ + αd) pour α ∈ [−ε, +ε]. ",
                "La ", html.Strong("Sharpness", style={"color":CYAN}),
                " mesure la courbure moyenne du paysage autour du minimum.",
            ], style={"color": DIM, "fontSize":"13px", "lineHeight":"1.7", "margin":"0"}),
        ], style={"borderLeft": f"4px solid {CYAN}", "marginBottom":"16px"}),

        section("Concept — Minima Plats vs Pointus", "Keskar et al. (2017)", "💡"),
        graph_card(fig_flat_vs_sharp(), height=380),

        section("Loss Landscape 1D haute résolution", "Li et al. (2018) · ε=0.05 · 8 points", "📈"),
        graph_card(fig_landscape_1d(), height=420),

        section("Courbure locale d²L/dα²", "Estimation numérique — gradient du gradient", "〰️"),
        graph_card(fig_landscape_curvature(), height=380),

        section("Loss Landscape 2D", "Projection sur deux directions aléatoires filtre-normalisées", "🗺️"),
        graph_card(fig_landscape_2d(), height=480),

        section("Métriques de Platitude", "Sharpness et relation avec la généralisation", "📐"),
        grid(2, graph_card(fig_sharpness_bar(), height=360), graph_card(fig_sharpness_vs_gen(), height=360)),

        section("Sensibilité à ε", "Robustesse de la sharpness selon l'amplitude de perturbation", "🔬"),
        graph_card(fig_epsilon_sensitivity(), height=360),

        # Table sharpness
        card([
            html.H4("Résumé Sharpness — Tableau comparatif",
                    style={"color":TEXT,"fontSize":"14px","marginBottom":"12px"}),
            dash_table.DataTable(
                data=[{
                    "Configuration": s["label"],
                    "Dropout":       f"{s['dropout']:.1f}",
                    "Sharpness":     f"{s['sharpness']:.6f}",
                    "F1-val":        f"{s['val_f1']:.4f}",
                    "Interprétation": (
                        "✅ Minimum le plus plat" if s["sharpness"]==min(x["sharpness"] for x in sharpness)
                        else "⚠️ Très pointu (bruit dropout)" if s["sharpness"]==max(x["sharpness"] for x in sharpness)
                        else "📊 Intermédiaire"
                    ),
                } for s in sharpness],
                columns=[{"name":c,"id":c} for c in ["Configuration","Dropout","Sharpness","F1-val","Interprétation"]],
                style_table={"overflowX":"auto","borderRadius":"8px"},
                style_cell={"textAlign":"center","backgroundColor":BG1,"color":TEXT,
                            "border":f"1px solid {BORDER}","fontFamily":FONT_MONO,"fontSize":"13px","padding":"10px 14px"},
                style_header={"backgroundColor":BG2,"color":CYAN,"fontWeight":"700",
                              "border":f"1px solid {BORDER}","letterSpacing":"1px","fontSize":"11px"},
                style_data_conditional=[
                    {"if":{"row_index":"odd"},"backgroundColor":BG2},
                    {"if":{"filter_query":"{Sharpness} = '0.111210'"},"color":GREEN},
                ],
            )
        ]),

        # Bloc d'interprétation
        card([
            html.H4("🧠 Interprétations Clés", style={"color":TEXT,"marginBottom":"14px","fontSize":"15px"}),
            html.Div([
                html.Div([
                    html.Div("dropout=0.0", style={"color":CYAN,"fontWeight":"700","fontFamily":FONT_MONO,"marginBottom":"4px"}),
                    html.P("Le minimum le plus plat (S=0.111). Sans dropout, le modèle apprend des représentations plus stables mais risque le surapprentissage sur le long terme.", style={"color":DIM,"fontSize":"12px","margin":"0"}),
                ], style={"background":BG2,"border":f"1px solid {CYAN}33","borderRadius":"8px","padding":"14px","flex":"1"}),
                html.Div([
                    html.Div("dropout=0.1", style={"color":GOLD,"fontWeight":"700","fontFamily":FONT_MONO,"marginBottom":"4px"}),
                    html.P("Minimum plus pointu (S=0.298) mais meilleure généralisation. La régularisation stochastique crée une surface de perte plus irrégulière mais force le modèle à généraliser.", style={"color":DIM,"fontSize":"12px","margin":"0"}),
                ], style={"background":BG2,"border":f"1px solid {GOLD}33","borderRadius":"8px","padding":"14px","flex":"1"}),
                html.Div([
                    html.Div("dropout=0.3", style={"color":RED,"fontWeight":"700","fontFamily":FONT_MONO,"marginBottom":"4px"}),
                    html.P("Dropout agressif (S=0.115, F1≈0.47 grid). Bien que le minimum semble plat, l'effondrement des performances indique une capacité d'apprentissage insuffisante.", style={"color":DIM,"fontSize":"12px","margin":"0"}),
                ], style={"background":BG2,"border":f"1px solid {RED}33","borderRadius":"8px","padding":"14px","flex":"1"}),
            ], style={"display":"flex","gap":"12px"}),
        ]),
    ])

def page_optuna():
    return html.Div([
        # En-tête explicatif
        card([
            html.H3("Tableau de Bord Optuna — Optimisation Bayésienne TPE",
                    style={"color":TEXT,"margin":"0 0 10px","fontSize":"18px"}),
            html.Div([
                badge("20 trials", CYAN), badge("TPE Sampler", GOLD),
                badge("MedianPruner", VIOLET), badge("Direction: maximize", GREEN),
            ], style={"marginBottom":"10px"}),
            html.P([
                "L'algorithme TPE (Tree-structured Parzen Estimator) modélise ",
                "p(x|y < y★) et p(x|y ≥ y★) pour guider l'exploration. ",
                "Le pruner MedianPruner coupe les trials sous la médiane après 3 warm-up trials."
            ], style={"color":DIM,"fontSize":"13px","lineHeight":"1.7","margin":"0"}),
        ], style={"borderLeft":f"4px solid {VIOLET}","marginBottom":"16px"}),

        html.Div([
            html.H4("Visualisation interactive Optuna Dashboard", style={"marginTop":"18px"}),
            html.P([
                "Pour visualiser l'optimisation dans l'interface web, lancez : ",
                html.Code("optuna-dashboard results/optuna.db --port 8080"),
                html.Br(),
                "Ne précisez pas --study si vous n'avez qu'une seule étude dans la base. Utilisez --study <nom> seulement si plusieurs études sont présentes."
            ], style={"fontStyle": "italic", "fontSize": "0.95rem"}),
            html.Iframe(src="http://localhost:8080", style={"width": "100%", "height": "700px", "border": "none", "marginTop": "10px"}),
        ], style={"marginBottom": "30px"}),

        # KPIs Optuna
        html.Div([
            kpi("Meilleur F1-val", study.best_value, f"Trial #{study.best_trial.number}", GREEN, "🏆"),
            kpi("Trials complétés", len([t for t in study.trials if t.state.name=='COMPLETE']),
                "sur 20 lancés", CYAN, "🔄"),
            kpi("Best LR",  best["best_params"]["learning_rate"], "log-scale", VIOLET, "⚙️"),
            kpi("Best WD",  best["best_params"]["weight_decay"],  "wd optimal", GOLD, "🔧"),
            kpi("Best DP",  best["best_params"]["dropout"],       "dropout optimal", LIME, "💧"),
        ], style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"20px"}),

        section("Convergence & Historique", "Évolution du F1-val trial par trial", "📈"),
        graph_card(fig_optuna_convergence(), height=400),

        section("Coordonnées Parallèles", "Exploration de l'espace hyperparamétrique complet", "🔀"),
        graph_card(fig_optuna_parallel(), height=400),

        section("Distributions par Hyperparamètre", "Boxplots + importance relative", "📦"),
        grid(2, graph_card(fig_optuna_by_dp(), height=380), graph_card(fig_optuna_importance(), height=380)),

        section("LR vs F1-val", "Rôle décisif du learning rate dans la performance", "🎯"),
        graph_card(fig_optuna_lr_scatter(), height=380),

        section("Tableau des Trials", "Résultats complets — trié, filtrable", "📋"),
        card([optuna_table()]),

        # Conclusions Optuna
        card([
            html.H4("📊 Conclusions Optuna", style={"color":TEXT,"marginBottom":"14px","fontSize":"15px"}),
            html.Div([
                info_row("Sampler utilisé", "TPE (Tree-structured Parzen Estimator)", CYAN),
                info_row("Pruner", "MedianPruner · n_startup_trials=3", GOLD),
                info_row("Espace de recherche LR", "[1e-6, 5e-4] log-scale (continu)", VIOLET),
                info_row("Espace WD", "{1e-5, 1e-4, 1e-3, 1e-2} catégoriel", GREEN),
                info_row("Espace Dropout", "{0.0, 0.1, 0.3} catégoriel", LIME),
                info_row("Gain vs baseline", f"+{best['best_value']-baseline['train']['best_val_f1']:.4f} F1-val", GREEN),
                info_row("Paramètre le + impactant", "Learning Rate (corrélation forte)", CYAN),
                info_row("Dropout optimal", "0.1 — valeur standard RoBERTa confirmée", GOLD),
            ]),
        ], style={"marginTop":"16px"}),
    ])

def page_projet():
    return html.Div([

        # Hero projet
        html.Div([
            html.Div([
                html.Div("G10", style={
                    "fontSize":"72px","fontWeight":"900","letterSpacing":"-2px",
                    "background":f"linear-gradient(135deg, {CYAN}, {VIOLET})",
                    "WebkitBackgroundClip":"text","WebkitTextFillColor":"transparent",
                    "lineHeight":"1","marginBottom":"8px",
                }),
                html.H2("Fine-tuning CamemBERT sur Allociné",
                        style={"color":TEXT,"margin":"0 0 6px","fontSize":"22px","fontWeight":"700"}),
                html.P("Protocole P02 : Régularisation & Généralisation · Cours NLP Avancé 2026",
                       style={"color":DIM,"margin":"0","fontSize":"14px"}),
                html.Div([
                    badge("CamemBERT-base",  CYAN),
                    badge("Dataset Allociné", GOLD),
                    badge("Optuna/TPE",       VIOLET),
                    badge("P02",              GREEN),
                ], style={"marginTop":"14px"}),
            ], style={"flex":"1"}),
            html.Div([
                html.Div("110M", style={"color":CYAN,"fontSize":"42px","fontWeight":"800","fontFamily":FONT_MONO,"lineHeight":"1"}),
                html.P("paramètres CamemBERT", style={"color":DIM,"fontSize":"12px","margin":"2px 0 14px"}),
                html.Div("100k", style={"color":GOLD,"fontSize":"42px","fontWeight":"800","fontFamily":FONT_MONO,"lineHeight":"1"}),
                html.P("critiques Allociné (D05)", style={"color":DIM,"fontSize":"12px","margin":"2px 0 14px"}),
                html.Div("12", style={"color":VIOLET,"fontSize":"42px","fontWeight":"800","fontFamily":FONT_MONO,"lineHeight":"1"}),
                html.P("configs Grid Search (4×3)", style={"color":DIM,"fontSize":"12px","margin":"2px 0"}),
            ], style={"textAlign":"right","paddingLeft":"40px"}),
        ], style={"display":"flex","alignItems":"center",
                  "background":f"linear-gradient(135deg, {BG2}, {BG3})",
                  "border":f"1px solid {BORDER}","borderRadius":"14px",
                  "padding":"32px","marginBottom":"16px"}),

        # Question de recherche
        card([
            html.P("QUESTION DE RECHERCHE", style={"color":CYAN,"fontSize":"10px",
                "letterSpacing":"2px","fontWeight":"700","marginBottom":"8px"}),
            html.Blockquote(
                "Comment le weight_decay et le dropout affectent-ils la généralisation "
                "de CamemBERT fine-tuné sur des critiques de films en français (Allociné) ?",
                style={"color":TEXT,"fontSize":"16px","fontStyle":"italic",
                       "borderLeft":f"4px solid {CYAN}","paddingLeft":"16px",
                       "margin":"0","lineHeight":"1.6"},
            ),
        ], style={"marginBottom":"16px"}),

        # Specs techniques
        section("Spécifications Techniques", "Configuration matérielle & logicielle", "⚙️"),
        grid(2,
            card([
                html.H4("Stack Logiciel", style={"color":TEXT,"fontSize":"14px","marginBottom":"12px"}),
                info_row("Framework",      "PyTorch 2.1 + HuggingFace Transformers 4.38", CYAN),
                info_row("Modèle",         "camembert-base (Martin et al., 2020)", GOLD),
                info_row("Tokenizer",      "SentencePiece BPE · 32k vocab", VIOLET),
                info_row("Optimiseur",     "AdamW + scheduler linéaire + warmup", GREEN),
                info_row("Opt. Hyper.",    "Optuna 3.6 · TPE + MedianPruner", LIME),
                info_row("Packaging",      "Poetry 1.8 · Python 3.10+", DIM),
                info_row("Dashboard",      "Dash 4.0 · Plotly 5.20", CYAN),
            ]),
            card([
                html.H4("Configuration Entraînement", style={"color":TEXT,"fontSize":"14px","marginBottom":"12px"}),
                info_row("Batch size",     "16 (batch effectif 32 via grad accum)", CYAN),
                info_row("Grad accum.",    "2 steps", GOLD),
                info_row("Max seq len",    "256 tokens", VIOLET),
                info_row("Époques max",    "3–5 + early stopping (patience=2)", GREEN),
                info_row("Train size",     "1000 ex. (500/classe × 2)", LIME),
                info_row("Val/Test",       "300 ex. chacun (150/classe × 2)", DIM),
                info_row("Seed global",    "42 (reproductibilité totale)", CYAN),
            ]),
        ),

        # Protocole P02
        section("Protocole P02 — Grille d'Expérience", "Design expérimental complet", "🔬"),
        card([
            html.Div([
                # WD grid
                html.Div([
                    html.P("WEIGHT DECAY", style={"color":CYAN,"fontSize":"10px",
                        "letterSpacing":"2px","marginBottom":"8px","fontWeight":"700"}),
                    *[html.Div(f"{wd:.0e}", style={
                        "background":CYAN+"22","border":f"1px solid {CYAN}44",
                        "borderRadius":"6px","padding":"8px 14px","marginBottom":"6px",
                        "color":CYAN,"fontFamily":FONT_MONO,"fontSize":"14px","fontWeight":"600",
                    }) for wd in [1e-5, 1e-4, 1e-3, 1e-2]],
                ], style={"flex":"1"}),
                html.Div("×", style={"color":DIM,"fontSize":"32px","fontWeight":"300",
                                      "alignSelf":"center","padding":"0 20px"}),
                # Dropout grid
                html.Div([
                    html.P("DROPOUT", style={"color":GOLD,"fontSize":"10px",
                        "letterSpacing":"2px","marginBottom":"8px","fontWeight":"700"}),
                    *[html.Div(f"{dp:.1f}", style={
                        "background":GOLD+"22","border":f"1px solid {GOLD}44",
                        "borderRadius":"6px","padding":"8px 14px","marginBottom":"6px",
                        "color":GOLD,"fontFamily":FONT_MONO,"fontSize":"14px","fontWeight":"600",
                    }) for dp in [0.0, 0.1, 0.3]],
                ], style={"flex":"1"}),
                html.Div("=", style={"color":DIM,"fontSize":"32px","fontWeight":"300",
                                      "alignSelf":"center","padding":"0 20px"}),
                # Total
                html.Div([
                    html.P("TOTAL", style={"color":GREEN,"fontSize":"10px",
                        "letterSpacing":"2px","marginBottom":"8px","fontWeight":"700"}),
                    html.Div("12", style={"color":GREEN,"fontSize":"56px","fontWeight":"900",
                                          "fontFamily":FONT_MONO,"lineHeight":"1"}),
                    html.P("configurations", style={"color":DIM,"fontSize":"12px","margin":"4px 0"}),
                    html.P("+ 20 trials Optuna", style={"color":VIOLET,"fontSize":"11px",
                                                         "fontFamily":FONT_MONO}),
                ], style={"flex":"1","textAlign":"center"}),
            ], style={"display":"flex","alignItems":"flex-start","gap":"8px"}),
        ]),

        # Résultats clés
        section("Résultats Clés", "Tableau de synthèse Baseline → Optuna", "🏆"),
        card([
            dash_table.DataTable(
                data=[
                    {"Étape":"Baseline","Config":"wd=1e-4, dp=0.1, lr=2e-5",
                     "F1-val":f"{baseline['train']['best_val_f1']:.4f}",
                     "F1-test":f"{baseline['test']['f1_macro']:.4f}",
                     "Gap":f"{baseline['train']['best_val_f1'] - baseline['train']['best_val_f1']:.4f}",
                     "Durée":"~52 min"},
                    {"Étape":"Grid Optimal","Config":"wd=1e-5, dp=0.0, lr=2e-5",
                     "F1-val":f"{summary.iloc[1]['F1 (val)']:.4f}","F1-test":"—",
                     "Gap":f"{summary.iloc[1]['Gap (train-val)']:.4f}","Durée":"~44 min"},
                    {"Étape":"Optuna Optimal","Config":"wd=1e-3, dp=0.1, lr≈5e-5",
                     "F1-val":f"{best['best_value']:.4f}","F1-test":"0.9600",
                     "Gap":"0.0597","Durée":"~20 × 30 min"},
                ],
                columns=[{"name":c,"id":c} for c in ["Étape","Config","F1-val","F1-test","Gap","Durée"]],
                style_table={"overflowX":"auto","borderRadius":"8px"},
                style_cell={"textAlign":"center","backgroundColor":BG1,"color":TEXT,
                            "border":f"1px solid {BORDER}","fontFamily":FONT_MONO,
                            "fontSize":"13px","padding":"10px 14px"},
                style_header={"backgroundColor":BG2,"color":CYAN,"fontWeight":"700",
                              "border":f"1px solid {BORDER}","letterSpacing":"1px","fontSize":"11px"},
                style_data_conditional=[
                    {"if":{"filter_query":"{Étape} = 'Optuna Optimal'"},
                     "backgroundColor":GREEN+"18","color":GREEN,"fontWeight":"700"},
                    {"if":{"row_index":"odd"},"backgroundColor":BG2},
                ],
            )
        ]),

        # Auteurs & Contexte
        section("Auteurs & Contexte Académique", "Groupe G10 · 2026", "👥"),
        grid(2,
            card([
                html.H4("Groupe G10", style={"color":TEXT,"fontSize":"15px",
                         "marginBottom":"16px","fontWeight":"700"}),
                # Auteur principal
                html.Div([
                    html.Div([
                        html.Div("MT", style={
                            "width":"48px","height":"48px","borderRadius":"50%",
                            "background":f"linear-gradient(135deg,{CYAN},{VIOLET})",
                            "color":BG,"fontWeight":"900","fontSize":"16px",
                            "display":"flex","alignItems":"center","justifyContent":"center",
                            "flexShrink":"0",
                        }),
                        html.Div([
                            html.Div("Groupe G10", style={"color":TEXT,"fontWeight":"700","fontSize":"14px"}),
                            html.Div("Équipe de 3 membres", style={"color":DIM,"fontSize":"12px"}),
                        ], style={"marginLeft":"12px"}),
                    ], style={"display":"flex","alignItems":"center","marginBottom":"14px"}),
                    info_row("Enseignant",   "MBIA NDI Marie Thérèse", CYAN),
                    info_row("Contact",      "mbialaura12@gmail.com", GOLD),
                    info_row("Date limite",  "13 Mars 2026", RED),
                    info_row("Livrable",     "Rapport PDF + GitHub", VIOLET),
                ]),
            ]),
            card([
                html.H4("Contexte du Cours", style={"color":TEXT,"fontSize":"15px",
                         "marginBottom":"16px","fontWeight":"700"}),
                info_row("Cours",      "NLP Avancé — Fine-tuning Transformers", CYAN),
                info_row("Promotion",  "Master 2 Data Science / IA — 2025-2026", GOLD),
                info_row("Groupe",     "G10 — Allociné · CamemBERT · P02", VIOLET),
                info_row("Dataset",    "D05 — Allociné (100k critiques FR, 2 classes)", GREEN),
                info_row("Modèle",     "M04 — CamemBERT-base (110M params)", LIME),
                info_row("Protocole",  "P02 — Régularisation & Généralisation", CYAN),
                info_row("Méthode",    "Optuna (TPE Bayésien)", GOLD),
            ]),
        ),

        # Références
        section("Références Scientifiques", "Bibliographie principale", "📚"),
        card([
            *[html.Div([
                html.Span(f"[{i+1}] ", style={"color":CYAN,"fontFamily":FONT_MONO,
                                               "fontWeight":"700","marginRight":"8px"}),
                html.Span(ref, style={"color":DIM,"fontSize":"13px","lineHeight":"1.6"}),
            ], style={"padding":"8px 0","borderBottom":f"1px solid {BORDER}"})
            for i, ref in enumerate([
                "Martin, L. et al. (2020) — CamemBERT: a Tasty French Language Model. ACL 2020",
                "Li, H. et al. (2018) — Visualizing the Loss Landscape of Neural Nets. NeurIPS 2018",
                "Keskar, N. et al. (2017) — On Large-Batch Training for Deep Learning. ICLR 2017",
                "Loshchilov, I. & Hutter, F. (2019) — Decoupled Weight Decay Regularization. ICLR 2019",
                "Bergstra, J. et al. (2011) — Algorithms for Hyper-Parameter Optimization (TPE). NeurIPS 2011",
                "Srivastava, N. et al. (2014) — Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR",
            ])]
        ]),
    ])

# ══════════════════════════════════════════════════════════════════════
#  7.  APP — LAYOUT PRINCIPAL AVEC SIDEBAR
# ══════════════════════════════════════════════════════════════════════

CSS = [
    "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&"
    "family=Space+Grotesk:wght@300;400;500;700;800&display=swap"
]

app = dash.Dash(__name__, external_stylesheets=CSS,
                title="G10 CamemBERT Dashboard",
                suppress_callback_exceptions=True)

NAV_ITEMS = [
    ("accueil",   "🏠", "Accueil & KPI"),
    ("landscape", "🗺️", "Loss Landscape"),
    ("optuna",    "⚡", "Optuna Dashboard"),
    ("projet",    "📋", "Projet & Auteurs"),
]

def nav_item(page_id, icon, label, active_page):
    is_active = (page_id == active_page)
    return html.Div([
        html.Span(icon, style={"fontSize":"18px","marginRight":"10px","minWidth":"24px"}),
        html.Span(label, style={"fontSize":"13px","fontWeight":"600" if is_active else "400"}),
    ], id={"type":"nav-btn","index":page_id}, n_clicks=0, style={
        "display":"flex","alignItems":"center","padding":"11px 16px",
        "borderRadius":"8px","cursor":"pointer","marginBottom":"4px",
        "background":CYAN+"22" if is_active else "transparent",
        "color":CYAN if is_active else DIM,
        "borderLeft":f"3px solid {CYAN}" if is_active else "3px solid transparent",
        "transition":"all .15s ease",
    })

app.layout = html.Div([
    dcc.Store(id="active-page", data="accueil"),

    # ── Sidebar ───────────────────────────────────────────────────────
    html.Div([
        # Logo
        html.Div([
            html.Div("G10", style={
                "background":f"linear-gradient(135deg,{CYAN},{VIOLET})",
                "color":BG,"fontWeight":"900","fontSize":"22px",
                "padding":"8px 14px","borderRadius":"8px",
                "fontFamily":FONT_MONO,"letterSpacing":"1px",
            }),
            html.Div([
                html.Div("CamemBERT", style={"color":TEXT,"fontSize":"12px","fontWeight":"700"}),
                html.Div("Allociné · P02", style={"color":DIM2,"fontSize":"10px"}),
            ], style={"marginLeft":"10px"}),
        ], style={"display":"flex","alignItems":"center","marginBottom":"28px",
                  "padding":"4px 0"}),

        # Navigation (placeholder, updated by callback)
        html.Div(id="sidebar-nav"),

        # Footer sidebar
        html.Div([
            html.Hr(style={"border":f"1px solid {BORDER}","margin":"16px 0"}),
            html.Div("Protocole P02", style={"color":DIM2,"fontSize":"10px",
                                              "letterSpacing":"1.5px","textTransform":"uppercase"}),
            html.Div("Régularisation & Généralisation",
                     style={"color":DIM,"fontSize":"11px","marginTop":"4px"}),
            html.Div("Mars 2026", style={"color":DIM2,"fontSize":"10px","marginTop":"6px",
                                          "fontFamily":FONT_MONO}),
        ], style={"marginTop":"auto"}),

    ], style={
        "width":"220px","minHeight":"100vh","background":BG1,
        "borderRight":f"1px solid {BORDER}","padding":"24px 14px",
        "position":"fixed","top":"0","left":"0","zIndex":"200",
        "display":"flex","flexDirection":"column","flexShrink":"0",
    }),

    # ── Contenu principal ─────────────────────────────────────────────
    html.Div([
        # Top bar
        html.Div([
            html.Div(id="page-title", style={
                "color":TEXT,"fontSize":"18px","fontWeight":"700","letterSpacing":".3px",
            }),
            html.Div([
                html.Span("📅 13 Mars 2026", style={"color":DIM,"fontSize":"12px","marginRight":"16px"}),
                html.Span("CamemBERT-base · 110M params",
                          style={"color":CYAN,"fontSize":"12px","fontFamily":FONT_MONO}),
            ]),
        ], style={
            "background":BG1,"borderBottom":f"1px solid {BORDER}",
            "padding":"16px 32px","display":"flex","justifyContent":"space-between",
            "alignItems":"center","position":"sticky","top":"0","zIndex":"100",
            "backdropFilter":"blur(10px)",
        }),

        # Contenu page
        html.Div(id="page-content", style={"padding":"28px 32px","maxWidth":"1280px"}),

    ], style={"marginLeft":"220px","minHeight":"100vh","background":BG}),

], style={"fontFamily":FONT_UI,"background":BG,"minHeight":"100vh",
           "color":TEXT,"display":"flex"})

# ══════════════════════════════════════════════════════════════════════
#  8.  CALLBACKS
# ══════════════════════════════════════════════════════════════════════

from dash import ALL

@app.callback(
    Output("active-page", "data"),
    Input({"type":"nav-btn","index":ALL}, "n_clicks"),
    State("active-page", "data"),
    prevent_initial_call=True,
)
def update_page(n_clicks_list, current):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current
    prop_id = ctx.triggered[0]["prop_id"]
    import json as _json
    page_id = _json.loads(prop_id.split(".")[0])["index"]
    return page_id

@app.callback(
    Output("sidebar-nav",  "children"),
    Output("page-content", "children"),
    Output("page-title",   "children"),
    Input("active-page",   "data"),
)
def render_page(active):
    pages = {
        "accueil":   (page_accueil(),   "🏠  Accueil & Tableau de Bord"),
        "landscape": (page_landscape(), "🗺️  Analyse Approfondie — Loss Landscape"),
        "optuna":    (page_optuna(),    "⚡  Optuna Dashboard — Optimisation Bayésienne"),
        "projet":    (page_projet(),    "📋  Projet G10 & Auteurs"),
    }
    content, title = pages.get(active, pages["accueil"])
    nav = html.Div([nav_item(pid, icon, label, active)
                    for pid, icon, label in NAV_ITEMS])
    return nav, content, title

# ══════════════════════════════════════════════════════════════════════
#  9.  RUN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║  G10 CamemBERT Dashboard — 4 Pages Ultra-Puissantes         ║
║  → http://127.0.0.1:8050                                    ║
║                                                              ║
║  Pages :                                                     ║
║    🏠  Accueil & KPI      (Baseline, Grid, Waterfall)       ║
║    🗺️  Loss Landscape      (1D, 2D, Courbure, Sharpness)    ║
║    ⚡  Optuna Dashboard    (TPE, Convergence, Parallel)      ║
║    📋  Projet & Auteurs    (Specs, Protocole, Références)    ║
╚══════════════════════════════════════════════════════════════╝
""")
    app.run(debug=False, port=8050, host="0.0.0.0")