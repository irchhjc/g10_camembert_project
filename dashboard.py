import dash
from dash import dcc, html, dash_table
import plotly.express as px
import pandas as pd
import json
import os


# Détection du dossier racine du projet (là où se trouve ce script)
project_root = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(project_root, 'results')
summary_path = os.path.join(results_dir, 'summary.csv')
grid_path = os.path.join(results_dir, 'grid_p02_results.csv')
best_params_path = os.path.join(results_dir, 'best_params.json')
baseline_path = os.path.join(results_dir, 'baseline_results.json')
sharpness_path = os.path.join(results_dir, 'sharpness_results.json')


# Chargement summary.csv (tableau)
summary_df = pd.read_csv(summary_path)

# Chargement grid_p02_results.csv (tableau)
grid_df = pd.read_csv(grid_path)

# Chargement best_params.json (clé best_params)
with open(best_params_path) as f:
    best_params_json = json.load(f)
best_params = best_params_json["best_params"]
best_params["best_value"] = best_params_json["best_value"]

# Chargement baseline_results.json (clé config + train)
with open(baseline_path) as f:
    baseline_json = json.load(f)
baseline_results = {
    **{f"config_{k}": v for k, v in baseline_json["config"].items()},
    **{f"train_{k}": v for k, v in baseline_json["train"].items()}
}

# Chargement sharpness_results.json (liste d'objets)
with open(sharpness_path) as f:
    sharpness_results = json.load(f)

# Création de l'app Dash
app = dash.Dash(__name__)


# --- App Dash ---
app = dash.Dash(__name__)
app.css.append_css({"external_url": [
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css",
    "https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap"
]})
CUSTOM_STYLE = {
    'fontFamily': 'Montserrat, sans-serif',
    'backgroundColor': '#f8f9fa',
    'color': '#222',
    'padding': '2rem'
}

# --- Layout enrichi pour G10 ---
app.layout = html.Div([
    html.Div([
        html.H1("Projet G10 : Fine-tuning Transformers, Optimisation & Loss Landscape", style={'textAlign': 'center', 'fontWeight': '700', 'marginBottom': '1.5rem'}),
        html.P([
            html.B("MBIA NDI Marie Thérèse"), html.Br(),
            "mbialaura12@gmail.com", html.Br(),
            "14 février 2026"
        ], style={'textAlign': 'center', 'fontSize': '1.1rem'}),
        html.Hr(),
        html.H3("Résumé du projet", style={'marginTop': '1rem'}),
        html.P("Ce projet vise à appliquer l’optimisation d’hyperparamètres et l’analyse du loss landscape au fine-tuning de Transformers pour la classification de textes. Focus sur la régularisation (dropout, weight decay), la généralisation, et l’adaptation aux contraintes matérielles (CPU, RAM, internet)."),
        html.H4("Objectifs pédagogiques"),
        html.Ul([
            html.Li("Maîtriser le fine-tuning de Transformers (HuggingFace)"),
            html.Li("Optimiser les hyperparamètres (Optuna, Grid Search)"),
            html.Li("Analyser la généralisation et la régularisation"),
            html.Li("Visualiser le loss landscape et la platitude des minima"),
            html.Li("Adapter les techniques aux contraintes matérielles réelles")
        ]),
        html.H4("Contraintes matérielles & adaptation"),
        dash_table.DataTable(
            data=[
                {"Contrainte": "Pas de GPU", "Adaptation": "Modèles réduits, CPU optimisé"},
                {"Contrainte": "RAM < 8Go", "Adaptation": "Sous-échantillonnage, batch_size réduit"},
                {"Contrainte": "Connexion limitée", "Adaptation": "Modèles pré-téléchargés, local"},
                {"Contrainte": "Temps limité", "Adaptation": "Early stopping, epochs réduites"}
            ],
            columns=[{"name": i, "id": i} for i in ["Contrainte", "Adaptation"]],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center', 'fontFamily': 'Montserrat'},
            style_header={'backgroundColor': '#343a40', 'color': 'white', 'fontWeight': 'bold'}
        ),
        html.H4("Configuration G10"),
        dash_table.DataTable(
            data=[{"Groupe": "G10", "Dataset": "Allociné", "Modèle": "CamemBERT", "Problématique": "Régularisation", "Méthode": "Optuna"}],
            columns=[{"name": i, "id": i} for i in ["Groupe", "Dataset", "Modèle", "Problématique", "Méthode"]],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center', 'fontFamily': 'Montserrat'},
            style_header={'backgroundColor': '#343a40', 'color': 'white', 'fontWeight': 'bold'}
        ),
    ], className="container"),
    html.Hr(),
    html.Div([
        html.H2("Optimisation d'hyperparamètres", style={'fontWeight': '700'}),
        html.H4("Meilleurs paramètres (Optuna/Grid)", style={'marginTop': '1rem'}),
        dash_table.DataTable(
            data=[best_params],
            columns=[{"name": i, "id": i} for i in best_params.keys()],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center', 'fontFamily': 'Montserrat'},
            style_header={'backgroundColor': '#343a40', 'color': 'white', 'fontWeight': 'bold'}
        ),
        html.H4("Résultats Baseline", style={'marginTop': '2rem'}),
        dash_table.DataTable(
            data=[baseline_results],
            columns=[{"name": i, "id": i} for i in baseline_results.keys()],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center', 'fontFamily': 'Montserrat'},
            style_header={'backgroundColor': '#343a40', 'color': 'white', 'fontWeight': 'bold'}
        ),
        html.H4("Comparatif F1 & Gap", style={'marginTop': '2rem'}),
        dcc.Graph(
            id='f1_gap_bar',
            figure=px.bar(
                summary_df,
                x='Configuration',
                y=['F1 (val)', 'F1 (test)', 'Gap (train-val)'],
                barmode='group',
                title='F1 (val), F1 (test) et Gap par configuration',
                labels={'value': 'Score', 'Configuration': 'Configuration', 'variable': 'Métrique'}
            )
        ),
    ], className="container"),
    html.Hr(),
    html.Div([
        html.H2("Grid Search : Dropout & Weight Decay", style={'fontWeight': '700'}),
        dcc.Graph(
            id='heatmap',
            figure=px.density_heatmap(
                grid_df,
                x='dropout',
                y='weight_decay',
                z='val_f1',
                color_continuous_scale='Viridis',
                title='Validation F1 selon Dropout et Weight Decay'
            )
        ),
        dcc.Graph(
            id='scatter',
            figure=px.scatter(
                grid_df,
                x='dropout',
                y='weight_decay',
                size='val_f1',
                color='gap',
                hover_data=['val_f1', 'gap'],
                color_continuous_scale='Plasma',
                title='F1 & Gap selon Dropout et Weight Decay'
            )
        ),
        html.H4("Tableau complet des résultats Grid Search", style={'marginTop': '2rem'}),
        dash_table.DataTable(
            data=grid_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in grid_df.columns],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center', 'fontFamily': 'Montserrat'},
            style_header={'backgroundColor': '#343a40', 'color': 'white', 'fontWeight': 'bold'}
        )
    ], className="container"),
    html.Hr(),
    html.Div([
        html.H2("Analyse du Loss Landscape & Platitude", style={'fontWeight': '700'}),
        html.H4("Sharpness des minima (platitude)", style={'marginTop': '1rem'}),
        dash_table.DataTable(
            data=sharpness_results,
            columns=[{"name": i, "id": i} for i in sharpness_results[0].keys()],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center', 'fontFamily': 'Montserrat'},
            style_header={'backgroundColor': '#343a40', 'color': 'white', 'fontWeight': 'bold'}
        ),
        html.P("Sharpness = moyenne des variations de loss autour du minimum (voir rapport pour la formule)", style={'fontStyle': 'italic', 'fontSize': '0.95rem'}),
    ], className="container"),
    html.Hr(),
    html.Div([
        html.H4("Contacts & Ressources", style={'marginTop': '1rem'}),
        html.P([
            "Enseignant : mbialaura12@gmail.com", html.Br(),
            "Livrables : rapport + code sur GitHub"
        ])
    ], className="container"),
], style=CUSTOM_STYLE)

if __name__ == "__main__":
    app.run(debug=True, port=8050)
