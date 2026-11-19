"""Palette et habillage de l'appli Dash : couleurs, figures Plotly, AG Grid, mini-courbes.

C'est le rôle de `.streamlit/config.toml` côté Streamlit : un seul endroit où les couleurs sont
nommées. Les styles de mise en page (cartes, bandeau, barre latérale) sont dans assets/style.css.
"""

import textwrap

import dash_ag_grid as dag
import plotly.express as px
import plotly.graph_objects as go

# Palette : les mêmes rôles que dans l'appli Streamlit.
VERT, ROUGE, ORANGE, NEUTRE, BLEU = "#1F7A5C", "#C8553D", "#E0A526", "#8FA3B5", "#4C7FA8"
VERT_CLAIR, ENCRE, TRAIT = "#2E8B67", "#16241E", "#6B7B73"
POLICE = "Segoe UI, system-ui, -apple-system, Roboto, sans-serif"

COULEURS_VERDICT = {
    "À travailler": ROUGE,
    "À surveiller": ORANGE,
    "Dans la moyenne": NEUTRE,
    "Solide": VERT_CLAIR,
}
ORDRE_VERDICT = list(COULEURS_VERDICT)
FR_COULEUR = {"white": "blancs", "black": "noirs"}
FR_RESULTAT = {"victoire": "victoire", "nulle": "nulle", "défaite": "défaite"}

LEGENDE_VERDICT = (
    "**Rouge** : sous 50 % de façon nette. **Orange** : sous 50 %, mais cela peut être du hasard. "
    "**Gris** : dans la moyenne. **Vert** : au-dessus de 50 % de façon nette. "
    "Le trait au bout de chaque barre est la marge d'erreur à 95 % : s'il touche la ligne "
    "des 50 %, on ne peut pas conclure. Le nom d'une ouverture décrit la partie, pas votre choix : "
    "avec les noirs, c'est souvent l'adversaire qui l'a imposée."
)

# dcc.Graph : pas de barre d'outils, le graphique suit la largeur de sa carte.
CONFIG_GRAPHIQUE = {"displayModeBar": False, "responsive": True}
CONFIG_MINI = {"staticPlot": True, "responsive": True}


def style(fig: go.Figure, hauteur: int = 420, legende: bool = False) -> go.Figure:
    """Même habillage pour toutes les figures : fond transparent, police sobre, grille discrète.

    `legende` affiche la légende des couleurs au-dessus du graphique.
    """
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legende,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title": None},
        height=hauteur,
        margin={"l": 0, "r": 16, "t": 90 if legende else 24, "b": 0},
        font={"family": POLICE, "size": 13, "color": TRAIT},
        bargap=0.28,
        hoverlabel={"font_family": POLICE},
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(107,123,115,0.2)", zeroline=False)
    return fig


def vide(message: str, hauteur: int = 240) -> go.Figure:
    """Une figure qui ne montre que `message` : l'état « rien à afficher » d'un graphique."""
    fig = go.Figure()
    fig.add_annotation(
        text="<br>".join(textwrap.wrap(message, 38)),
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, hauteur)


def barres_verdict(d, x: str, hauteur: int = 380, **libelles) -> go.Figure:
    """Barres verticales de scores avec marge d'erreur, colorées par verdict, ligne à 50 %."""
    fig = px.bar(
        d,
        x=x,
        y="score",
        error_y="marge",
        color="verdict",
        color_discrete_map=COULEURS_VERDICT,
        category_orders={"verdict": ORDRE_VERDICT},
        hover_data={"n": True, "score": ":.0%", "marge": ":.0%", "verdict": False},
        labels={"score": "score moyen", "n": "parties", "marge": "± marge", **libelles},
    )
    fig.update_traces(error_y_color=TRAIT, error_y_thickness=1.5)
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.add_hline(y=0.5, line_dash="dot", line_color=TRAIT)
    return style(fig, hauteur, legende=True)


def mini_courbe(serie, genre: str, couleur: str = VERT) -> go.Figure:
    """La mini-courbe d'une carte-chiffre : une valeur par mois, sans axes ni grille."""
    x, y = list(serie.index), list(serie.to_numpy())
    fig = go.Figure()
    if genre == "bar":
        fig.add_bar(x=x, y=y, marker_color=couleur, marker_opacity=0.55)
        fig.update_layout(bargap=0.25)
    else:
        remplissage = "tozeroy" if genre == "area" else "none"
        fig.add_scatter(
            x=x,
            y=y,
            mode="lines",
            line={"color": couleur, "width": 2, "shape": "spline"},
            fill=remplissage,
            fillcolor="rgba(31,122,92,0.14)",
        )
    pas = (max(y) - min(y)) * 0.15 or 1
    fig.update_yaxes(visible=False, range=[min(y) - pas, max(y) + pas])
    fig.update_xaxes(visible=False)
    fig.update_traces(hovertemplate="%{x|%b %Y} : %{y:,.2f}<extra></extra>")
    fig.update_layout(
        showlegend=False,
        height=64,
        margin={"l": 0, "r": 0, "t": 4, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": POLICE},
    )
    return fig


# AG Grid : le thème Balham (celui de AGENTS.md) aux couleurs de l'appli.
THEME_GRILLE = {
    "function": "themeBalham.withParams({"
    f"accentColor: '{VERT}', headerBackgroundColor: '#EAF0EC', headerTextColor: '{ENCRE}', "
    "oddRowBackgroundColor: 'rgba(31, 122, 92, 0.035)', borderColor: '#DCE4DF', "
    "fontFamily: 'inherit', rowHoverColor: 'rgba(31, 122, 92, 0.08)'})"
}


def grille(
    identifiant: str,
    colonnes: list[dict],
    *,
    selection: bool = False,
    lignes: int = 10,
    id_ligne: str | None = None,
):
    """Un dag.AgGrid avec les réglages communs : thème, pagination, tri et filtre sur chaque colonne.

    `selection=True` : un clic sur une ligne la sélectionne (propriété `selectedRows`).
    `id_ligne` : la colonne qui identifie une ligne, pour que la sélection survive au tri.
    """
    options = {
        "theme": THEME_GRILLE,
        "animateRows": True,
        "pagination": True,
        "paginationPageSize": lignes,
        "paginationPageSizeSelector": False,
        "domLayout": "autoHeight",  # la grille a la hauteur de ses lignes, pas de vide en dessous
        "suppressCellFocus": True,
    }
    if id_ligne:
        options["getRowId"] = f"params.data.{id_ligne}"
    if selection:
        options["rowSelection"] = {
            "mode": "singleRow",
            "enableClickSelection": True,
            "checkboxes": False,
        }
    return dag.AgGrid(
        id=identifiant,
        columnDefs=colonnes,
        rowData=[],
        columnSize="responsiveSizeToFit",
        defaultColDef={"filter": True, "sortable": True},
        dashGridOptions=options,
        style={"height": None},
        className="grille",
    )


def format_d3(motif: str) -> dict:
    """Un valueFormatter d'AG Grid qui applique un motif d3-format (« .0% », « .2f »...)."""
    return {"function": f"params.value == null ? '' : d3.format('{motif}')(params.value)"}


def prechauffer_plotly() -> None:
    """Construit une figure de chaque type utilisé, une fois, au démarrage.

    Dash lance les callbacks d'une page en parallèle. Plotly Express charge son gabarit à la
    demande, et deux threads qui le font en même temps se marchent dessus (« Invalid value »).
    """
    donnees = {"x": [1, 2], "y": [1, 2], "c": ["a", "b"]}
    for fabrique in (px.bar, px.line):
        fabrique(donnees, x="x", y="y", color="c", error_y=None)
        fabrique(donnees, x="x", y="y")
    px.bar(donnees, x="y", y="x", orientation="h", error_x=None)
