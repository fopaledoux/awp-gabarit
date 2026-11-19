"""Exo 4 : la même appli en Dash, avec un drill-down. Environ 25 min.

Question : quelles ouvertures coûtent des points, et dans quelles parties ?
À produire : un curseur (parties minimum) et le graphique des familles ; un clic sur une barre
liste les parties, un clic sur une partie liste les coups jugés mauvais, dans deux AG Grid.
Lancer : uv run python fil_rouge/app_dash.py   (http://127.0.0.1:8050)

C'est la même appli que celle de l'exo 2 (mêmes cinq onglets, mêmes textes d'aide à la décision) :
les calculs sont dans analyses.py, les couleurs et les figures communes dans theme.py, le style
des cartes dans assets/style.css. Ici : la mise en page et les callbacks.

Comment ça marche : les filtres de la barre latérale alimentent un `dcc.Store` (juste des
réglages, quelques octets). Chaque carte a son callback, qui lit ce Store et ne touche qu'à ses
sorties. Les données se lisent dans les callbacks (jamais en haut du fichier) et sont gardées en
mémoire tant que les CSV ne changent pas.
"""

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
import plotly.express as px
from analyses import (
    GRAVITES,
    PHASES,
    POSITIONNELLE,
    TYPES,
    elo_mensuel,
    elo_series,
    fins_de_partie,
    gaffes_par_phase,
    lecture_couleur,
    lecture_ecart,
    lecture_echantillon,
    lecture_elo,
    lecture_fins,
    lecture_heures,
    lecture_ouvertures,
    lecture_phases,
    lecture_score,
    lecture_temps,
    lecture_types,
    load_phases,
    ouvertures,
    par_ecart,
    recommandations,
    score_glissant,
    score_par,
    temps_restant,
    tendance_elo,
    types_gaffes,
)
from dash import Dash, Input, Output, callback, dcc, html
from theme import (
    CONFIG_GRAPHIQUE,
    CONFIG_MINI,
    COULEURS_VERDICT,
    ENCRE,
    FR_COULEUR,
    LEGENDE_VERDICT,
    NEUTRE,
    ORANGE,
    ORDRE_VERDICT,
    ROUGE,
    TRAIT,
    VERT,
    VERT_CLAIR,
    barres_verdict,
    format_d3,
    grille,
    mini_courbe,
    prechauffer_plotly,
    style,
    vide,
)

from data import DATA, load_coups, load_data

app = Dash(__name__, title="Mes parties d'échecs")
server = app.server  # Dash tourne sur Flask
prechauffer_plotly()

COULEURS = {"toutes": None, "blancs": "white", "noirs": "black"}
ORDRE_PHASES = ["ouverture", "milieu de jeu", "finale"]
MSG_VIDE = "Aucune partie ne correspond à ces filtres."
FILTRES = Input("filtres-store", "data")  # toutes les cartes lisent les mêmes filtres


# ------------------------------------------------------------------ les données


@lru_cache(maxsize=1)
def _lire(version: tuple[float, ...]):
    return load_data(), load_coups(), load_phases()


def _version() -> tuple[float, ...]:
    """La date de modification des CSV : si l'un change, on relit tout au prochain callback."""
    fichiers = ("parties.csv", "coups_juges.csv", "phases.csv")
    return tuple((DATA / nom).stat().st_mtime for nom in fichiers)


def charger():
    """(parties, coups jugés, phases). Lues dans les callbacks, pas à l'import."""
    return _lire(_version())


@dataclass(frozen=True)
class Selection:
    """Ce que voient tous les onglets : les parties filtrées et ce qui en découle."""

    tout: pd.DataFrame
    vue: pd.DataFrame
    coups: pd.DataFrame
    phases: pd.DataFrame
    f: pd.DataFrame
    min_n: int


@lru_cache(maxsize=16)
def _selection(cadences, debut, fin, couleur, min_n, version) -> Selection:
    df, coups, phases = charger()
    jours = df["date_jeu"].dt.date
    vue = df[
        df["cadence"].isin(cadences)
        & jours.between(pd.Timestamp(debut).date(), pd.Timestamp(fin).date())
    ]
    if COULEURS[couleur]:
        vue = vue[vue["couleur"] == COULEURS[couleur]]
    ids = vue["partie_id"]
    return Selection(
        df,
        vue,
        coups[coups["partie_id"].isin(ids)],
        phases[phases["partie_id"].isin(ids)],
        ouvertures(vue, min_n),
        min_n,
    )


def selection(filtres: dict) -> Selection:
    """Les parties qui passent les filtres. Le résultat est mis en cache : ne le modifiez pas."""
    f = filtres
    return _selection(
        tuple(f["cadences"]), f["debut"], f["fin"], f["couleur"], f["min_n"], _version()
    )


# ------------------------------------------------------------------ les petits composants


def lecture(texte: str | None, niveau: str = "info"):
    """L'encadré qui dit quoi conclure du graphique (rien s'il n'y a pas de texte)."""
    if not texte:
        return None
    return html.Div([dcc.Markdown(texte)], className=f"lecture lecture-{niveau}")


def carte(titre: str, *corps, classe: str = ""):
    return html.Section([html.H3(titre), *corps], className=f"carte {classe}".strip())


def graphique(identifiant: str, hauteur: int = 380):
    return dcc.Loading(
        dcc.Graph(
            id=identifiant, config=CONFIG_GRAPHIQUE, responsive=True, style={"height": hauteur}
        ),
        type="circle",
        color=VERT,
        delay_show=300,
    )


def zone(identifiant: str):
    """Un emplacement que le callback remplit (un encadré de lecture, une note)."""
    return html.Div(id=identifiant)


def note(texte: str):
    return html.P(texte, className="note")


def aide(titre: str, texte: str):
    return html.Details([html.Summary(titre), dcc.Markdown(texte)], className="aide")


def kpi(titre, valeur, delta, ton, courbe, aide_texte):
    """Une carte-chiffre : le chiffre, sa comparaison à l'ensemble, son évolution mois par mois."""
    return html.Div(
        [
            html.Div(titre, className="kpi-titre"),
            html.Div(valeur, className="kpi-valeur"),
            html.Div(delta, className=f"kpi-delta kpi-{ton}"),
            dcc.Graph(figure=courbe, config=CONFIG_MINI, style={"height": 64}) if courbe else None,
        ],
        className="kpi",
        title=aide_texte,
    )


def courbe(vue: pd.DataFrame, colonne: str, agg: str, genre: str):
    """La mini-courbe d'une carte-chiffre : une valeur par mois (rien s'il y a moins de 3 mois)."""
    mois = vue.resample("MS", on="date_jeu")[colonne].agg(agg).dropna()
    return mini_courbe(mois, genre) if len(mois) >= 3 else None


def ton_delta(ecart: float, seuil: float, hausse_est_bonne: bool = True) -> str:
    """bon, mauvais ou neutre : plus de score est bon, plus de gaffes ne l'est pas."""
    if abs(ecart) < seuil:
        return "neutre"
    return "bon" if (ecart > 0) == hausse_est_bonne else "mauvais"


ICONES = {"menu_book": "📖", "shield": "🛡️", "bolt": "⚡", "psychology": "🧠", "target": "🎯"}


# ------------------------------------------------------------------ les figures partagées


def figure_familles(f: pd.DataFrame, choisie: str | None = None):
    """Score par famille d'ouvertures, la pire en haut, avec marge d'erreur et verdict."""
    fig = px.bar(
        f,
        x="score",
        y="famille",
        orientation="h",
        error_x="marge",
        color="verdict",
        color_discrete_map=COULEURS_VERDICT,
        category_orders={"verdict": ORDRE_VERDICT, "famille": list(f["famille"])},
        text="score",
        hover_data={"n": True, "gaffes": ":.2f", "score": ":.0%", "marge": ":.0%"},
        labels={
            "score": "score moyen",
            "famille": "",
            "n": "parties",
            "gaffes": "gaffes/partie",
            "marge": "± marge",
        },
    )
    fig.update_traces(
        texttemplate="%{x:.0%}",
        textposition="inside",
        insidetextanchor="start",
        textfont={"color": ENCRE, "size": 12},
        error_x_color=TRAIT,
        error_x_thickness=1.5,
    )
    if choisie:  # un contour noir autour de la barre dont on affiche les parties
        fig.for_each_trace(
            lambda t: t.update(
                marker_line_width=[3 if y == choisie else 0 for y in t.y], marker_line_color=ENCRE
            )
        )
    fig.update_yaxes(autorange="reversed")  # la pire en haut
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.add_vline(x=0.5, line_dash="dot", line_color=TRAIT)
    return style(fig, max(380, 30 * len(f) + 120), legende=True)


def nombre_gaffes(s: Selection) -> pd.DataFrame:
    """Les gaffes par phase : leur nombre (coups jugés) et leur taux pour 100 coups (phases)."""
    taux = gaffes_par_phase(s.phases)
    n = (
        s.coups[s.coups["jugement"] == "blunder"]["phase"]
        .map(PHASES)
        .value_counts()
        .reindex(ORDRE_PHASES, fill_value=0)
    )
    return taux.assign(gaffes=n.to_numpy())


def figure_phases(taux: pd.DataFrame, colonne: str, libelle: str, motif: str, hauteur: int):
    """Barres par phase, la plus lourde en rouge."""
    pire = taux.loc[taux[colonne].idxmax()]
    fig = px.bar(
        taux,
        x="phase",
        y=colonne,
        text=colonne,
        category_orders={"phase": ORDRE_PHASES},
        hover_data={"taux": ":.1f"},
        labels={"phase": "", colonne: libelle, "taux": "pour 100 de mes coups"},
    )
    fig.update_traces(
        marker_color=[ROUGE if p == pire["phase"] else NEUTRE for p in taux["phase"]],
        texttemplate=motif,
        textposition="outside",
        cliponaxis=False,
    )
    return style(fig, hauteur)


# ------------------------------------------------------------------ la mise en page


def onglet(titre: str, valeur: str, *contenu):
    return dcc.Tab(
        list(contenu),
        label=titre,
        value=valeur,
        className="onglet",
        selected_className="onglet onglet-actif",
    )


def deux(*cartes):
    return html.Div(list(cartes), className="grille-2")


def synthese():
    return onglet(
        "Synthèse",
        "synthese",
        html.H2("Que faire cette semaine ?"),
        html.Div(id="actions-cartes", className="grille-3"),
        zone("score-lecture"),
        deux(
            carte(
                "Score par ouverture : les pires en haut",
                graphique("synthese-familles-graph", 520),
                zone("synthese-familles-lecture"),
                aide("Comment lire ce graphique ?", LEGENDE_VERDICT),
                note(
                    "Pour voir les parties d'une ouverture, cliquez sa barre dans l'onglet Ouvertures."
                ),
            ),
            carte(
                "Où tombent mes gaffes",
                graphique("synthese-phases-graph", 520),
                zone("synthese-phases-lecture"),
            ),
        ),
    )


COLONNE_SCORE = {
    "field": "score",
    "headerName": "Score",
    "flex": 1.5,
    "valueFormatter": format_d3(".0%"),
    # une barre de fond proportionnelle au score, comme la ProgressColumn de Streamlit
    "cellStyle": {
        "function": "({background: 'linear-gradient(90deg, rgba(31,122,92,0.28) ' "
        "+ Math.round(params.value * 100) + '%, transparent ' "
        "+ Math.round(params.value * 100) + '%)'})"
    },
}
COLONNE_GAFFES = {
    "field": "gaffes",
    "headerName": "Gaffes/partie",
    "flex": 1,
    "valueFormatter": format_d3(".2f"),
}
COLONNES_FAMILLES = [
    {"field": "famille", "headerName": "Ouverture", "flex": 2},
    {
        "field": "verdict",
        "headerName": "Verdict",
        "flex": 1.5,
        "cellClassRules": {
            "verdict-travailler": "params.value === 'À travailler'",
            "verdict-surveiller": "params.value === 'À surveiller'",
            "verdict-solide": "params.value === 'Solide'",
        },
    },
    {"field": "n", "headerName": "Parties", "flex": 1},
    COLONNE_SCORE,
    {"field": "marge", "headerName": "± marge", "flex": 1, "valueFormatter": format_d3(".0%")},
    COLONNE_GAFFES,
]
COLONNES_VARIANTES = [
    {"field": "ouverture_nom", "headerName": "Variante", "flex": 3},
    {"field": "n", "headerName": "Parties", "flex": 1},
    COLONNE_SCORE,
    COLONNE_GAFFES,
]
COLONNES_PARTIES = [
    {"field": "date", "headerName": "Date", "flex": 1.2},
    {"field": "resultat", "headerName": "Résultat", "flex": 1},
    {"field": "couleur", "headerName": "Couleur", "flex": 1},
    {"field": "ouverture_nom", "headerName": "Variante", "flex": 3},
    {"field": "gaffes_joueur", "headerName": "Gaffes", "flex": 0.8},
    {"field": "partie_id", "headerName": "Partie", "flex": 0.9},
]
COLONNES_COUPS = [
    {"field": "numero_coup", "headerName": "N°", "flex": 0.6},
    {"field": "coup", "headerName": "Mon coup", "flex": 1},
    {"field": "meilleur_coup", "headerName": "Meilleur coup", "flex": 1},
    {
        "field": "jugement",
        "headerName": "Jugement",
        "flex": 1.2,
        "cellClassRules": {"verdict-travailler": "params.value === 'gaffe'"},
    },
    {"field": "perte", "headerName": "Perte (cp)", "flex": 1},
    {"field": "phase", "headerName": "Phase", "flex": 1.2},
    {"field": "nature", "headerName": "Nature", "flex": 2.4},
    {"field": "temps", "headerName": "Temps restant (s)", "flex": 1.2},
]
COLONNES_REJOUER = [
    {"field": "date", "headerName": "Date", "flex": 1.2},
    {"field": "gaffes_joueur", "headerName": "Gaffes", "flex": 0.8},
    {"field": "resultat", "headerName": "Résultat", "flex": 1},
    {"field": "couleur", "headerName": "Couleur", "flex": 1},
    {"field": "ouverture_famille", "headerName": "Ouverture", "flex": 2},
    {"field": "partie_id", "headerName": "Partie", "flex": 1},
]


def ouvertures_tab():
    return onglet(
        "Ouvertures",
        "ouvertures",
        html.H2("Mes ouvertures, du plus faible au plus solide"),
        deux(
            carte(
                "Cliquez une barre pour voir ses parties",
                graphique("familles-graph", 520),
                zone("familles-lecture"),
                aide("Comment lire ce graphique ?", LEGENDE_VERDICT),
            ),
            carte(
                "Le tableau, avec les chiffres exacts",
                grille("familles-grid", COLONNES_FAMILLES, lignes=12),
                note(
                    "Chaque colonne se trie et se filtre. Le verdict tient compte de la marge d'erreur."
                ),
            ),
        ),
        html.H2(id="famille-titre"),
        deux(
            carte(
                "Par couleur",
                graphique("famille-couleur-graph", 300),
                zone("famille-couleur-lecture"),
            ),
            carte(
                "Variantes les plus jouées (5 parties minimum)",
                grille("variantes-grid", COLONNES_VARIANTES, lignes=6),
                zone("variantes-note"),
            ),
        ),
        carte(
            "Ses parties : cliquez une partie pour voir ses mauvais coups",
            grille("parties-grid", COLONNES_PARTIES, selection=True, id_ligne="partie_id"),
            classe="pleine",
        ),
        carte(
            "Les coups jugés mauvais de la partie choisie",
            html.P(id="coups-titre", className="note"),
            grille("coups-grid", COLONNES_COUPS),
            classe="pleine",
        ),
    )


def gaffes_tab():
    return onglet(
        "Gaffes",
        "gaffes",
        html.H2("Quelle sorte de fautes, et quand ?"),
        html.Div(
            [
                html.Span("Gravité", className="etiquette"),
                dcc.RadioItems(
                    id="gravite-radio",
                    options=list(GRAVITES),
                    value="gaffes",
                    inline=True,
                    className="pastilles",
                ),
                html.Span(
                    "Une gaffe est la faute la plus lourde ; les erreurs et les imprécisions "
                    "pèsent moins.",
                    className="note",
                ),
            ],
            className="reglage",
        ),
        deux(
            carte(
                "Nature de mes fautes, par phase", graphique("types-graph"), zone("types-lecture")
            ),
            carte(
                "Gaffes pour 100 de mes coups, par phase",
                graphique("taux-graph"),
                zone("taux-lecture"),
            ),
        ),
        deux(
            carte(
                "Temps qu'il me restait quand je faisais une faute",
                graphique("temps-graph", 340),
                zone("temps-lecture"),
            ),
            carte(
                "Parties à rejouer en premier (celles où j'ai le plus de gaffes)",
                grille("rejouer-grid", COLONNES_REJOUER),
                note(
                    "Rejouer une partie pleine de gaffes apprend plus que rejouer une défaite au "
                    "hasard : on y retrouve les mêmes schémas."
                ),
            ),
        ),
    )


def progression_tab():
    return onglet(
        "Progression",
        "progression",
        html.H2("Est-ce que je progresse vraiment ?"),
        zone("elo-lectures"),
        deux(
            carte(
                "Mon Elo, mois par mois",
                graphique("elo-graph"),
                note(
                    "Un Elo n'est pas comparable d'une plateforme ou d'une cadence à l'autre : "
                    "chaque courbe se lit seule."
                ),
            ),
            carte(
                "Mon score sur les 100 dernières parties",
                graphique("glissant-graph"),
                note(
                    "Un score autour de 50 % alors que l'Elo monte est normal : le classement vous "
                    "donne des adversaires de plus en plus forts. Pour mesurer la progression, "
                    "regardez l'Elo, pas le score."
                ),
            ),
        ),
    )


def contexte_tab():
    return onglet(
        "Contexte",
        "contexte",
        html.H2("Dans quelles conditions je joue bien ou mal"),
        deux(
            carte(
                "Selon l'heure de début (UTC, 30 parties minimum)",
                graphique("heures-graph"),
                zone("heures-lecture"),
            ),
            carte(
                "Comment finissent mes victoires et mes défaites",
                graphique("fins-graph"),
                zone("fins-lecture"),
            ),
        ),
        deux(
            carte("Blancs contre noirs", graphique("couleur-graph", 340), zone("couleur-lecture")),
            carte(
                "Selon la force de l'adversaire (30 parties minimum)",
                graphique("ecart-graph", 340),
                zone("ecart-lecture"),
            ),
        ),
    )


def serve_layout():
    """La mise en page, refaite à chaque chargement de la page (les cadences viennent des données)."""
    df, _, _ = charger()
    cadences = sorted(df["cadence"].unique())
    debut, fin = df["date_jeu"].min().date(), df["date_jeu"].max().date()
    defaut = {
        "cadences": ["blitz"] if "blitz" in cadences else cadences,
        "debut": debut.isoformat(),
        "fin": fin.isoformat(),
        "couleur": "toutes",
        "min_n": 30,
    }
    return html.Div(
        className="appli",
        children=[
            dcc.Store(id="filtres-store", data=defaut),
            html.Aside(
                className="barre-laterale",
                children=[
                    html.H2("Filtres"),
                    html.Label("Cadence", className="etiquette"),
                    dcc.Checklist(
                        id="cadence-checklist",
                        options=cadences,
                        value=defaut["cadences"],
                        className="pastilles",
                    ),
                    html.Small("« aucune » : parties par correspondance ou sans pendule."),
                    html.Label("Période", className="etiquette"),
                    dcc.DatePickerRange(
                        id="periode-datepicker",
                        start_date=defaut["debut"],
                        end_date=defaut["fin"],
                        min_date_allowed=debut,
                        max_date_allowed=fin,
                        display_format="DD/MM/YYYY",
                        minimum_nights=0,
                        clearable=False,
                    ),
                    html.Label("Couleur", className="etiquette"),
                    dcc.RadioItems(
                        id="couleur-radio",
                        options=list(COULEURS),
                        value="toutes",
                        inline=True,
                        className="pastilles",
                    ),
                    html.Label("Parties minimum par ouverture", className="etiquette"),
                    dcc.Slider(
                        id="min-n-slider",
                        min=5,
                        max=60,
                        step=5,
                        value=defaut["min_n"],
                        marks={5: "5", 30: "30", 60: "60"},
                    ),
                    html.Small("Sous ce nombre, un score ne dit rien : la famille est écartée."),
                ],
            ),
            html.Main(
                className="contenu",
                children=[
                    html.Header(
                        [
                            html.H1("Qu'est-ce que je devrais travailler cette semaine ?"),
                            html.P(
                                "Mes parties d'échecs depuis 2023 : où je perds des points, et dans "
                                "quelle phase."
                            ),
                        ],
                        className="hero",
                    ),
                    html.Div(id="filtres-badges", className="badges"),
                    html.Div(id="kpis", className="grille-3"),
                    zone("alerte-echantillon"),
                    dcc.Tabs(
                        id="onglets",
                        value="synthese",
                        className="onglets",
                        children=[
                            synthese(),
                            ouvertures_tab(),
                            gaffes_tab(),
                            progression_tab(),
                            contexte_tab(),
                        ],
                    ),
                ],
            ),
        ],
    )


app.layout = serve_layout


# ------------------------------------------------------------------ les filtres


@callback(
    Output("filtres-store", "data"),
    Input("cadence-checklist", "value"),
    Input("periode-datepicker", "start_date"),
    Input("periode-datepicker", "end_date"),
    Input("couleur-radio", "value"),
    Input("min-n-slider", "value"),
    prevent_initial_call=True,  # le Store a déjà ses valeurs par défaut
)
def maj_filtres(cadences, debut, fin, couleur, min_n):
    """Les widgets de la barre latérale -> un seul dictionnaire que lisent tous les callbacks."""
    df, _, _ = charger()
    return {
        "cadences": sorted(cadences or []),
        "debut": (debut or df["date_jeu"].min().date().isoformat())[:10],
        "fin": (fin or df["date_jeu"].max().date().isoformat())[:10],
        "couleur": couleur,
        "min_n": min_n,
    }


# ------------------------------------------------------------------ l'en-tête


@callback(
    Output("filtres-badges", "children"),
    Output("kpis", "children"),
    Output("alerte-echantillon", "children"),
    FILTRES,
)
def maj_en_tete(filtres):
    """Rappel des filtres actifs, trois chiffres comparés à toutes mes parties, alerte d'échantillon."""
    s = selection(filtres)
    debut, fin = (pd.Timestamp(filtres[k]).strftime("%d/%m/%Y") for k in ("debut", "fin"))
    badges = [
        html.Span(", ".join(filtres["cadences"]) or "aucune cadence", className="badge badge-vert"),
        html.Span(f"{debut} → {fin}", className="badge badge-bleu"),
        html.Span(f"Couleur : {filtres['couleur']}", className="badge badge-gris"),
    ]
    if s.vue.empty:
        return badges, None, lecture(MSG_VIDE + " Élargissez la cadence ou la période.", "warning")

    vue, tout = s.vue, s.tout
    ecart_score = vue["score"].mean() - tout["score"].mean()
    ecart_gaffes = vue["gaffes_joueur"].mean() - tout["gaffes_joueur"].mean()
    cartes = [
        kpi(
            "Parties",
            f"{len(vue):,}".replace(",", " "),
            f"{len(vue) / len(tout):.0%} de toutes mes parties",
            "neutre",  # un volume n'est ni bon ni mauvais
            courbe(vue, "partie_id", "count", "bar"),
            "La mini-courbe montre le nombre de parties par mois.",
        ),
        kpi(
            "Score moyen",
            f"{vue['score'].mean():.0%}",
            f"{ecart_score:+.0%} vs toutes",
            ton_delta(ecart_score, 0.005),
            courbe(vue, "score", "mean", "area"),
            "1 pour une victoire, 0,5 pour une nulle, 0 pour une défaite. 50 % = autant de parties "
            "gagnées que perdues. Les premiers mois ont peu de parties : leur courbe est bruitée.",
        ),
        kpi(
            "Gaffes par partie",
            f"{vue['gaffes_joueur'].mean():.2f}",
            f"{ecart_gaffes:+.2f} vs toutes",
            ton_delta(ecart_gaffes, 0.005, hausse_est_bonne=False),
            courbe(vue, "gaffes_joueur", "mean", "line"),
            "Une gaffe est un coup qui fait perdre beaucoup de chances de gain, selon le moteur.",
        ),
    ]
    alerte = lecture_echantillon(len(vue))
    return badges, cartes, lecture(alerte[1], alerte[0]) if alerte else None


# ------------------------------------------------------------------ Synthèse


@callback(Output("actions-cartes", "children"), Output("score-lecture", "children"), FILTRES)
def maj_actions(filtres):
    """Trois actions pour la semaine, chacune avec sa raison chiffrée."""
    s = selection(filtres)
    if s.vue.empty:
        return None, None
    cartes = [
        html.Div(
            [
                html.Div(ICONES.get(a["icone"].strip(":").split("/")[-1], "💡"), className="icone"),
                html.Div([html.Strong(a["titre"]), html.P(a["texte"])]),
            ],
            className="action",
        )
        for a in recommandations(s.vue, s.coups, s.phases, s.f, s.min_n)
    ]
    return cartes, lecture(lecture_score(s.vue["score"].mean(), s.tout["score"].mean(), len(s.vue)))


@callback(
    Output("synthese-familles-graph", "figure"),
    Output("synthese-familles-lecture", "children"),
    FILTRES,
)
def maj_synthese_familles(filtres):
    s = selection(filtres)
    if s.vue.empty:
        return vide(MSG_VIDE), None
    if s.f.empty:
        return vide(f"Aucune ouverture n'a {s.min_n} parties ici : baissez le minimum."), None
    return figure_familles(s.f), lecture(lecture_ouvertures(s.f, s.min_n))


@callback(
    Output("synthese-phases-graph", "figure"),
    Output("synthese-phases-lecture", "children"),
    FILTRES,
)
def maj_synthese_phases(filtres):
    s = selection(filtres)
    g = nombre_gaffes(s) if not s.vue.empty else None
    if g is None or g["gaffes"].sum() == 0:
        return vide("Aucune gaffe dans cette sélection."), None
    fig = figure_phases(g, "gaffes", "nombre de gaffes", "%{y}", 520)
    return fig, lecture(lecture_phases(gaffes_par_phase(s.phases)))


# ------------------------------------------------------------------ Ouvertures (le drill-down)


def famille_choisie(f: pd.DataFrame, clic) -> tuple[str | None, bool]:
    """La famille dont on affiche le détail : celle de la barre cliquée, sinon la plus faible."""
    try:
        nom = clic["points"][0]["y"]
    except (TypeError, KeyError, IndexError):
        nom = None
    if nom in set(f["famille"]):
        return nom, True
    return (f["famille"].iloc[0], False) if not f.empty else (None, False)


CLIC = Input("familles-graph", "clickData")


@callback(Output("familles-graph", "figure"), Output("familles-lecture", "children"), FILTRES, CLIC)
def maj_familles(filtres, clic):
    s = selection(filtres)
    if s.vue.empty:
        return vide(MSG_VIDE), None
    if s.f.empty:
        return vide(f"Aucune ouverture n'a {s.min_n} parties ici : baissez le minimum."), None
    nom, cliquee = famille_choisie(s.f, clic)
    fig = figure_familles(s.f, nom if cliquee else None)
    return fig, lecture(lecture_ouvertures(s.f, s.min_n))


@callback(Output("familles-grid", "rowData"), FILTRES)
def maj_tableau_familles(filtres):
    return selection(filtres).f.to_dict("records")


@callback(
    Output("famille-titre", "children"),
    Output("famille-couleur-graph", "figure"),
    Output("famille-couleur-lecture", "children"),
    Output("variantes-grid", "rowData"),
    Output("variantes-note", "children"),
    FILTRES,
    CLIC,
)
def maj_detail_famille(filtres, clic):
    """Le zoom sur la famille choisie : par couleur, et ses variantes."""
    s = selection(filtres)
    nom, cliquee = famille_choisie(s.f, clic)
    if nom is None:
        return "Détail d'une famille", vide("Aucune famille à détailler."), None, [], None
    titre = f"Détail : {nom}" + ("" if cliquee else " (la plus faible : cliquez une autre barre)")
    fam = s.vue[s.vue["ouverture_famille"] == nom]

    par_couleur = score_par(fam, "couleur", 1)
    if len(par_couleur) == 2:
        d = par_couleur.assign(couleur=par_couleur["couleur"].map(FR_COULEUR))
        fig, texte = barres_verdict(d, "couleur", 300), lecture(lecture_couleur(par_couleur))
    else:
        seule = FR_COULEUR[par_couleur["couleur"].iloc[0]]
        fig = vide(f"{nom} n'apparaît que dans vos parties avec les {seule} : pas de comparaison.")
        texte = None

    variantes = (
        fam.dropna(subset=["ouverture_nom"])
        .groupby("ouverture_nom")
        .agg(n=("score", "size"), score=("score", "mean"), gaffes=("gaffes_joueur", "mean"))
        .query("n >= 5")
        .sort_values("n", ascending=False)
        .head(10)
        .reset_index()
    )
    remarque = None
    if variantes.empty:
        remarque = note("Aucune variante n'a 5 parties : les parties sont trop dispersées.")
    else:
        faible = variantes.sort_values("score").iloc[0]
        if faible["score"] < 0.45:
            remarque = note(
                f"La variante la plus faible est « {faible['ouverture_nom']} » "
                f"({faible['score']:.0%} sur {faible['n']:.0f} parties) : commencez par elle."
            )
    return titre, fig, texte, variantes.to_dict("records"), remarque


@callback(Output("parties-grid", "rowData"), FILTRES, CLIC)
def maj_parties(filtres, clic):
    """Un clic sur une barre -> les parties de cette famille (les plus récentes d'abord)."""
    s = selection(filtres)
    nom, _ = famille_choisie(s.f, clic)
    if nom is None:
        return []
    fam = s.vue[s.vue["ouverture_famille"] == nom].sort_values("date_jeu", ascending=False)
    return fam.assign(
        date=fam["date_jeu"].dt.strftime("%Y-%m-%d"),
        couleur=fam["couleur"].map(FR_COULEUR),
        ouverture_nom=fam["ouverture_nom"].fillna("—"),
    )[["date", "resultat", "couleur", "ouverture_nom", "gaffes_joueur", "partie_id"]].to_dict(
        "records"
    )


@callback(
    Output("coups-grid", "rowData"),
    Output("coups-titre", "children"),
    Input("parties-grid", "selectedRows"),
)
def maj_coups(lignes):
    """Un clic sur une partie -> les coups que le moteur juge mauvais, dans l'ordre de la partie."""
    if not lignes:
        return [], "Cliquez une partie dans le tableau ci-dessus."
    pid = lignes[0]["partie_id"]
    _, coups, _ = charger()
    c = coups[coups["partie_id"] == pid].sort_values("ply")
    gravite = {"inaccuracy": "imprécision", "mistake": "erreur", "blunder": "gaffe"}
    rangees = c.assign(
        jugement=c["jugement"].map(gravite),
        perte=c["variation_eval_cp"],
        phase=c["phase"].map(PHASES),
        nature=c["sous_type"].map(TYPES).fillna(POSITIONNELLE),
        temps=c["temps_restant_s"],
        meilleur_coup=c["meilleur_coup"].fillna("—"),
    )[["numero_coup", "coup", "meilleur_coup", "jugement", "perte", "phase", "nature", "temps"]]
    n_gaffes = int((c["jugement"] == "blunder").sum())
    pluriel = "s" if n_gaffes > 1 else ""
    titre = f"Partie {pid} : {len(c)} coups jugés mauvais, dont {n_gaffes} gaffe{pluriel}."
    return rangees.to_dict("records"), titre


# ------------------------------------------------------------------ Gaffes

GRAVITE = Input("gravite-radio", "value")


@callback(Output("types-graph", "figure"), Output("types-lecture", "children"), FILTRES, GRAVITE)
def maj_types(filtres, gravite):
    s = selection(filtres)
    t = types_gaffes(s.coups, gravite)
    if s.vue.empty or t.empty:
        return vide("Aucune faute de ce type."), None
    fig = px.bar(
        t,
        x="phase",
        y="part",
        color="type",
        text="part",
        category_orders={"phase": ORDRE_PHASES},
        color_discrete_map={
            "Tactique laissée à l'adversaire": ROUGE,
            "Tactique ratée": ORANGE,
            "Faute de position": NEUTRE,
        },
        hover_data={"n": True, "part": ":.0%"},
        labels={"phase": "", "part": "part des fautes", "n": "fautes", "type": ""},
    )
    fig.update_traces(texttemplate="%{y:.0%}", textposition="inside")
    fig.update_yaxes(tickformat=".0%")
    return style(fig, 380, legende=True), lecture(lecture_types(t, s.coups, gravite))


@callback(Output("taux-graph", "figure"), Output("taux-lecture", "children"), FILTRES)
def maj_taux(filtres):
    s = selection(filtres)
    taux = gaffes_par_phase(s.phases)
    if s.vue.empty or taux["gaffes"].sum() == 0:
        return vide("Aucune gaffe."), None
    fig = figure_phases(taux, "taux", "gaffes pour 100 coups", "%{y:.1f}", 380)
    return fig, lecture(lecture_phases(taux))


@callback(Output("temps-graph", "figure"), Output("temps-lecture", "children"), FILTRES, GRAVITE)
def maj_temps(filtres, gravite):
    s = selection(filtres)
    tr = temps_restant(s.coups, gravite)
    if s.vue.empty or tr.empty:
        return vide("Pas de données de pendule."), None
    fig = px.bar(
        tr,
        x="temps",
        y="part",
        text="part",
        labels={"temps": "temps restant après le coup", "part": "part des fautes", "n": "fautes"},
        hover_data={"n": True},
    )
    fig.update_traces(
        marker_color=[ROUGE if x == "moins de 10 s" else NEUTRE for x in tr["temps"]],
        texttemplate="%{y:.0%}",
        textposition="outside",
        cliponaxis=False,
    )
    fig.update_yaxes(tickformat=".0%")
    return style(fig, 340), lecture(lecture_temps(tr, gravite))


@callback(Output("rejouer-grid", "rowData"), FILTRES)
def maj_rejouer(filtres):
    """Les dix parties où j'ai le plus de gaffes."""
    s = selection(filtres)
    top = s.vue.sort_values(["gaffes_joueur", "date_jeu"], ascending=[False, False]).head(10)
    return top.assign(
        date=top["date_jeu"].dt.strftime("%Y-%m-%d"), couleur=top["couleur"].map(FR_COULEUR)
    )[["date", "gaffes_joueur", "resultat", "couleur", "ouverture_famille", "partie_id"]].to_dict(
        "records"
    )


# ------------------------------------------------------------------ Progression


@callback(Output("elo-graph", "figure"), Output("elo-lectures", "children"), FILTRES)
def maj_elo(filtres):
    s = selection(filtres)
    series = elo_series(s.vue) if not s.vue.empty else {}
    if not series:
        return (
            vide("Pas assez de parties avec un Elo ici (30 par plateforme et cadence)."),
            lecture("Pas assez de parties avec un Elo pour juger de la progression."),
        )
    lectures = []
    for nom, g in series.items():
        tendance = tendance_elo(g)
        if tendance:
            lectures.append(lecture(lecture_elo(nom, tendance, s.vue["date_jeu"].max())))
    fig = px.line(
        elo_mensuel(series),
        x="date_jeu",
        y="elo",
        color="série",
        markers=True,
        labels={"date_jeu": "", "elo": "Elo moyen du mois", "série": ""},
    )
    fig.update_traces(hovertemplate="%{x|%b %Y} : %{y:.0f}")
    return style(fig, 380, legende=True), lectures


@callback(Output("glissant-graph", "figure"), FILTRES)
def maj_glissant(filtres):
    s = selection(filtres)
    sg = score_glissant(s.vue) if not s.vue.empty else None
    if sg is None or sg.empty:
        return vide("Pas assez de parties pour une moyenne glissante.")
    fig = px.line(sg, x="date_jeu", y="glissant", labels={"date_jeu": "", "glissant": "score"})
    fig.update_traces(line_color=VERT_CLAIR)
    fig.update_yaxes(range=[0.3, 0.7], tickformat=".0%")
    fig.add_hline(y=0.5, line_dash="dot", line_color=TRAIT)
    return style(fig, 380)


# ------------------------------------------------------------------ Contexte


@callback(Output("heures-graph", "figure"), Output("heures-lecture", "children"), FILTRES)
def maj_heures(filtres):
    s = selection(filtres)
    h = score_par(s.vue, "heure", 30) if not s.vue.empty else None
    if h is None or h.empty:
        return vide("Aucune heure n'a 30 parties."), None
    fig = barres_verdict(h, "heure", 380, heure="heure (UTC)")
    fig.update_xaxes(dtick=1)
    return fig, lecture(lecture_heures(h))


@callback(Output("fins-graph", "figure"), Output("fins-lecture", "children"), FILTRES)
def maj_fins(filtres):
    s = selection(filtres)
    fins = fins_de_partie(s.vue) if not s.vue.empty else None
    if fins is None or fins.empty:
        return vide("Aucune victoire ou défaite."), None
    fig = px.bar(
        fins,
        x="fin",
        y="part",
        color="resultat",
        barmode="group",
        text="part",
        color_discrete_map={"victoire": VERT_CLAIR, "défaite": ROUGE},
        category_orders={"fin": ["abandon", "mat", "temps", "autre"]},
        hover_data={"n": True, "part": ":.0%"},
        labels={"fin": "", "part": "part des parties", "n": "parties", "resultat": ""},
    )
    fig.update_traces(texttemplate="%{y:.0%}", textposition="outside", cliponaxis=False)
    fig.update_yaxes(tickformat=".0%")
    return style(fig, 380, legende=True), lecture(lecture_fins(fins))


@callback(Output("couleur-graph", "figure"), Output("couleur-lecture", "children"), FILTRES)
def maj_couleur(filtres):
    s = selection(filtres)
    co = score_par(s.vue, "couleur", 1) if not s.vue.empty else None
    if co is None or len(co) < 2:
        return vide("Une seule couleur dans la sélection."), None
    fig = barres_verdict(co.assign(couleur=co["couleur"].map(FR_COULEUR)), "couleur", 340)
    return fig, lecture(lecture_couleur(co))


@callback(Output("ecart-graph", "figure"), Output("ecart-lecture", "children"), FILTRES)
def maj_ecart(filtres):
    s = selection(filtres)
    ec = par_ecart(s.vue) if not s.vue.empty else None
    if ec is None or ec.empty:
        return vide("Pas assez de parties pour comparer."), None
    return barres_verdict(ec, "tranche", 340, tranche="adversaire"), lecture(lecture_ecart(ec))


if __name__ == "__main__":
    # debug=True : bouton « Callbacks » en bas à droite, le graphe des callbacks
    app.run(debug=False, port=8050)
