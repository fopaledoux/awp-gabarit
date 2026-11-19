"""Exo 2 : l'appli Streamlit du fil rouge. Environ 1 h.

Question : l'appli que ce joueur ouvrirait chaque semaine pour savoir quoi travailler.
À produire : une barre latérale (cadence, période, couleur), trois st.metric, deux figures
en colonnes. Le titre de la page dit la question, pas « Dashboard ».
Colonnes utilisées : cadence (dont « aucune »), date_jeu (datetime), couleur (white ou black),
score, gaffes_joueur. Si votre load_data() ne les donne pas ainsi, corrigez-la.
Lancer : uv run streamlit run fil_rouge/app_streamlit.py --server.address 127.0.0.1

Les calculs et les textes d'aide à la décision sont dans analyses.py ; ici, seulement l'affichage.
"""

from pathlib import Path

import plotly.express as px
import streamlit as st
from analyses import (
    GRAVITES,
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

from data import load_coups, load_data

st.set_page_config(page_title="Mes parties d'échecs", page_icon="♟️", layout="wide")
print("relance")  # regardez le terminal à chaque clic

# Couleurs : le thème (.streamlit/config.toml) donne la palette, on nomme ici ses rôles.
ROUGE, ORANGE, NEUTRE, VERT = "#C8553D", "#E0A526", "#8FA3B5", "#2E8B67"
COULEURS_VERDICT = {
    "À travailler": ROUGE,
    "À surveiller": ORANGE,
    "Dans la moyenne": NEUTRE,
    "Solide": VERT,
}
ORDRE_VERDICT = list(COULEURS_VERDICT)
LEGENDE_VERDICT = (
    "**Rouge** : sous 50 % de façon nette. **Orange** : sous 50 %, mais cela peut être du hasard. "
    "**Gris** : dans la moyenne. **Vert** : au-dessus de 50 % de façon nette. "
    "Le trait gris au bout de chaque barre est la marge d'erreur à 95 % : s'il touche la ligne "
    "des 50 %, on ne peut pas conclure. Le nom d'une ouverture décrit la partie, pas votre choix : "
    "avec les noirs, c'est souvent l'adversaire qui l'a imposée."
)
COULEURS = {"toutes": None, "blancs": "white", "noirs": "black"}
FR_COULEUR = {"white": "blancs", "black": "noirs"}


# 1. Charger une seule fois : Streamlit relance tout le script à chaque clic.
@st.cache_data
def charger():
    return load_data(), load_coups(), load_phases()


df, coups, phases = charger()
st.html(Path(__file__).parent / "style.css")  # un fichier .css est enveloppé dans <style>


def style(fig, hauteur=420, legende=False):
    """Même habillage pour toutes les figures : fond transparent (suit le thème clair ou sombre),
    police sobre, grille discrète. `legende` affiche la légende des couleurs au-dessus."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legende,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title": None},
        height=hauteur,
        margin={"l": 0, "r": 16, "t": 80 if legende else 24, "b": 0},
        font={"size": 13, "color": "#8A9A92"},
        bargap=0.28,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(138,154,146,0.25)", zeroline=False)
    return fig


def mensuel(vue, colonne, agg="mean"):
    """Une valeur par mois, pour la mini-courbe d'un st.metric (None s'il y a moins de 3 mois)."""
    s = vue.resample("MS", on="date_jeu")[colonne].agg(agg).dropna()
    return s.tolist() if len(s) >= 3 else None


def lecture(texte, niveau="info"):
    """Le texte qui dit quoi conclure du graphique, dans un encadré natif."""
    boite = {"info": st.info, "warning": st.warning}[niveau]
    boite(texte, icon=":material/lightbulb:" if niveau == "info" else None)


def barres_verdict(d, x, hauteur=380, **libelles):
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
    fig.update_traces(error_y_color="#8A9A92", error_y_thickness=1.5)
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.add_hline(y=0.5, line_dash="dot", line_color="#8A9A92")
    return style(fig, hauteur, legende=True)


# 2. La barre latérale : widgets natifs avec key= (l'exo 6 les pilote par leur nom).
with st.sidebar:
    st.header("Filtres")
    cadences = sorted(df["cadence"].unique())
    choix_cadences = st.pills(
        "Cadence",
        cadences,
        selection_mode="multi",
        default=["blitz"] if "blitz" in cadences else cadences,
        key="cadence",
        help="« aucune » : parties par correspondance ou sans pendule.",
    )
    debut, fin = df["date_jeu"].min().date(), df["date_jeu"].max().date()
    periode = st.date_input(
        "Période", value=(debut, fin), min_value=debut, max_value=fin, key="periode"
    )
    choix_couleur = st.pills(
        "Couleur", list(COULEURS), default="toutes", required=True, key="couleur"
    )
    min_n = st.slider(
        "Parties minimum par ouverture",
        5,
        60,
        30,
        5,
        key="min_n",
        help="Sous ce nombre, un score ne dit rien : la famille est écartée.",
    )

# 3. Filtrer, et s'arrêter proprement si la sélection est vide ou incomplète.
with st.container(key="hero"):
    st.title("Qu'est-ce que je devrais travailler cette semaine ?")
    st.caption("Mes parties d'échecs depuis 2023 : où je perds des points, et dans quelle phase.")

if len(periode) != 2:
    st.info("Choisissez la seconde date de la période.")
    st.stop()

jours = df["date_jeu"].dt.date
vue = df[df["cadence"].isin(choix_cadences) & jours.between(periode[0], periode[1])]
if COULEURS[choix_couleur]:
    vue = vue[vue["couleur"] == COULEURS[choix_couleur]]

if vue.empty:
    st.warning("Aucune partie ne correspond à ces filtres : élargissez la cadence ou la période.")
    st.stop()

# Rappel des filtres actifs, en badges.
with st.container(horizontal=True, gap="xsmall"):
    st.badge(", ".join(choix_cadences), icon=":material/timer:", color="green")
    st.badge(
        f"{periode[0]:%d/%m/%Y} → {periode[1]:%d/%m/%Y}", icon=":material/event:", color="blue"
    )
    st.badge(f"Couleur : {choix_couleur}", icon=":material/contrast:", color="gray")

# 4. Trois chiffres, comparés à l'ensemble de mes parties, avec leur évolution mois par mois.
c1, c2, c3 = st.columns(3)
with c1, st.container(key="kpi-parties", border=True):
    st.metric(
        "Parties",
        f"{len(vue):,}".replace(",", " "),
        delta=f"{len(vue) / len(df):.0%} de toutes mes parties",
        delta_color="off",  # un volume n'est ni bon ni mauvais : delta gris
        delta_arrow="off",
        icon=":material/sports_esports:",
        chart_data=mensuel(vue, "partie_id", "count"),
        chart_type="bar",
        help="La mini-courbe montre le nombre de parties par mois.",
    )
with c2, st.container(key="kpi-score", border=True):
    st.metric(
        "Score moyen",
        f"{vue['score'].mean():.0%}",
        delta=f"{vue['score'].mean() - df['score'].mean():+.0%} vs toutes",
        icon=":material/emoji_events:",
        chart_data=mensuel(vue, "score"),
        chart_type="area",
        help="1 pour une victoire, 0,5 pour une nulle, 0 pour une défaite. 50 % = autant de "
        "parties gagnées que perdues. Les premiers mois ont peu de parties : leur courbe est bruitée.",
    )
with c3, st.container(key="kpi-gaffes", border=True):
    st.metric(
        "Gaffes par partie",
        f"{vue['gaffes_joueur'].mean():.2f}",
        delta=f"{vue['gaffes_joueur'].mean() - df['gaffes_joueur'].mean():+.2f} vs toutes",
        delta_color="inverse",  # moins de gaffes, c'est mieux : le vert pour une baisse
        icon=":material/warning:",
        chart_data=mensuel(vue, "gaffes_joueur"),
        chart_type="line",
        help="Une gaffe est un coup qui fait perdre beaucoup de chances de gain, selon le moteur.",
    )

# Les données communes aux onglets, calculées sur les parties filtrées.
ids = vue["partie_id"]
coups_vue = coups[coups["partie_id"].isin(ids)]
phases_vue = phases[phases["partie_id"].isin(ids)]
f = ouvertures(vue, min_n)
taux_phases = gaffes_par_phase(phases_vue)

alerte = lecture_echantillon(len(vue))
if alerte:
    lecture(alerte[1], alerte[0])

tab_synthese, tab_ouv, tab_gaffes, tab_prog, tab_ctx = st.tabs(
    ["Synthèse", "Ouvertures", "Gaffes", "Progression", "Contexte"]
)

# ------------------------------------------------------------------ Synthèse
with tab_synthese:
    st.subheader("Que faire cette semaine ?")
    actions = recommandations(vue, coups_vue, phases_vue, f, min_n)
    if actions:
        for col, (i, a) in zip(st.columns(len(actions)), enumerate(actions), strict=True):
            with col, st.container(key=f"action-{i}", border=True):
                st.markdown(f"**{a['icone']} {a['titre']}**")
                st.write(a["texte"])
    lecture(lecture_score(vue["score"].mean(), df["score"].mean(), len(vue)))

    # Les figures 1 et 2 de l'exo 1, chacune dans sa carte.
    gauche, droite = st.columns(2)
    with gauche, st.container(key="carte-familles", border=True):
        if f.empty:
            st.info(f"Aucune ouverture n'a {min_n} parties ici : baissez le minimum.")
        else:
            st.subheader("Score par ouverture : les pires en haut")
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
                textfont={"color": "#10201A", "size": 12},
                error_x_color="#8A9A92",
                error_x_thickness=1.5,
            )
            fig.update_yaxes(autorange="reversed")  # la pire en haut
            fig.update_xaxes(range=[0, 1], tickformat=".0%")
            fig.add_vline(x=0.5, line_dash="dot", line_color="#8A9A92")
            st.plotly_chart(
                style(fig, 520, legende=True),
                width="stretch",
                alt="Score moyen par famille d'ouvertures avec marge d'erreur, des pires aux meilleures.",
            )
            lecture(lecture_ouvertures(f, min_n))
            with st.expander("Comment lire ce graphique ?"):
                st.markdown(LEGENDE_VERDICT)

    with droite, st.container(key="carte-phases", border=True):
        if taux_phases["gaffes"].sum() == 0:
            st.info("Aucune gaffe dans cette sélection.")
        else:
            st.subheader("Où tombent mes gaffes")
            nombre = (
                coups_vue[coups_vue["jugement"] == "blunder"]["phase"]
                .value_counts()
                .reindex(["opening", "middlegame", "endgame"], fill_value=0)
            )
            g = taux_phases.assign(gaffes=nombre.to_numpy())
            pire = g.loc[g["gaffes"].idxmax()]
            fig = px.bar(
                g,
                x="phase",
                y="gaffes",
                text="gaffes",
                hover_data={"taux": ":.1f"},
                labels={"phase": "", "gaffes": "nombre de gaffes", "taux": "pour 100 de mes coups"},
            )
            fig.update_traces(
                marker_color=[ROUGE if p == pire["phase"] else NEUTRE for p in g["phase"]],
                textposition="outside",
                cliponaxis=False,
            )
            st.plotly_chart(
                style(fig, 520),
                width="stretch",
                alt="Nombre de gaffes par phase de la partie : ouverture, milieu de jeu, finale.",
            )
            lecture(lecture_phases(taux_phases))

# ------------------------------------------------------------------ Ouvertures
with tab_ouv:
    st.subheader("Mes ouvertures, du plus faible au plus solide")
    if f.empty:
        st.info(f"Aucune ouverture n'a {min_n} parties ici : baissez le minimum.")
    else:
        lecture(lecture_ouvertures(f, min_n))
        tableau = f.assign(score=f["score"] * 100, marge=f["marge"] * 100)[
            ["famille", "verdict", "n", "score", "marge", "gaffes"]
        ]
        st.dataframe(
            tableau,
            hide_index=True,
            width="stretch",
            column_config={
                "famille": st.column_config.TextColumn("Ouverture"),
                "verdict": st.column_config.TextColumn(
                    "Verdict", help="Voir « Comment lire ce tableau » ci-dessous."
                ),
                "n": st.column_config.NumberColumn("Parties"),
                "score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=100, format="%.0f %%"
                ),
                "marge": st.column_config.NumberColumn(
                    "± marge", format="%.0f pts", help="Marge d'erreur à 95 %."
                ),
                "gaffes": st.column_config.NumberColumn("Gaffes/partie", format="%.2f"),
            },
        )
        with st.expander("Comment lire ce tableau ?"):
            st.markdown(LEGENDE_VERDICT)

        st.subheader("Zoom sur une famille")
        choix = st.selectbox(
            "Famille à détailler",
            list(f["famille"]),
            key="famille",
            help="Triées de la pire à la meilleure.",
        )
        fam = vue[vue["ouverture_famille"] == choix]
        par_couleur = score_par(fam, "couleur", 1)
        a, b = st.columns(2)
        with a, st.container(border=True):
            st.markdown(f"**Par couleur : {choix}**")
            if len(par_couleur) == 2:
                d = par_couleur.assign(couleur=par_couleur["couleur"].map(FR_COULEUR))
                st.plotly_chart(
                    barres_verdict(d, "couleur", 300),
                    width="stretch",
                    alt=f"Score en {choix} avec les blancs et avec les noirs.",
                )
                lecture(lecture_couleur(par_couleur))
            else:
                seule = FR_COULEUR[par_couleur["couleur"].iloc[0]]
                st.write(
                    f"{choix} n'apparaît que dans vos parties avec les **{seule}** : pas de comparaison."
                )
        with b, st.container(border=True):
            st.markdown("**Variantes les plus jouées** (5 parties minimum)")
            variantes = (
                fam.dropna(subset=["ouverture_nom"])
                .groupby("ouverture_nom")
                .agg(n=("score", "size"), score=("score", "mean"), gaffes=("gaffes_joueur", "mean"))
                .query("n >= 5")
                .sort_values("n", ascending=False)
                .head(10)
                .reset_index()
            )
            if variantes.empty:
                st.write("Aucune variante n'a 5 parties : les parties sont trop dispersées.")
            else:
                variantes["score"] *= 100
                st.dataframe(
                    variantes,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "ouverture_nom": st.column_config.TextColumn("Variante"),
                        "n": st.column_config.NumberColumn("Parties"),
                        "score": st.column_config.ProgressColumn(
                            "Score", min_value=0, max_value=100, format="%.0f %%"
                        ),
                        "gaffes": st.column_config.NumberColumn("Gaffes/partie", format="%.2f"),
                    },
                )
                faible = variantes.sort_values("score").iloc[0]
                if faible["score"] < 45:
                    st.caption(
                        f"La variante la plus faible est « {faible['ouverture_nom']} » "
                        f"({faible['score']:.0f} % sur {faible['n']:.0f} parties) : "
                        "commencez par elle."
                    )

# ------------------------------------------------------------------ Gaffes
with tab_gaffes:
    st.subheader("Quelle sorte de fautes, et quand ?")
    gravite = st.pills(
        "Gravité",
        list(GRAVITES),
        default="gaffes",
        required=True,
        key="gravite",
        help="Une gaffe est la faute la plus lourde ; les erreurs et les imprécisions pèsent moins.",
    )
    t = types_gaffes(coups_vue, gravite)
    a, b = st.columns(2)
    with a, st.container(key="carte-types", border=True):
        st.markdown(f"**Nature de mes {gravite}, par phase**")
        if t.empty:
            st.info("Aucune faute de ce type.")
        else:
            fig = px.bar(
                t,
                x="phase",
                y="part",
                color="type",
                text="part",
                category_orders={"phase": ["ouverture", "milieu de jeu", "finale"]},
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
            st.plotly_chart(
                style(fig, 380, legende=True),
                width="stretch",
                alt=f"Répartition des {gravite} : tactiques laissées, ratées, ou fautes de position.",
            )
            lecture(lecture_types(t, coups_vue, gravite))
    with b, st.container(key="carte-taux", border=True):
        st.markdown("**Gaffes pour 100 de mes coups, par phase**")
        if taux_phases["gaffes"].sum() == 0:
            st.info("Aucune gaffe.")
        else:
            pire = taux_phases.loc[taux_phases["taux"].idxmax()]
            fig = px.bar(
                taux_phases,
                x="phase",
                y="taux",
                text="taux",
                labels={"phase": "", "taux": "gaffes pour 100 coups"},
            )
            fig.update_traces(
                marker_color=[
                    ROUGE if p == pire["phase"] else NEUTRE for p in taux_phases["phase"]
                ],
                texttemplate="%{y:.1f}",
                textposition="outside",
                cliponaxis=False,
            )
            st.plotly_chart(
                style(fig, 380),
                width="stretch",
                alt="Gaffes pour 100 coups joués, par phase de la partie.",
            )
            lecture(lecture_phases(taux_phases))

    c, d = st.columns(2)
    with c, st.container(key="carte-temps", border=True):
        st.markdown(f"**Temps qu'il me restait quand je faisais une faute ({gravite})**")
        tr = temps_restant(coups_vue, gravite)
        if tr.empty:
            st.info("Pas de données de pendule.")
        else:
            fig = px.bar(
                tr,
                x="temps",
                y="part",
                text="part",
                labels={
                    "temps": "temps restant après le coup",
                    "part": "part des fautes",
                    "n": "fautes",
                },
                hover_data={"n": True},
            )
            fig.update_traces(
                marker_color=[ROUGE if x == "moins de 10 s" else NEUTRE for x in tr["temps"]],
                texttemplate="%{y:.0%}",
                textposition="outside",
                cliponaxis=False,
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(
                style(fig, 340),
                width="stretch",
                alt="Part des fautes selon le temps restant à la pendule.",
            )
            lecture(lecture_temps(tr, gravite))
    with d, st.container(key="carte-rejouer", border=True):
        st.markdown("**Parties à rejouer en premier** (celles où j'ai le plus de gaffes)")
        top = (
            vue.sort_values(["gaffes_joueur", "date_jeu"], ascending=[False, False])
            .head(10)
            .assign(couleur=lambda x: x["couleur"].map(FR_COULEUR))[
                [
                    "date_jeu",
                    "gaffes_joueur",
                    "resultat",
                    "couleur",
                    "ouverture_famille",
                    "partie_id",
                ]
            ]
        )
        st.dataframe(
            top,
            hide_index=True,
            width="stretch",
            column_config={
                "date_jeu": st.column_config.DatetimeColumn("Date", format="DD/MM/YY"),
                "ouverture_famille": st.column_config.TextColumn("Ouverture"),
                "couleur": st.column_config.TextColumn("Couleur"),
                "resultat": st.column_config.TextColumn("Résultat"),
                "gaffes_joueur": st.column_config.NumberColumn("Gaffes"),
                "partie_id": st.column_config.TextColumn("Partie"),
            },
        )
        st.caption(
            "Rejouer une partie pleine de gaffes apprend plus que rejouer une défaite au hasard : "
            "on y retrouve les mêmes schémas."
        )

# ------------------------------------------------------------------ Progression
with tab_prog:
    st.subheader("Est-ce que je progresse vraiment ?")
    series = elo_series(vue)
    if not series:
        st.info("Pas assez de parties avec un Elo ici (30 par plateforme et cadence).")
    else:
        for nom, g in series.items():
            tendance = tendance_elo(g)
            if tendance:
                lecture(lecture_elo(nom, tendance, vue["date_jeu"].max()))
        a, b = st.columns(2)
        with a, st.container(key="carte-elo", border=True):
            st.markdown("**Mon Elo, mois par mois**")
            fig = px.line(
                elo_mensuel(series),
                x="date_jeu",
                y="elo",
                color="série",
                markers=True,
                labels={"date_jeu": "", "elo": "Elo moyen du mois", "série": ""},
            )
            fig.update_traces(hovertemplate="%{x|%b %Y} : %{y:.0f}")
            st.plotly_chart(
                style(fig, 380, legende=True),
                width="stretch",
                alt="Elo moyen par mois pour chaque plateforme et cadence.",
            )
            st.caption(
                "Un Elo n'est pas comparable d'une plateforme ou d'une cadence à l'autre : "
                "chaque courbe se lit seule."
            )
        with b, st.container(key="carte-glissant", border=True):
            st.markdown("**Mon score sur les 100 dernières parties**")
            sg = score_glissant(vue)
            if sg.empty:
                st.info("Pas assez de parties pour une moyenne glissante.")
            else:
                fig = px.line(
                    sg, x="date_jeu", y="glissant", labels={"date_jeu": "", "glissant": "score"}
                )
                fig.update_traces(line_color=VERT)
                fig.update_yaxes(range=[0.3, 0.7], tickformat=".0%")
                fig.add_hline(y=0.5, line_dash="dot", line_color="#8A9A92")
                st.plotly_chart(
                    style(fig, 380),
                    width="stretch",
                    alt="Score moyen glissant sur 100 parties, autour de 50 %.",
                )
                st.caption(
                    "Un score autour de 50 % alors que l'Elo monte est normal : le classement vous "
                    "donne des adversaires de plus en plus forts. Pour mesurer la progression, "
                    "regardez l'Elo, pas le score."
                )

# ------------------------------------------------------------------ Contexte
with tab_ctx:
    st.subheader("Dans quelles conditions je joue bien ou mal")
    a, b = st.columns(2)
    with a, st.container(key="carte-heures", border=True):
        st.markdown("**Selon l'heure de début (UTC, 30 parties minimum)**")
        h = score_par(vue, "heure", 30)
        if h.empty:
            st.info("Aucune heure n'a 30 parties.")
        else:
            fig = barres_verdict(h, "heure", 380, heure="heure (UTC)")
            fig.update_xaxes(dtick=1)
            st.plotly_chart(fig, width="stretch", alt="Score moyen selon l'heure de début.")
            lecture(lecture_heures(h))
    with b, st.container(key="carte-fins", border=True):
        st.markdown("**Comment finissent mes victoires et mes défaites**")
        fins = fins_de_partie(vue)
        if fins.empty:
            st.info("Aucune victoire ou défaite.")
        else:
            fig = px.bar(
                fins,
                x="fin",
                y="part",
                color="resultat",
                barmode="group",
                text="part",
                color_discrete_map={"victoire": VERT, "défaite": ROUGE},
                category_orders={"fin": ["abandon", "mat", "temps", "autre"]},
                hover_data={"n": True, "part": ":.0%"},
                labels={"fin": "", "part": "part des parties", "n": "parties", "resultat": ""},
            )
            fig.update_traces(texttemplate="%{y:.0%}", textposition="outside", cliponaxis=False)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(
                style(fig, 380, legende=True),
                width="stretch",
                alt="Comment finissent les victoires et les défaites : abandon, mat, temps.",
            )
            lecture(lecture_fins(fins))

    c, d = st.columns(2)
    with c, st.container(key="carte-couleur", border=True):
        st.markdown("**Blancs contre noirs**")
        co = score_par(vue, "couleur", 1)
        if len(co) < 2:
            st.info("Une seule couleur dans la sélection.")
        else:
            fig = barres_verdict(co.assign(couleur=co["couleur"].map(FR_COULEUR)), "couleur", 340)
            st.plotly_chart(fig, width="stretch", alt="Score avec les blancs et avec les noirs.")
            lecture(lecture_couleur(co))
    with d, st.container(key="carte-ecart", border=True):
        st.markdown("**Selon la force de l'adversaire** (30 parties minimum)")
        ec = par_ecart(vue)
        if ec.empty:
            st.info("Pas assez de parties pour comparer.")
        else:
            fig = barres_verdict(ec, "tranche", 340, tranche="adversaire")
            st.plotly_chart(fig, width="stretch", alt="Score selon l'Elo de l'adversaire.")
            lecture(lecture_ecart(ec))
