"""Exo 1 : trois figures. Environ 25 min.

Question : où ce joueur perd-il des points ?
À produire : trois figures Plotly dont le titre dit ce qu'elles montrent.
Lancer : uv run python fil_rouge/01_figures.py
"""

import plotly.express as px

from data import familles, load_coups, load_data

df, coups = load_data(), load_coups()
print(df.shape, coups.shape)

# 1. Le score par famille d'ouvertures, la pire en haut (px.bar, familles(df))
f = familles(df)
fig1 = px.bar(
    f,
    x="score",
    y="famille",
    orientation="h",
    hover_data=["n", "gaffes"],
    title="Hongroise, Anglaise et Philidor : mes ouvertures les moins rentables (≥ 30 parties)",
    labels={"score": "score moyen (1 = victoire)", "famille": ""},
)
# Plotly dessine la première catégorie en bas : on inverse l'axe pour mettre la pire en haut.
fig1.update_yaxes(autorange="reversed")
fig1.add_vline(x=0.5, line_dash="dot")
fig1.show()

# 2. La phase de la partie où tombent ses gaffes : ouverture, milieu, finale (coups)
ordre = ["opening", "middlegame", "endgame"]
noms = {"opening": "ouverture", "middlegame": "milieu de jeu", "endgame": "finale"}
gaffes = (
    coups[coups["jugement"] == "blunder"]["phase"]
    .value_counts()
    .reindex(ordre)
    .rename(index=noms)
    .rename_axis("phase")
    .reset_index(name="gaffes")
)
fig2 = px.bar(
    gaffes,
    x="phase",
    y="gaffes",
    title="Plus de la moitié de mes gaffes tombent au milieu de jeu",
    labels={"phase": "", "gaffes": "nombre de gaffes"},
)
fig2.show()

# 3. Le score selon l'heure de la partie (heures UTC, au moins 30 parties par heure)
h = df.groupby("heure", as_index=False).agg(n=("score", "size"), score=("score", "mean"))
h = h[h["n"] >= 30]
fig3 = px.bar(
    h,
    x="heure",
    y="score",
    hover_data=["n"],
    title="Je marque moins la nuit (22 h-1 h UTC), où je joue le plus",
    labels={"heure": "heure de début (UTC)", "score": "score moyen"},
)
fig3.add_hline(y=0.5, line_dash="dot")
fig3.update_xaxes(dtick=1)
fig3.show()
