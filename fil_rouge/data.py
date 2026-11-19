"""Exo 1 : lire mes parties d'échecs. Environ 25 min.

Question : je joue depuis 2023, je stagne, qu'est-ce que je devrais travailler ?
À produire : un DataFrame propre, prêt pour les figures de l'exo 1 et pour l'appli de l'exo 2.
Les colonnes et leurs pièges sont dans data/echecs/LISEZMOI.md : lisez-le d'abord.
Lancer : uv run python fil_rouge/data.py
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "echecs"

COLONNES = [
    "partie_id",
    "date_jeu",
    "cadence",
    "couleur",
    "resultat",
    "elo_joueur",
    "elo_adverse",
    "ouverture_famille",
    "gaffes_joueur",
]
NUMERIQUES = ["elo_joueur", "elo_adverse", "gaffes_joueur"]
RESULTATS = {"win": "victoire", "draw": "nulle", "lose": "défaite"}
SCORES = {"victoire": 1.0, "nulle": 0.5, "défaite": 0.0}


def valider(df: pd.DataFrame) -> pd.DataFrame:
    """Plan B : s'arrêter avec un message clair si le fichier n'a pas la forme attendue.

    Colonnes absentes, colonnes entièrement vides, nombres lus comme du texte,
    identifiant de partie en double, résultat inconnu.
    """
    absentes = [c for c in COLONNES if c not in df.columns]
    if absentes:
        raise ValueError(f"parties.csv : colonnes absentes {absentes}")
    vides = [c for c in COLONNES if df[c].isna().all()]
    if vides:
        raise ValueError(f"parties.csv : colonnes entièrement vides {vides}")
    texte = [c for c in NUMERIQUES if not pd.api.types.is_numeric_dtype(df[c])]
    if texte:
        raise ValueError(f"parties.csv : colonnes lues comme du texte au lieu de nombres {texte}")
    doublons = df.loc[df["partie_id"].duplicated(), "partie_id"].unique().tolist()
    if doublons:
        raise ValueError(f"parties.csv : partie_id en double {doublons[:5]}")
    inconnus = set(df["resultat"].dropna()) - set(RESULTATS)
    if inconnus or df["resultat"].isna().any():
        raise ValueError(f"parties.csv : résultat inconnu ou vide {sorted(inconnus)}")
    return df


def load_data() -> pd.DataFrame:
    """Une ligne par partie (data/echecs/parties.csv), validée.

    Ajoutez : le résultat de mon point de vue en français, un score (1, 0,5 ou 0),
    l'année, l'heure, et l'écart d'Elo avec l'adversaire. Une cadence vide veut dire
    « aucune » : ne la laissez pas disparaître des filtres.
    Les exos suivants lisent ces colonnes, gardez ces noms : date_jeu (en datetime),
    resultat (victoire, nulle, défaite), score, annee, heure (UTC), ecart_elo
    (Elo de l'adversaire moins le mien), cadence.
    """
    df = valider(pd.read_csv(DATA / "parties.csv"))
    # Dates UTC sans fuseau écrit : on les garde « naïves », en sachant que c'est de l'UTC.
    df["date_jeu"] = pd.to_datetime(df["date_jeu"])
    df["resultat"] = df["resultat"].map(RESULTATS)
    df["score"] = df["resultat"].map(SCORES)
    df["annee"] = df["date_jeu"].dt.year
    df["heure"] = df["date_jeu"].dt.hour
    # Vide (NaN) pour les parties sans Elo d'avant : premières de série Chess.com, importées.
    df["ecart_elo"] = df["elo_adverse"] - df["elo_joueur"]
    # Correspondance, sans pendule, importées : sans « aucune », un filtre les ferait disparaître.
    df["cadence"] = df["cadence"].fillna("aucune")
    return df


def load_coups() -> pd.DataFrame:
    """Un de mes coups jugés mauvais par ligne (data/echecs/coups_juges.csv)."""
    return pd.read_csv(DATA / "coups_juges.csv")


def familles(df: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """Score moyen par famille d'ouvertures, la pire d'abord.

    Colonnes : famille, n (parties), score, gaffes (par partie).
    Sous min_n parties le chiffre ne dit rien : écartez la famille.
    """
    f = (
        df.dropna(subset=["ouverture_famille"])
        .groupby("ouverture_famille", as_index=False)
        .agg(n=("partie_id", "size"), score=("score", "mean"), gaffes=("gaffes_joueur", "mean"))
        .rename(columns={"ouverture_famille": "famille"})
    )
    return f[f["n"] >= min_n].sort_values("score").reset_index(drop=True)


if __name__ == "__main__":
    d = load_data()
    print(d.shape)
    print(d["cadence"].value_counts())
    print(familles(d).head(10))
