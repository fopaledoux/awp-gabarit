"""Aides à la décision pour l'appli de l'exo 2 : chiffres, marges d'erreur et textes qui expliquent.

Chaque fonction `lecture_*` renvoie une phrase (markdown) qui dépend de la situation : le même
graphique ne se lit pas pareil selon le nombre de parties, l'écart à 50 % ou la tendance.
Rien ici ne dépend de Streamlit : on peut tout tester avec pytest.
"""

import numpy as np
import pandas as pd

from data import DATA, familles

Z = 1.96  # intervalle à 95 %
FAIBLE, MOYEN = 100, 300  # sous ces nombres de parties, les tendances sont fragiles

PHASES = {"opening": "ouverture", "middlegame": "milieu de jeu", "endgame": "finale"}
DANS = {"ouverture": "en ouverture", "milieu de jeu": "au milieu de jeu", "finale": "en finale"}
TYPES = {
    "allowed_tactic": "Tactique laissée à l'adversaire",
    "missed_tactic": "Tactique ratée",
}
POSITIONNELLE = "Faute de position"
GRAVITES = {
    "gaffes": ["blunder"],
    "erreurs et gaffes": ["mistake", "blunder"],
    "toutes les fautes": ["inaccuracy", "mistake", "blunder"],
}
FINS = {
    "checkmate": "mat",
    "resign": "abandon",
    "timeout": "temps",
    "draw": "nulle",
    "other": "autre",
}
PLATEFORMES = {"chesscom": "Chess.com", "lichess": "Lichess", "imported": "Importées"}


# ---------------------------------------------------------------- chargement


def load_phases() -> pd.DataFrame:
    """Une ligne par phase de partie (data/echecs/phases.csv), sans `opening_theory`.

    `opening_theory` est incluse dans `opening` : l'additionner compterait deux fois.
    """
    p = pd.read_csv(DATA / "phases.csv")
    return p[p["phase"].isin(PHASES)]


# ---------------------------------------------------------------- statistiques de base


def verdict(score: float, marge: float) -> str:
    """Compare un score à 50 % en tenant compte de l'incertitude (marge = demi-intervalle à 95 %).

    « À travailler » : même en comptant la marge, le score reste sous 50 %.
    « À surveiller » : sous 50 %, mais l'écart pourrait être du hasard.
    """
    if score + marge < 0.5:
        return "À travailler"
    if score < 0.5:
        return "À surveiller"
    if score - marge > 0.5:
        return "Solide"
    return "Dans la moyenne"


def score_par(vue: pd.DataFrame, colonne: str, min_n: int = 30) -> pd.DataFrame:
    """Score moyen par modalité de `colonne`, avec marge d'erreur et verdict.

    Colonnes : colonne, n, score, marge (± en points de 0 à 1), verdict. Écarte n < min_n.
    """
    g = vue.groupby(colonne, observed=True)["score"].agg(n="size", score="mean", sd="std")
    g["marge"] = Z * g["sd"] / np.sqrt(g["n"])
    g["verdict"] = [verdict(s, m) for s, m in zip(g["score"], g["marge"], strict=True)]
    return g[g["n"] >= min_n].drop(columns="sd").reset_index()


def ouvertures(vue: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """Les familles d'ouvertures de `familles()`, avec marge d'erreur et verdict, la pire d'abord."""
    f = familles(vue, min_n)
    if f.empty:
        return f.assign(marge=[], verdict=[])
    sd = vue.groupby("ouverture_famille")["score"].std()
    f["marge"] = Z * f["famille"].map(sd) / np.sqrt(f["n"])
    f["verdict"] = [verdict(s, m) for s, m in zip(f["score"], f["marge"], strict=True)]
    return f


def lecture_echantillon(n: int) -> tuple[str, str] | None:
    """Prévient quand il y a trop peu de parties pour tirer des conclusions. (niveau, texte)"""
    if n < FAIBLE:
        return "warning", (
            f"**{n} parties seulement.** Avec si peu de parties, un score de 60 % peut être du "
            "pur hasard : lisez les tendances comme des pistes, pas comme des preuves."
        )
    if n < MOYEN:
        return "info", (
            f"{n} parties : assez pour les grandes tendances, trop peu pour comparer des "
            "ouvertures précises. Regardez les marges d'erreur (barres noires sur les figures)."
        )
    return None


def lecture_score(score: float, ref: float, n: int) -> str:
    """Que dit le score moyen de la sélection."""
    marge = Z * 0.45 / np.sqrt(n)  # écart-type d'un score d'une partie ≈ 0,45
    if abs(score - 0.5) <= marge:
        return (
            f"Score de {score:.0%} : **dans la marge de 50 %** (±{marge:.0%}). Un joueur régulier "
            "marque autour de 50 % parce que le classement lui donne des adversaires de son niveau : "
            "ce n'est pas un signe de stagnation, regardez l'onglet Progression."
        )
    if score < 0.5:
        return (
            f"Score de {score:.0%} : **sous 50 %** au-delà de la marge (±{marge:.0%}). "
            "Sur cette sélection, vous perdez plus que vous ne gagnez."
        )
    return (
        f"Score de {score:.0%} : **au-dessus de 50 %** au-delà de la marge (±{marge:.0%}). "
        f"{'Mieux' if score > ref else 'Un peu moins bien'} que sur l'ensemble de vos parties ({ref:.0%})."
    )


def lecture_ouvertures(f: pd.DataFrame, min_n: int) -> str:
    """Quelle conclusion tirer du classement des ouvertures."""
    if f.empty:
        return f"Aucune famille n'a {min_n} parties : baissez le minimum dans la barre latérale."
    sûres = f[f["verdict"] == "À travailler"]
    a_voir = f[f["verdict"] == "À surveiller"]
    if not sûres.empty:
        noms = ", ".join(f"{r.famille} ({r.score:.0%}, {r.n} parties)" for r in sûres.itertuples())
        return (
            f"**À travailler : {noms}.** Même en tenant compte de la marge d'erreur, le score reste "
            "sous 50 % : ce n'est probablement pas du hasard. Rejouez vos parties dans ces "
            "ouvertures pour trouver où vous quittez la théorie, ou changez de variante."
        )
    if not a_voir.empty:
        p = a_voir.iloc[0]
        noms = ", ".join(f"{r.famille} ({r.score:.0%})" for r in a_voir.head(3).itertuples())
        return (
            f"**Aucune ouverture n'est clairement mauvaise.** Les plus faibles sont {noms}, mais "
            f"avec {p.n:.0f} parties la marge d'erreur est de ±{p.marge * 100:.0f} points : l'écart avec "
            "50 % peut venir du hasard. Ne changez pas de répertoire sur cette base. Jouez-les encore "
            "ou regardez le détail des variantes avant de conclure."
        )
    return (
        "**Aucun point faible dans vos ouvertures** : toutes sont dans la moyenne ou au-dessus. "
        "Le travail à faire est ailleurs (gaffes, milieu de jeu)."
    )


# ---------------------------------------------------------------- gaffes


def gaffes_par_phase(phases: pd.DataFrame) -> pd.DataFrame:
    """Gaffes et gaffes pour 100 de mes coups, par phase (source : phases.csv).

    `nb_coups` compte les demi-coups des deux camps : la moitié sont les miens (à un près).
    """
    g = phases.groupby("phase").agg(gaffes=("gaffes_joueur", "sum"), demi=("nb_coups", "sum"))
    g = g.reindex(list(PHASES))
    g["taux"] = 100 * g["gaffes"] / (g["demi"] / 2)
    g.index = g.index.map(PHASES)
    return g.rename_axis("phase").reset_index()


def types_gaffes(coups: pd.DataFrame, gravite: str = "gaffes") -> pd.DataFrame:
    """Nature des fautes : tactique laissée, tactique ratée, ou faute de position, par phase."""
    c = coups[coups["jugement"].isin(GRAVITES[gravite])].copy()
    c["type"] = c["sous_type"].map(TYPES).fillna(POSITIONNELLE)
    c["phase"] = c["phase"].map(PHASES)
    t = c.groupby(["phase", "type"]).size().rename("n").reset_index()
    t["part"] = t["n"] / t.groupby("phase")["n"].transform("sum")
    return t


def temps_restant(coups: pd.DataFrame, gravite: str = "gaffes") -> pd.DataFrame:
    """Part des fautes selon le temps qu'il restait à la pendule après le coup."""
    c = coups[coups["jugement"].isin(GRAVITES[gravite])].dropna(subset=["temps_restant_s"])
    tranches = pd.cut(
        c["temps_restant_s"],
        [-1, 10, 30, 60, np.inf],
        labels=["moins de 10 s", "10 à 30 s", "30 s à 1 min", "plus d'1 min"],
    )
    t = tranches.value_counts().sort_index().rename("n").rename_axis("temps").reset_index()
    t["part"] = t["n"] / t["n"].sum()
    return t


def lecture_types(t: pd.DataFrame, coups: pd.DataFrame, gravite: str) -> str:
    """Ce que révèle la nature des fautes, et quoi s'entraîner."""
    n = t.groupby("type")["n"].sum()
    if n.sum() == 0:
        return "Aucune faute de ce type dans la sélection."
    part = (n / n.sum()).to_dict()
    laissee = part.get(TYPES["allowed_tactic"], 0)
    ratee = part.get(TYPES["missed_tactic"], 0)
    pos = part.get(POSITIONNELLE, 0)
    if laissee + ratee >= 0.6:
        base = (
            f"**{laissee + ratee:.0%} de vos {gravite} sont tactiques** (une pièce en prise, un mat, "
            f"une fourchette…). Seulement {pos:.0%} sont de simples fautes de position. "
        )
        if abs(laissee - ratee) < 0.1:
            return base + (
                f"Elles se partagent à peu près à égalité entre tactiques laissées à l'adversaire "
                f"({laissee:.0%}) et tactiques ratées ({ratee:.0%}) : travaillez les deux, en "
                "commençant par regarder ce que menace l'adversaire avant chaque coup."
            )
        if laissee >= ratee:
            return base + (
                f"Surtout, **{laissee:.0%} sont des tactiques laissées à l'adversaire** contre "
                f"{ratee:.0%} de tactiques ratées : vous ne voyez pas assez ce que l'adversaire "
                "menace. Habitude à prendre : avant chaque coup, passez en revue ses échecs, ses "
                "prises et ses menaces. Les problèmes tactiques « défendre » sont plus utiles que "
                "les problèmes « attaquer »."
            )
        return base + (
            f"Surtout, **{ratee:.0%} sont des tactiques ratées** contre {laissee:.0%} laissées à "
            "l'adversaire : l'occasion est là et vous ne la voyez pas. Les problèmes tactiques "
            "(fourchettes, clouages, attaques à la découverte) sont le bon entraînement."
        )
    return (
        f"**{pos:.0%} de vos {gravite} sont des fautes de position** (plan, structure, activité des "
        "pièces), pas des oublis tactiques. Les problèmes aideront peu : analysez plutôt des parties "
        "complètes et cherchez le coup où le plan change."
    )


def lecture_temps(t: pd.DataFrame, gravite: str) -> str:
    """La pendule explique-t-elle les fautes ?"""
    if t.empty or t["n"].sum() == 0:
        return "Pas de données de pendule pour cette sélection."
    pression = t.loc[t["temps"] == "moins de 10 s", "part"].sum()
    if pression >= 0.25:
        return (
            f"**{pression:.0%} de vos {gravite} arrivent avec moins de 10 s** à la pendule : la "
            "gestion du temps coûte cher. Essayez une cadence plus longue ou fixez-vous un "
            "temps maximum par coup pendant les 15 premiers coups."
        )
    return (
        f"Seulement **{pression:.0%} de vos {gravite} arrivent avec moins de 10 s** à la pendule : "
        "la plupart tombent quand il vous restait du temps. Ce n'est donc pas d'abord un problème "
        "de pendule : c'est un problème de vérification, avant de jouer."
    )


# ---------------------------------------------------------------- progression


def elo_series(vue: pd.DataFrame, min_n: int = 30) -> dict[str, pd.DataFrame]:
    """Une courbe d'Elo par plateforme et cadence : un Elo n'est jamais comparable d'une série à l'autre."""
    out = {}
    d = vue.dropna(subset=["elo_joueur"])
    for (plateforme, cadence), g in d.groupby(["plateforme", "cadence"]):
        if len(g) >= min_n:
            nom = f"{PLATEFORMES.get(plateforme, plateforme)} · {cadence}"
            out[nom] = g.sort_values("date_jeu")
    return out


def tendance_elo(g: pd.DataFrame) -> dict | None:
    """Gain total (début contre fin) et gain récent (3 derniers mois de la série contre les 3 précédents)."""
    if len(g) < 40:
        return None
    k = max(10, len(g) // 10)
    debut, fin = g["elo_joueur"].iloc[:k].mean(), g["elo_joueur"].iloc[-k:].mean()
    t_fin = g["date_jeu"].max()
    trois = pd.Timedelta(days=90)
    recent = g[g["date_jeu"] > t_fin - trois]["elo_joueur"]
    avant = g[(g["date_jeu"] <= t_fin - trois) & (g["date_jeu"] > t_fin - 2 * trois)]["elo_joueur"]
    gain_recent = recent.mean() - avant.mean() if len(recent) >= 15 and len(avant) >= 15 else None
    return {
        "debut": debut,
        "fin": fin,
        "total": fin - debut,
        "recent": gain_recent,
        "derniere": t_fin,
    }


def lecture_elo(nom: str, t: dict, aujourd_hui: pd.Timestamp) -> str:
    """Progression, plateau, recul ou stagnation : une phrase par série."""
    total, recent = t["total"], t["recent"]
    chiffres = f"**{nom}** : {t['debut']:.0f} → {t['fin']:.0f} ({total:+.0f} points)"
    arret = ""
    if (aujourd_hui - t["derniere"]).days > 30:
        arret = (
            f" Dernière partie le {t['derniere']:%d/%m/%Y} : vous ne jouez plus cette série, "
            "« récent » veut dire les 3 derniers mois avant l'arrêt."
        )
    if total >= 30 and recent is not None and recent <= -30:
        txt = (
            f"{chiffres}. Vous aviez progressé, **mais l'Elo a reculé de {-recent:.0f} points sur les "
            "3 derniers mois** : regardez si vos gaffes ont augmenté (onglet Gaffes)."
        )
    elif total >= 30 and recent is not None and recent < 15:
        txt = (
            f"{chiffres}. Vous avez bien progressé, **mais plus depuis 3 mois** ({recent:+.0f} sur "
            "cette période) : c'est un **plateau**. Pour repartir, ciblez un point faible "
            "(onglets Ouvertures et Gaffes) plutôt que de jouer plus de parties."
        )
    elif total >= 30:
        suite = f" ({recent:+.0f} sur les 3 derniers mois)" if recent is not None else ""
        txt = f"{chiffres}. **Vous progressez**{suite} : continuez ce que vous faites."
    elif total > -30:
        txt = (
            f"{chiffres}. **Stagnation** sur toute la période : l'Elo ne bouge pas. Il faut changer "
            "quelque chose, par exemple cibler un point faible."
        )
    else:
        txt = f"{chiffres}. **Recul** : regardez si vos gaffes ont augmenté récemment (onglet Gaffes)."
    return txt + arret


def score_glissant(vue: pd.DataFrame, fenetre: int = 100) -> pd.DataFrame:
    """Score moyen sur les `fenetre` dernières parties : une courbe lisse au lieu de mois bruités."""
    d = vue.sort_values("date_jeu")[["date_jeu", "score"]].reset_index(drop=True)
    d["glissant"] = d["score"].rolling(fenetre, min_periods=fenetre // 2).mean()
    return d.dropna()


# ---------------------------------------------------------------- contexte


def lecture_heures(h: pd.DataFrame) -> str:
    """Les écarts selon l'heure sont-ils réels ?"""
    if h.empty:
        return "Aucune heure n'a assez de parties dans cette sélection."
    sorties = h[h["verdict"].isin(["À travailler", "Solide"])]
    if sorties.empty:
        return (
            "**Aucune heure ne se détache vraiment.** Les écarts que vous voyez sont dans la marge "
            "d'erreur : changer vos horaires de jeu ne changerait probablement rien."
        )
    liste = ", ".join(
        f"{int(r.heure)} h ({r.score:.0%}, {r.n} parties)"
        for r in sorties.sort_values("score").itertuples()
    )
    return (
        f"Heures qui sortent du lot : {liste}. Attention : on teste {len(h)} créneaux, donc **un ou "
        "deux sortent du lot par pur hasard**. Ne changez vos horaires que si le motif est cohérent "
        "(par exemple, tout ce qui est tard le soir va mal : fatigue)."
    )


def lecture_couleur(c: pd.DataFrame) -> str:
    """Blancs contre noirs."""
    if len(c) < 2:
        return "Une seule couleur dans la sélection."
    b = c[c["couleur"] == "white"].iloc[0]
    n = c[c["couleur"] == "black"].iloc[0]
    ecart = b["score"] - n["score"]
    bruit = np.sqrt(b["marge"] ** 2 + n["marge"] ** 2)  # marge de la différence de deux scores
    if abs(ecart) <= bruit:
        return (
            f"Blancs {b['score']:.0%}, noirs {n['score']:.0%} : l'écart de {abs(ecart) * 100:.0f} points "
            "est dans la marge d'erreur. Pas de différence nette entre vos deux couleurs."
        )
    meilleure, pire = ("blancs", "noirs") if ecart > 0 else ("noirs", "blancs")
    return (
        f"Vous marquez nettement mieux avec les **{meilleure}** ({max(b['score'], n['score']):.0%}) "
        f"qu'avec les **{pire}** ({min(b['score'], n['score']):.0%}). Priorité : le répertoire avec les {pire}."
    )


def par_ecart(vue: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """Score selon l'Elo de l'adversaire comparé au mien (tranches), avec marge d'erreur."""
    d = vue.dropna(subset=["ecart_elo"]).copy()
    d["tranche"] = pd.cut(
        d["ecart_elo"],
        [-np.inf, -100, -30, 30, 100, np.inf],
        labels=[
            "Plus faible de 100+",
            "Plus faible de 30 à 100",
            "Équivalent (±30)",
            "Plus fort de 30 à 100",
            "Plus fort de 100+",
        ],
    )
    return score_par(d, "tranche", min_n)


def lecture_ecart(e: pd.DataFrame) -> str:
    """Contre plus fort, contre plus faible."""
    if e.empty:
        return "Pas assez de parties pour comparer selon l'écart d'Elo."
    forts = e[e["tranche"].str.contains("fort")]
    faibles = e[e["tranche"].str.contains("faible")]
    p_fort = np.average(forts["score"], weights=forts["n"]) if not forts.empty else None
    p_faible = np.average(faibles["score"], weights=faibles["n"]) if not faibles.empty else None
    if p_fort is None or p_faible is None:
        return "Pas assez de parties contre des adversaires plus forts ou plus faibles."
    if p_faible < 0.55:
        return (
            f"Contre plus faible que vous, vous ne marquez que {p_faible:.0%} : **vous laissez "
            "échapper des parties gagnables** (relâchement, précipitation). Contre plus fort : "
            f"{p_fort:.0%}."
        )
    return (
        f"Contre plus faible : {p_faible:.0%}. Contre plus fort : {p_fort:.0%}. "
        "Le classement fait son travail : plus l'adversaire est fort, moins vous marquez."
    )


def fins_de_partie(vue: pd.DataFrame) -> pd.DataFrame:
    """Comment finissent mes victoires et mes défaites (part de chaque mode de fin)."""
    d = vue[vue["resultat"].isin(["victoire", "défaite"])].copy()
    d["fin"] = d["fin_partie"].map(FINS)
    t = d.groupby(["resultat", "fin"]).size().rename("n").reset_index()
    t["part"] = t["n"] / t.groupby("resultat")["n"].transform("sum")
    return t


def lecture_fins(t: pd.DataFrame) -> str:
    """Le temps est-il un risque ou un atout ?"""

    def n(res: str, fin: str) -> int:
        m = t[(t["resultat"] == res) & (t["fin"] == fin)]["n"]
        return int(m.iloc[0]) if len(m) else 0

    perdues, gagnees = n("défaite", "temps"), n("victoire", "temps")
    total_def = t[t["resultat"] == "défaite"]["n"].sum()
    if total_def == 0:
        return "Aucune défaite dans la sélection."
    part = perdues / total_def
    if perdues == 0:
        return f"Aucune défaite au temps ({gagnees} victoires au temps) : la pendule n'est pas un souci."
    if gagnees >= perdues:
        return (
            f"{part:.0%} de vos défaites viennent de la pendule ({perdues} parties), mais vous en "
            f"gagnez {gagnees} au temps : **la pendule est plutôt un atout** pour vous."
        )
    return (
        f"**{part:.0%} de vos défaites viennent de la pendule** ({perdues} parties) contre "
        f"{gagnees} victoires au temps : vous perdez plus de parties au temps que vous n'en gagnez. "
        "Entraînez-vous à jouer plus vite dans les positions simples."
    )


# ---------------------------------------------------------------- recommandations


def recommandations(vue, coups, phases, f, min_n: int) -> list[dict]:
    """Jusqu'à trois actions pour la semaine, chacune avec la raison chiffrée."""
    actions = []
    # 1. Ouvertures
    sûres = f[f["verdict"] == "À travailler"] if not f.empty else f
    if not sûres.empty:
        r = sûres.iloc[0]
        actions.append(
            {
                "icone": ":material/menu_book:",
                "titre": f"Réviser {r.famille}",
                "texte": f"Vous y marquez {r.score:.0%} sur {r.n:.0f} parties, sous 50 % même en comptant "
                "la marge d'erreur. Rejouez ces parties pour voir où vous quittez la théorie.",
            }
        )
    else:
        actions.append(
            {
                "icone": ":material/menu_book:",
                "titre": "Ne pas toucher au répertoire",
                "texte": "Aucune ouverture n'est clairement mauvaise : les écarts sont dans la marge "
                "d'erreur. Changer de répertoire serait un pari sans preuve.",
            }
        )
    # 2. Nature des fautes
    t = types_gaffes(coups)
    if t["n"].sum():
        n = t.groupby("type")["n"].sum()
        tot = n.sum()
        laissee = n.get(TYPES["allowed_tactic"], 0) / tot
        ratee = n.get(TYPES["missed_tactic"], 0) / tot
        if laissee + ratee >= 0.6 and laissee >= ratee:
            actions.append(
                {
                    "icone": ":material/shield:",
                    "titre": "Vérifier les menaces avant chaque coup",
                    "texte": f"{laissee:.0%} de vos gaffes sont des tactiques laissées à l'adversaire. "
                    "Avant de jouer : ses échecs, ses prises, ses menaces.",
                }
            )
        elif laissee + ratee >= 0.6:
            actions.append(
                {
                    "icone": ":material/bolt:",
                    "titre": "Faire des problèmes tactiques",
                    "texte": f"{ratee:.0%} de vos gaffes sont des tactiques ratées : l'occasion est là "
                    "et vous ne la voyez pas.",
                }
            )
        else:
            actions.append(
                {
                    "icone": ":material/psychology:",
                    "titre": "Analyser des parties complètes",
                    "texte": f"{1 - laissee - ratee:.0%} de vos gaffes sont des fautes de position : "
                    "les problèmes aideront peu, l'analyse de parties davantage.",
                }
            )
    # 3. Phase la plus fragile
    p = gaffes_par_phase(phases).dropna()
    if not p.empty and p["gaffes"].sum():
        pire = p.loc[p["taux"].idxmax()]
        meilleure = p.loc[p["taux"].idxmin()]
        actions.append(
            {
                "icone": ":material/target:",
                "titre": f"Travailler {pire['phase']}",
                "texte": f"{pire['taux']:.1f} gaffes pour 100 de vos coups {DANS[pire['phase']]}, "
                f"contre {meilleure['taux']:.1f} {DANS[meilleure['phase']]} : c'est la phase où vous "
                "perdez le plus, à durée égale.",
            }
        )
    return actions


def elo_mensuel(series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Elo moyen par mois pour chaque série (colonnes : date_jeu, elo, série)."""
    morceaux = []
    for nom, g in series.items():
        m = g.resample("MS", on="date_jeu")["elo_joueur"].mean().dropna().rename("elo")
        morceaux.append(m.reset_index().assign(série=nom))
    return pd.concat(morceaux, ignore_index=True) if morceaux else pd.DataFrame()


def lecture_phases(p: pd.DataFrame) -> str:
    """Pourquoi regarder le taux de gaffes plutôt que leur nombre."""
    p = p.dropna()
    if p.empty or p["gaffes"].sum() == 0:
        return "Aucune gaffe dans la sélection."
    taux, nombre = p.loc[p["taux"].idxmax()], p.loc[p["gaffes"].idxmax()]
    bas = p.loc[p["taux"].idxmin()]
    base = (
        f"**{taux['phase'].capitalize()}** : {taux['taux']:.1f} gaffes pour 100 de vos coups, "
        f"contre {bas['taux']:.1f} {DANS[bas['phase']]}."
    )
    if taux["phase"] == nombre["phase"]:
        return (
            base
            + " C'est aussi la phase qui en compte le plus en nombre : aucun doute sur la priorité."
        )
    return (
        base
        + f" {nombre['phase'].capitalize()} en compte plus en nombre, mais c'est surtout parce qu'elle "
        "est plus longue : à durée égale, c'est l'autre phase qui vous coûte le plus."
    )
