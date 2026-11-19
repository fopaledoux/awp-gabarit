import analyses
import pandas as pd

import data


def test_verdict_tient_compte_de_la_marge():
    assert analyses.verdict(0.30, 0.10) == "À travailler"  # sous 50 % même avec la marge
    assert analyses.verdict(0.40, 0.15) == "À surveiller"  # sous 50 %, mais peut-être du hasard
    assert analyses.verdict(0.52, 0.10) == "Dans la moyenne"
    assert analyses.verdict(0.70, 0.10) == "Solide"


def test_ouvertures_pire_en_premier_avec_marge_et_verdict():
    f = analyses.ouvertures(data.load_data(), 30)
    assert f["score"].is_monotonic_increasing
    assert (f["marge"] > 0).all()
    assert set(f["verdict"]) <= {"À travailler", "À surveiller", "Dans la moyenne", "Solide"}


def test_gaffes_par_phase_utilise_phases_sans_opening_theory():
    p = analyses.load_phases()
    assert "opening_theory" not in set(p["phase"])
    g = analyses.gaffes_par_phase(p)
    assert list(g["phase"]) == ["ouverture", "milieu de jeu", "finale"]
    assert (g["taux"] > 0).all()


def test_types_gaffes_part_somme_a_un_par_phase():
    t = analyses.types_gaffes(data.load_coups(), "gaffes")
    assert t.groupby("phase")["part"].sum().round(6).eq(1).all()


def _serie(gains: list[float]) -> pd.DataFrame:
    """Une série d'Elo : un point par jour, le niveau suit `gains` par tranches égales."""
    n = len(gains)
    return pd.DataFrame(
        {"date_jeu": pd.date_range("2026-01-01", periods=n, freq="D"), "elo_joueur": gains}
    )


def test_tendance_elo_detecte_un_plateau():
    elo = [1500 + i for i in range(150)] + [1650] * 150  # monte, puis stagne
    t = analyses.tendance_elo(_serie(elo))
    assert t["total"] > 30
    assert abs(t["recent"]) < 15
    texte = analyses.lecture_elo("Test", t, pd.Timestamp("2026-10-27"))
    assert "plateau" in texte


def test_tendance_elo_detecte_une_progression_continue():
    t = analyses.tendance_elo(_serie([1500 + i for i in range(300)]))
    assert t["recent"] > 15
    assert "progressez" in analyses.lecture_elo("Test", t, pd.Timestamp("2026-10-27"))


def test_lecture_echantillon_previent_quand_il_y_a_peu_de_parties():
    assert analyses.lecture_echantillon(50)[0] == "warning"
    assert analyses.lecture_echantillon(200)[0] == "info"
    assert analyses.lecture_echantillon(2000) is None
