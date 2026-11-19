import app_dash
import pytest

FILTRES = {
    "cadences": ["blitz"],
    "debut": "2023-07-09",
    "fin": "2026-11-15",
    "couleur": "toutes",
    "min_n": 30,
}
AUCUNE_PARTIE = FILTRES | {"cadences": []}
SANS_CLIC = [
    app_dash.maj_actions,
    app_dash.maj_en_tete,
    app_dash.maj_synthese_familles,
    app_dash.maj_synthese_phases,
    app_dash.maj_tableau_familles,
    app_dash.maj_taux,
    app_dash.maj_rejouer,
    app_dash.maj_elo,
    app_dash.maj_glissant,
    app_dash.maj_heures,
    app_dash.maj_fins,
    app_dash.maj_couleur,
    app_dash.maj_ecart,
]


def test_la_page_se_construit_et_expose_server():
    assert app_dash.server is app_dash.app.server
    assert app_dash.serve_layout() is not None


@pytest.mark.parametrize("callback", SANS_CLIC)
@pytest.mark.parametrize("filtres", [FILTRES, AUCUNE_PARTIE])
def test_chaque_callback_repond_meme_sans_partie(callback, filtres):
    callback(filtres)  # ne plante pas : figure vide ou message


def test_le_clic_sur_une_barre_liste_les_parties_de_la_famille():
    s = app_dash.selection(FILTRES)
    nom = s.f["famille"].iloc[-1]  # la meilleure, pas la valeur par défaut
    clic = {"points": [{"y": nom}]}
    parties = app_dash.maj_parties(FILTRES, clic)
    attendu = (s.vue["ouverture_famille"] == nom).sum()
    assert len(parties) == attendu > 0
    assert {"partie_id", "date", "resultat"} <= set(parties[0])


def test_sans_clic_on_montre_la_famille_la_plus_faible():
    s = app_dash.selection(FILTRES)
    titre, *_ = app_dash.maj_detail_famille(FILTRES, None)
    assert s.f["famille"].iloc[0] in titre


def test_le_clic_sur_une_partie_liste_ses_coups_jugés():
    s = app_dash.selection(FILTRES)
    pid = s.coups["partie_id"].iloc[0]
    coups, titre = app_dash.maj_coups([{"partie_id": pid}])
    assert coups and pid in titre
    assert app_dash.maj_coups([]) == ([], "Cliquez une partie dans le tableau ci-dessus.")


def test_les_filtres_se_normalisent():
    f = app_dash.maj_filtres(["rapid", "blitz"], None, None, "noirs", 20)
    assert f["cadences"] == ["blitz", "rapid"] and f["couleur"] == "noirs"
    assert len(f["debut"]) == len(f["fin"]) == 10
