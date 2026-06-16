from altsignal.models import Entity


def test_short_name_strips_allcaps_corporate_suffixes():
    assert Entity(query="WGO", name="WINNEBAGO INDUSTRIES, INC.").short_name == "Winnebago Industries"
    assert Entity(query="AAPL", name="APPLE INC").short_name == "Apple"
    assert Entity(query="HD", name="THE HOME DEPOT, INC.").short_name == "The Home Depot"


def test_short_name_handles_mixed_case_and_state_markers():
    assert Entity(query="GOOG", name="Alphabet Inc.").short_name == "Alphabet"
    assert Entity(query="X", name="ACME CORP /DE/").short_name == "Acme"


def test_short_name_falls_back_to_query():
    assert Entity(query="Some Brand").short_name == "Some Brand"
