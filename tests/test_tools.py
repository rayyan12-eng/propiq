import pytest

from services import tools


def test_estimate_mortgage_basic():
    result = tools.estimate_mortgage(price_aed=1_000_000, down_payment_pct=20, annual_rate_pct=4.0, term_years=25)
    assert result["down_payment_aed"] == pytest.approx(200_000, rel=1e-6)
    assert result["loan_amount_aed"] == pytest.approx(800_000, rel=1e-6)
    assert result["monthly_payment_aed"] > 0
    # sanity: total interest should be less than the principal at 4% over 25y, but positive
    assert 0 < result["total_interest_aed"] < result["loan_amount_aed"] * 2


def test_estimate_mortgage_zero_rate():
    result = tools.estimate_mortgage(price_aed=500_000, down_payment_pct=0, annual_rate_pct=0, term_years=10)
    assert result["monthly_payment_aed"] == pytest.approx(500_000 / 120, rel=1e-6)
    assert result["total_interest_aed"] == pytest.approx(0, abs=1e-6)


def test_estimate_mortgage_invalid_down_payment():
    with pytest.raises(ValueError):
        tools.estimate_mortgage(price_aed=500_000, down_payment_pct=150)


def test_get_comparable_listings_shape():
    listings = tools.get_comparable_listings(area="JVC", bedrooms=2, limit=3)
    assert isinstance(listings, list)
    assert len(listings) <= 3
    for row in listings:
        assert row["area"] == "JVC"
        assert row["bedrooms"] == 2


def test_get_comparable_listings_unknown_area():
    listings = tools.get_comparable_listings(area="Nonexistent Area", bedrooms=2)
    assert listings == []


def test_get_neighborhood_stats_known_area():
    stats = tools.get_neighborhood_stats("Downtown Dubai")
    assert stats["area"] == "Downtown Dubai"
    assert stats["sample_size"] > 0
    assert stats["median_price_aed"] > 0
    assert stats["median_gross_yield_pct"] > 0


def test_get_neighborhood_stats_unknown_area():
    stats = tools.get_neighborhood_stats("Nowhere")
    assert "error" in stats
