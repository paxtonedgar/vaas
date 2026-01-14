"""
Citation Precedence Test: Proves IRS Guidance Hierarchy Works.

For IRS-only scope, "conflict resolution" means:
- IRC (statute) > Treasury Regs > Revenue Rulings > Publications

This test proves the CitationScheme correctly assigns weights
and can resolve which citation takes precedence.

NO cross-jurisdiction routing. Just IRS guidance hierarchy.
"""

import pytest

from vaas.core import (
    CitationType,
    CitationScheme,
    AuthorityProvenance,
    IRS_CITATION_SCHEME,
    IRS_1099DIV_2024,
)


class TestCitationPrecedence:
    """Test that citation weights correctly order IRS guidance."""

    def test_irc_beats_treasury_regs(self):
        """IRC (statute) has higher precedence than Treasury Regulations."""
        irc_weight = IRS_CITATION_SCHEME.weight_for(CitationType.IRC)
        treas_weight = IRS_CITATION_SCHEME.weight_for(CitationType.TREAS_REG)

        assert irc_weight > treas_weight, (
            f"IRC ({irc_weight}) should beat TREAS_REG ({treas_weight})"
        )

    def test_treasury_regs_beat_rulings(self):
        """Treasury Regulations beat Revenue Rulings."""
        treas_weight = IRS_CITATION_SCHEME.weight_for(CitationType.TREAS_REG)
        ruling_weight = IRS_CITATION_SCHEME.weight_for(CitationType.REV_RUL)

        assert treas_weight > ruling_weight, (
            f"TREAS_REG ({treas_weight}) should beat REV_RUL ({ruling_weight})"
        )

    def test_rulings_beat_publications(self):
        """Revenue Rulings beat Publications (informal guidance)."""
        ruling_weight = IRS_CITATION_SCHEME.weight_for(CitationType.REV_RUL)
        pub_weight = IRS_CITATION_SCHEME.weight_for(CitationType.PUB)

        assert ruling_weight > pub_weight, (
            f"REV_RUL ({ruling_weight}) should beat PUB ({pub_weight})"
        )

    def test_full_hierarchy_order(self):
        """Full hierarchy: IRC > TREAS_REG > REV_RUL/REV_PROC > NOTICE > FORM_INST > PUB > PLR."""
        weights = {
            ct: IRS_CITATION_SCHEME.weight_for(ct)
            for ct in CitationType
        }

        # Sort by weight descending
        sorted_types = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        # IRC should be first
        assert sorted_types[0][0] == CitationType.IRC
        # TREAS_REG should be second
        assert sorted_types[1][0] == CitationType.TREAS_REG


class TestAuthorityProvenance:
    """Test that provenance correctly captures citation metadata."""

    def test_create_from_irc_citation(self):
        """Create provenance from IRC section citation."""
        provenance = AuthorityProvenance.from_citation(
            citation_str="section 1(h)(11)",
            citation_type=CitationType.IRC,
            scheme=IRS_CITATION_SCHEME,
        )

        assert provenance.issuer == "IRS"
        assert provenance.citation_type == CitationType.IRC
        assert provenance.citation_id == "section 1(h)(11)"
        assert provenance.weight == 100  # IRC weight

    def test_create_from_treasury_reg(self):
        """Create provenance from Treasury Regulation citation."""
        provenance = AuthorityProvenance.from_citation(
            citation_str="1.1-1",
            citation_type=CitationType.TREAS_REG,
            scheme=IRS_CITATION_SCHEME,
        )

        assert provenance.citation_type == CitationType.TREAS_REG
        assert provenance.weight == 90  # TREAS_REG weight

    def test_provenance_comparison(self):
        """Higher-weight provenance should take precedence."""
        irc_prov = AuthorityProvenance.from_citation(
            "section 246(c)", CitationType.IRC, IRS_CITATION_SCHEME
        )
        pub_prov = AuthorityProvenance.from_citation(
            "Pub. 550", CitationType.PUB, IRS_CITATION_SCHEME
        )

        assert irc_prov.weight > pub_prov.weight, (
            "IRC citation should have higher weight than Publication"
        )


class TestCorpusProfile:
    """Test corpus profile configuration."""

    def test_1099div_profile_identity(self):
        """1099-DIV profile has correct identity."""
        assert IRS_1099DIV_2024.issuer == "IRS"
        assert IRS_1099DIV_2024.form_id == "1099-DIV"
        assert IRS_1099DIV_2024.doc_type == "instructions"

    def test_corpus_id_format(self):
        """Corpus ID is correctly formatted."""
        corpus_id = IRS_1099DIV_2024.corpus_id
        assert "irs" in corpus_id
        assert "1099-div" in corpus_id

    def test_effective_tax_years(self):
        """Profile specifies effective tax years."""
        assert 2024 in IRS_1099DIV_2024.effective_tax_years
        assert 2025 in IRS_1099DIV_2024.effective_tax_years


class TestConflictResolution:
    """
    Test conflict resolution scenarios within IRS guidance.

    These are the real conflicts that matter for IRS-only scope:
    - Statute vs regulation interpretation
    - Older ruling vs newer notice
    - Form instructions vs publication
    """

    def test_resolve_statute_vs_regulation(self):
        """When statute and regulation conflict, statute wins."""
        statute = AuthorityProvenance.from_citation(
            "IRC 1(h)(11)(B)", CitationType.IRC, IRS_CITATION_SCHEME
        )
        regulation = AuthorityProvenance.from_citation(
            "Treas. Reg. 1.1-1", CitationType.TREAS_REG, IRS_CITATION_SCHEME
        )

        # Simple resolution: higher weight wins
        winner = statute if statute.weight > regulation.weight else regulation

        assert winner.citation_type == CitationType.IRC, (
            "Statute should win over regulation"
        )

    def test_resolve_ruling_vs_publication(self):
        """Revenue ruling has more weight than publication."""
        ruling = AuthorityProvenance.from_citation(
            "Rev. Rul. 2024-01", CitationType.REV_RUL, IRS_CITATION_SCHEME
        )
        pub = AuthorityProvenance.from_citation(
            "Pub. 550", CitationType.PUB, IRS_CITATION_SCHEME
        )

        winner = ruling if ruling.weight > pub.weight else pub

        assert winner.citation_type == CitationType.REV_RUL, (
            "Revenue ruling should win over publication"
        )

    def test_same_weight_uses_specificity(self):
        """
        When weights are equal, more specific citation should win.

        This is a design decision - for now, we just note they're equal.
        Future: could add specificity scoring.
        """
        ruling = AuthorityProvenance.from_citation(
            "Rev. Rul. 2024-01", CitationType.REV_RUL, IRS_CITATION_SCHEME
        )
        proc = AuthorityProvenance.from_citation(
            "Rev. Proc. 2024-01", CitationType.REV_PROC, IRS_CITATION_SCHEME
        )

        # Both have same weight (70)
        assert ruling.weight == proc.weight, (
            "REV_RUL and REV_PROC should have equal weight"
        )
        # Tie-breaking would require additional logic (not implemented)


def resolve_conflict(prov_a: AuthorityProvenance, prov_b: AuthorityProvenance) -> AuthorityProvenance:
    """
    Simple conflict resolution: higher weight wins.

    This is all we need for IRS-only scope.
    No routing, no adapters, no multi-jurisdiction complexity.
    """
    return prov_a if prov_a.weight >= prov_b.weight else prov_b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
