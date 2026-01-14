"""
1099-INT Atoms and Bindings: Portability Test.

This module defines atoms and bindings for Form 1099-INT (Interest Income)
to validate that the semantic core architecture works across multiple forms.

Key differences from 1099-DIV:
- No holding period requirements (interest accrues daily)
- OID (Original Issue Discount) accrual rules
- Tax-exempt vs taxable interest distinction
- Bond premium amortization options
- Market discount treatment

Boxes on 1099-INT:
- Box 1: Interest income (total)
- Box 2: Early withdrawal penalty
- Box 3: Interest on U.S. Savings Bonds and Treasury obligations
- Box 4: Federal income tax withheld
- Box 5: Investment expenses
- Box 6: Foreign tax paid
- Box 8: Tax-exempt interest
- Box 9: Specified private activity bond interest (AMT)
- Box 10: Market discount
- Box 11: Bond premium
- Box 12: Bond premium on Treasury obligations
- Box 13: Bond premium on tax-exempt bonds
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional

from vaas.core.atoms import (
    AtomType,
    AtomKind,
    SemanticAtom,
    ThresholdAtom,
    ScopeAtom,
    ATOM_REGISTRY,
    ATOM_THRESHOLD_10_DOLLARS,  # Reused from core!
)
# NOTE: DerivationAtom, DisclosureAtom removed - those are rules, not atoms.
# Use DeriveInstruction and DiscloseInstruction from operators.py instead.
from vaas.core.bindings import (
    CarrierType,
    SemanticBinding,
    BINDING_REGISTRY,
)


# =============================================================================
# 1099-INT SPECIFIC ATOM TYPES
# =============================================================================

@dataclass
class OIDAccrualAtom(SemanticAtom):
    """
    Atom for Original Issue Discount accrual rules.

    OID accrues daily using constant yield method.
    """
    accrual_method: str = "constant_yield"  # or "ratable"
    de_minimis_threshold_bps: int = 25  # basis points
    governed_by: str = ""

    def __post_init__(self):
        self.atom_type = AtomType.TEMPORAL_WINDOW
        self.atom_kind = AtomKind.TEMPORAL


# NOTE: BondPremiumAtom removed - "election to amortize" is a taxpayer CHOICE,
# not a fact about the world. Elections should be modeled as FactSet inputs,
# not atoms. The election affects how rules evaluate, not what facts exist.


@dataclass
class MarketDiscountAtom(SemanticAtom):
    """
    Atom for market discount treatment.

    Market discount is generally ordinary income upon disposition.
    """
    recognition_timing: str = "disposition"  # or "accrual_election"
    de_minimis_threshold_bps: int = 25
    governed_by: str = ""

    def __post_init__(self):
        self.atom_type = AtomType.DAYCOUNT_RULE
        self.atom_kind = AtomKind.TEMPORAL


# =============================================================================
# 1099-INT CANONICAL ATOMS
# =============================================================================

# OID Accrual - constant yield method per IRC 1272
ATOM_OID_CONSTANT_YIELD = OIDAccrualAtom(
    atom_id="int:oid_constant_yield",
    atom_type=AtomType.TEMPORAL_WINDOW,
    description="OID accrues daily using constant yield method",
    accrual_method="constant_yield",
    de_minimis_threshold_bps=25,
    governed_by="IRC_1272",
)

# De minimis OID threshold
ATOM_OID_DE_MINIMIS = ThresholdAtom(
    atom_id="int:oid_de_minimis",
    atom_type=AtomType.THRESHOLD,
    description="De minimis OID: 0.25% per year to maturity",
    amount=25,  # basis points
    currency="BPS",
    comparison="lt",
    governed_by="IRC_1273(a)(3)",
)

# NOTE: Bond premium election removed - taxpayer elections are inputs, not atoms.
# Model elections as FactSet.actor.elections or similar.

# Market discount accrual election
ATOM_MARKET_DISCOUNT_ACCRUAL = MarketDiscountAtom(
    atom_id="int:market_discount_accrual",
    atom_type=AtomType.DAYCOUNT_RULE,
    description="Election to accrue market discount currently",
    recognition_timing="accrual_election",
    de_minimis_threshold_bps=25,
    governed_by="IRC_1278",
)

# Tax-exempt interest scope (for Box 8)
ATOM_TAX_EXEMPT_SCOPE = ScopeAtom(
    atom_id="int:tax_exempt_scope",
    atom_type=AtomType.ACTOR_REQUIREMENT,
    description="Municipal bond interest exempt from federal tax",
    actor_types=["municipal_bond_issuer", "qualified_501c3"],
    inclusion=True,
    governed_by="IRC_103",
)

# AMT interest scope (for Box 9)
ATOM_AMT_INTEREST_SCOPE = ScopeAtom(
    atom_id="int:amt_pab_scope",
    atom_type=AtomType.ACTOR_REQUIREMENT,
    description="Private activity bond interest subject to AMT",
    actor_types=["private_activity_bond_issuer"],
    inclusion=True,
    governed_by="IRC_57(a)(5)",
)

# NOTE: Derivation removed - "Box 1 = taxable + OID" is a COMPUTATION rule,
# not a fact. Use DeriveInstruction in operators.py:
#   DeriveInstruction(
#       target_id="interest_income_total",
#       operator="SUM",
#       source_ids=["taxable_interest", "oid_interest"],
#   )


# =============================================================================
# 1099-INT CANONICAL BINDINGS
# =============================================================================

BINDING_INTEREST_INCOME_BOX1 = SemanticBinding(
    semantic_id="interest_income_total",
    description="Total interest income (Box 1)",
    form_bindings={"1099-INT": "box_1"},
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_EARLY_WITHDRAWAL_PENALTY = SemanticBinding(
    semantic_id="early_withdrawal_penalty",
    description="Early withdrawal penalty (Box 2)",
    form_bindings={"1099-INT": "box_2"},
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_US_SAVINGS_BOND_INTEREST = SemanticBinding(
    semantic_id="us_savings_bond_interest",
    description="Interest on U.S. Savings Bonds and Treasury obligations (Box 3)",
    form_bindings={"1099-INT": "box_3"},
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

# Note: federal_tax_withheld has same semantic_id as 1099-DIV - demonstrates cross-form semantics!
BINDING_FEDERAL_TAX_WITHHELD_INT = SemanticBinding(
    semantic_id="federal_tax_withheld",
    description="Federal income tax withheld (Box 4)",
    form_bindings={"1099-INT": "box_4"},
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_INVESTMENT_EXPENSES = SemanticBinding(
    semantic_id="investment_expenses",
    description="Investment expenses (Box 5)",
    form_bindings={"1099-INT": "box_5"},
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

# Note: foreign_tax_paid has same semantic_id as 1099-DIV - demonstrates cross-form semantics!
BINDING_FOREIGN_TAX_PAID_INT = SemanticBinding(
    semantic_id="foreign_tax_paid",
    description="Foreign tax paid (Box 6)",
    form_bindings={"1099-INT": "box_6"},
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_TAX_EXEMPT_INTEREST = SemanticBinding(
    semantic_id="tax_exempt_interest",
    description="Tax-exempt interest (Box 8)",
    form_bindings={"1099-INT": "box_8"},
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_PAB_INTEREST_AMT = SemanticBinding(
    semantic_id="pab_interest_amt",
    description="Specified private activity bond interest (Box 9)",
    form_bindings={"1099-INT": "box_9"},
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_MARKET_DISCOUNT = SemanticBinding(
    semantic_id="market_discount",
    description="Market discount (Box 10)",
    form_bindings={"1099-INT": "box_10"},
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_BOND_PREMIUM = SemanticBinding(
    semantic_id="bond_premium",
    description="Bond premium (Box 11)",
    form_bindings={"1099-INT": "box_11"},
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_BOND_PREMIUM_TREASURY = SemanticBinding(
    semantic_id="bond_premium_treasury",
    description="Bond premium on Treasury obligations (Box 12)",
    form_bindings={"1099-INT": "box_12"},
    carrier_type=CarrierType.BOX,
    data_type="amount",
)

BINDING_BOND_PREMIUM_TAX_EXEMPT = SemanticBinding(
    semantic_id="bond_premium_tax_exempt",
    description="Bond premium on tax-exempt bonds (Box 13)",
    form_bindings={"1099-INT": "box_13"},
    carrier_type=CarrierType.BOX,
    data_type="amount",
)


# =============================================================================
# REGISTRATION
# =============================================================================

def register_1099_int_atoms() -> None:
    """
    Register all 1099-INT specific atoms.

    NOTE: Only pure atoms (evaluate to values/booleans) are registered.
    Effectful operations (derivations, disclosures) use operator instructions.
    """
    atoms = [
        # Temporal atoms (evaluate to time/method values)
        ATOM_OID_CONSTANT_YIELD,
        ATOM_MARKET_DISCOUNT_ACCRUAL,
        # Threshold atoms (evaluate to numbers)
        ATOM_OID_DE_MINIMIS,
        # Scope atoms (evaluate to booleans)
        ATOM_TAX_EXEMPT_SCOPE,
        ATOM_AMT_INTEREST_SCOPE,
    ]

    for atom in atoms:
        if not ATOM_REGISTRY.contains(atom.atom_id):
            ATOM_REGISTRY.register(atom)


def register_1099_int_bindings() -> None:
    """Register all 1099-INT bindings."""
    bindings = [
        BINDING_INTEREST_INCOME_BOX1,
        BINDING_EARLY_WITHDRAWAL_PENALTY,
        BINDING_US_SAVINGS_BOND_INTEREST,
        BINDING_FEDERAL_TAX_WITHHELD_INT,
        BINDING_INVESTMENT_EXPENSES,
        BINDING_FOREIGN_TAX_PAID_INT,
        BINDING_TAX_EXEMPT_INTEREST,
        BINDING_PAB_INTEREST_AMT,
        BINDING_MARKET_DISCOUNT,
        BINDING_BOND_PREMIUM,
        BINDING_BOND_PREMIUM_TREASURY,
        BINDING_BOND_PREMIUM_TAX_EXEMPT,
    ]

    for binding in bindings:
        # Use semantic_id as the key - bindings are looked up by semantic concept
        if BINDING_REGISTRY.get(binding.semantic_id) is None:
            BINDING_REGISTRY.register(binding)


def initialize_1099_int() -> None:
    """Initialize all 1099-INT atoms and bindings."""
    register_1099_int_atoms()
    register_1099_int_bindings()
