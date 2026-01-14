"""
Form-Specific Atoms and Bindings.

This package contains form-specific configurations that build on
the core semantic infrastructure.

Each form module validates that the architecture is portable across
different IRS information returns.

Available forms:
- form_1099_div: Dividend income (primary development target)
- form_1099_int: Interest income (portability test)
"""

from vaas.core.forms.form_1099_int import (
    # Atom types
    OIDAccrualAtom,
    BondPremiumAtom,
    MarketDiscountAtom,
    # Canonical atoms
    ATOM_OID_CONSTANT_YIELD,
    ATOM_OID_DE_MINIMIS,
    ATOM_BOND_PREMIUM_ELECTION,
    ATOM_MARKET_DISCOUNT_ACCRUAL,
    ATOM_TAX_EXEMPT_SCOPE,
    ATOM_AMT_INTEREST_SCOPE,
    ATOM_INTEREST_DERIVATION,
    # Bindings
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
    # Registration
    register_1099_int_atoms,
    register_1099_int_bindings,
    initialize_1099_int,
)

__all__ = [
    # Atom types
    "OIDAccrualAtom",
    "BondPremiumAtom",
    "MarketDiscountAtom",
    # Canonical atoms
    "ATOM_OID_CONSTANT_YIELD",
    "ATOM_OID_DE_MINIMIS",
    "ATOM_BOND_PREMIUM_ELECTION",
    "ATOM_MARKET_DISCOUNT_ACCRUAL",
    "ATOM_TAX_EXEMPT_SCOPE",
    "ATOM_AMT_INTEREST_SCOPE",
    "ATOM_INTEREST_DERIVATION",
    # Bindings
    "BINDING_INTEREST_INCOME_BOX1",
    "BINDING_EARLY_WITHDRAWAL_PENALTY",
    "BINDING_US_SAVINGS_BOND_INTEREST",
    "BINDING_FEDERAL_TAX_WITHHELD_INT",
    "BINDING_INVESTMENT_EXPENSES",
    "BINDING_FOREIGN_TAX_PAID_INT",
    "BINDING_TAX_EXEMPT_INTEREST",
    "BINDING_PAB_INTEREST_AMT",
    "BINDING_MARKET_DISCOUNT",
    "BINDING_BOND_PREMIUM",
    "BINDING_BOND_PREMIUM_TREASURY",
    "BINDING_BOND_PREMIUM_TAX_EXEMPT",
    # Registration
    "register_1099_int_atoms",
    "register_1099_int_bindings",
    "initialize_1099_int",
]
