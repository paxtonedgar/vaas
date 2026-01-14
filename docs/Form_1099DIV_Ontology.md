# Form 1099-DIV Ontology: Rules, Entities, and Semantic Relationships

This ontology models the compliance logic, reporting requirements, and semantic relationships for IRS Form 1099-DIV dividend reporting. It uses two key modeling primitives:

- **Rule** nodes: Capture default behaviors, conditional logic, and compliance requirements
- **Window** nodes: Model holding-period calculations with explicit temporal parameters

---

## Table of Contents

1. [Default Reporting Rules](#1-default-reporting-rules)
2. [Holding Period Rules & Windows](#2-holding-period-rules--windows)
3. [Constructive Receipt Timing](#3-constructive-receipt-timing)
4. [Backup Withholding Cross-Year Reporting](#4-backup-withholding-cross-year-reporting)
5. [Section 1202 Statement Requirements](#5-section-1202-statement-requirements)
6. [IRC Section Attachments](#6-irc-section-attachments)
7. [WHFIT Reporting Thresholds](#7-whfit-reporting-thresholds)
8. [TIN Notice & Compliance Rules](#8-tin-notice--compliance-rules)
9. [TIN Truncation Rules](#9-tin-truncation-rules)
10. [Surrogate Foreign Corporation Rules](#10-surrogate-foreign-corporation-rules)
11. [Box 2e/2f U.S. Individual Exemption](#11-box-2e2f-us-individual-exemption)
12. [WHFIT Trust Expense Reporting](#12-whfit-trust-expense-reporting)
13. [Account Number Requirements](#13-account-number-requirements)
14. [Section 1202 Epistemic Uncertainty](#14-section-1202-epistemic-uncertainty)
15. [Section 1223 Scope Limitation](#15-section-1223-scope-limitation)
16. [Reference Source Nodes](#16-reference-source-nodes)
17. [2nd TIN Checkbox Effects](#17-2nd-tin-checkbox-effects)
18. [Dividend-as-Interest Institution Classification](#18-dividend-as-interest-institution-classification)
19. [RIC Section 199A Pass-Through Rules](#19-ric-section-199a-pass-through-rules)

---

## 1. Default Reporting Rules

### Rule: DefaultDividendOnUncertainty

**Entities**
- Payment
- Potential Dividend
- 1099-DIV Filing Deadline

**Relationships**

* **Payment**
  —(classification_status)—> **Potential Dividend**

* **Payer**
  —(unable_to_determine_dividend_component_by)—> **1099-DIV Filing Deadline**

* **Rule: DefaultDividendOnUncertainty**
  —(when)—> **(Payment is Potential Dividend AND Dividend Component Undetermined by Filing Deadline)**

* **Rule: DefaultDividendOnUncertainty**
  —(requires_reporting_as)—> **Dividend (100% of Payment Amount)**

* **Dividend (per default rule)**
  —(reported_on)—> **Form 1099-DIV**

> This is the exact "if unsure → report entire payment as dividend" behavior.

---

## 2. Holding Period Rules & Windows

### Entities

* **Preferred Stock** (distinct security class)

### Shared Day-Counting Rule

* **Rule: HoldingPeriodDayCount**

* **Holding Period Calculation**
  —(includes_day)—> **Disposition Date**

* **Holding Period Calculation**
  —(excludes_day)—> **Acquisition Date**

* **Holding Period Calculation**
  —(excludes_days_when)—> **Risk of Loss Diminished**

### Window: Common Stock Qualified Dividend (121-Day)

* **Window: 121DayWindow_Common**
  —(length_days)—> **121**
  —(starts_offset_from)—> **Ex-Dividend Date**
  —(start_offset_days)—> **-60**

* **Threshold: CommonStockMinHolding**
  —(min_days_held)—> **61**
  —(applies_within)—> **Window: 121DayWindow_Common**

* **Rule: QualifiedDividend_Disqualify_Common**
  —(when)—> **DaysHeld < 61 within Window: 121DayWindow_Common**
  —(effect)—> **Dividend —(is_not)—> Qualified Dividend**

### Window: Preferred Stock Special Rule (181-Day)

* **Window: 181DayWindow_Preferred**
  —(length_days)—> **181**
  —(starts_offset_from)—> **Ex-Dividend Date**
  —(start_offset_days)—> **-90**

* **Threshold: PreferredMinHolding**
  —(min_days_held)—> **91**
  —(applies_within)—> **Window: 181DayWindow_Preferred**

* **Preferred Stock Dividend**
  —(paid_on_security_type)—> **Preferred Stock**

* **Rule: QualifiedDividend_Disqualify_Preferred**
  —(when)—> **(Dividend is Preferred Stock Dividend AND AttributablePeriodTotal > 366 days AND DaysHeld < 91 within 181DayWindow_Preferred)**
  —(effect)—> **Dividend —(is_not)—> Qualified Dividend**

* **Rule: PreferredDividend_UsesCommonRule_WhenAttributablePeriod < 367**
  —(when)—> **Preferred Dividend attributable_period_total < 367**
  —(effect)—> **Apply Threshold: CommonStockMinHolding**

### Window: Qualified REIT Dividend (91-Day)

* **Window: 91DayWindow_REIT**
  —(length_days)—> **91**
  —(starts_offset_from)—> **Ex-Dividend Date**
  —(start_offset_days)—> **-45**

* **Threshold: REITMinHolding**
  —(min_days_held)—> **46**
  *(text says "45 days or less is disqualified" ⇒ qualification requires ≥46)*

* **Rule: QualifiedREITDividend_Disqualify_HoldingPeriod**
  —(when)—> **DaysHeld ≤ 45 within Window: 91DayWindow_REIT**
  —(effect)—> **REIT Dividend —(is_not)—> Qualified REIT Dividend**

---

## 3. Constructive Receipt Timing

### Rule: Q4DeclaredPaidJan_TreatedAsDec31

**Relationships**

* **Dividend Declaration Month**
  —(allowed_values)—> **October | November | December**

* **Dividend**
  —(declared_in)—> **October/November/December**
  —(payable_to)—> **Shareholders of Record**
  —(record_date_in)—> **October/November/December**
  —(actually_paid_in)—> **January (following year)**

* **Rule: Q4DeclaredPaidJan_TreatedAsDec31**
  —(when)—> **(conditions above true)**
  —(treats_as_paid_on)—> **December 31 (tax year of declaration)**

* **Rule: Q4DeclaredPaidJan_TreatedAsDec31**
  —(governed_by)—> **IRC §852(b)(7)**

* **Rule: Q4DeclaredPaidJan_TreatedAsDec31**
  —(governed_by)—> **IRC §857(b)(9)**

* **RIC** —(subject_to)—> **IRC §852(b)(7)**

* **REIT** —(subject_to)—> **IRC §857(b)(9)**

---

## 4. Backup Withholding Cross-Year Reporting

### Rule: BackupWithholding_JanPaid_CrossYearReporting

**Relationships**

* **Dividend (Jan Payment)**
  —(reporting_year_on_1099DIV)—> **Prior Tax Year**  *(via constructive receipt rule)*

* **Backup Withholding**
  —(withheld_when)—> **Dividend Actually Paid (January)**
  —(deposited_in)—> **Year Withheld (January year)**

* **Backup Withholding**
  —(reported_on)—> **Form 945**
  —(form_year)—> **Year Withheld (January year)**

* **Backup Withholding Amount**
  —(also_reported_on)—> **Form 1099-DIV**
  —(1099DIV_year)—> **Prior Tax Year**  *(cross-year asymmetry)*

> This models the "withhold/deposit/report on 945 in Jan-year, but show withholding on prior-year 1099-DIV."

---

## 5. Section 1202 Statement Requirements

### Entity

* **Section 1202 Statement** (separate from 1099-DIV)

### Relationships

* **RIC**
  —(must_furnish)—> **Section 1202 Statement**
  —(when)—> **Any Box 2a capital gain distribution may qualify for §1202 exclusion**

* **Section 1202 Gain**
  —(reported_in)—> **Box 2c**

* **Section 1202 Statement**
  —(must_include_field)—> **Issuer Corporation Name**

* **Section 1202 Statement**
  —(must_include_field)—> **RIC Acquisition Date(s)**

* **Section 1202 Statement**
  —(must_include_field)—> **Date Sold**

* **Section 1202 Statement**
  —(must_include_field)—> **Recipient Share of Sales Price**

* **Section 1202 Statement**
  —(must_include_field)—> **Recipient Share of RIC Basis**

* **Section 1202 Statement**
  —(must_include_field)—> **Recipient §1202 Gain Amount**

* **Section 1202 Statement**
  —(must_include_field)—> **Exclusion Percentage**

---

## 6. IRC Section Attachments

### Bond Tax Credits / Interest Inclusion

* **Bond Tax Credit Allowed**
  —(included_in_gross_income_as)—> **Interest**
  —(governed_by)—> **IRC §54A(f)**
  —(governed_by)—> **IRC §54AA(f)(2)**

* **RIC**
  —(may_elect_to_distribute)—> **Bond Tax Credits**
  —(governed_by)—> **IRC §853A**

### PFIC & Surrogate Foreign Corporation Disqualification

* **Passive Foreign Investment Company (PFIC)**
  —(defined_by)—> **IRC §1297**

* **Surrogate Foreign Corporation**
  —(defined_by)—> **IRC §7874(a)(2)(B)**

* **Foreign Corporation**
  —(not_qualified_foreign_corporation_if)—> **(is Surrogate Foreign Corporation AND first_became_surrogate_after Dec 22, 2017 AND not_treated_as_domestic_under §7874(b))**

* **Foreign Corporation**
  —(treated_as_domestic_if)—> **IRC §7874(b)**

### Section 1202 Acquisition-Date Determination

* **QSBS 75%/100% Exclusion Acquisition Date**
  —(determined_after_application_of)—> **IRC §1223**

---

## 7. WHFIT Reporting Thresholds

* **WHFIT Dividend Income**
  —(reported_on_1099DIV_if)—> **Amount > $10**

* **Trustee**
  —(reports_WHfit_dividends_if)—> **Amount > $10**

* **Middleman**
  —(reports_WHfit_dividends_if)—> **Amount > $10**

---

## 8. TIN Notice & Compliance Rules

### Rule: SecondTINNoticeCounting

**Relationships**

* **B-Notice**
  —(is_a)—> **Second TIN Notification**

* **Payer**
  —(receives)—> **B-Notice**

* **Rule: SecondTINNoticeCounting**
  —(two_notices_in_3_years_triggers)—> **Escalated TIN Compliance Requirement**
  *(exact downstream effect depends on your larger spec, but the trigger condition is what matters here)*

### Nuanced Counting

* **Two B-Notices**
  —(count_as_one_notice_if)—> **(received_in_same_calendar_year OR relate_to_same_information_return_year)**

* **B-Notice Counting Window**
  —(length)—> **3 Years**

---

## 9. TIN Truncation Rules

### Rule: TINTruncation_Asymmetry

**Relationships**

* **Recipient Statement**
  —(may_display)—> **Truncated Payee TIN**

* **Payee TIN**
  —(may_be_truncated_on)—> **Recipient Statement**

* **Payer TIN**
  —(may_not_be_truncated_on)—> **Any Statement**

* **Rule: TINTruncation_Asymmetry**
  —(governed_by)—> **Treas. Reg. §301.6109-4**

---

## 10. Surrogate Foreign Corporation Rules

### Temporal Constraint

* **Surrogate Foreign Corporation Status**
  —(first_became_after)—> **December 22, 2017**

* **Foreign Corporation**
  —(disqualified_as_qualified_foreign_corporation_if)—> **(SurrogateForeignCorp AND first_became_after Dec 22, 2017 AND NOT treated_as_domestic_under §7874(b))**

* **Surrogate Foreign Corporation**
  —(defined_by)—> **IRC §7874(a)(2)(B)**

---

## 11. Box 2e/2f U.S. Individual Exemption

### Entities

* U.S. Individual (recipient classification)

### Relationships

* **Box 2e – Section 897 Ordinary Gain**
  —(completion_not_required_if_recipient_is)—> **U.S. Individual**

* **Box 2f – Section 897 Capital Gain**
  —(completion_not_required_if_recipient_is)—> **U.S. Individual**

* **Rule: Section897BoxesCompletionScope**
  —(applies_to)—> **RIC**
  —(applies_to)—> **REIT**
  —(limits_completion_to)—> **Recipients that are NOT U.S. Individuals**

> This captures both the "only RIC/REIT complete these boxes" and the U.S. individual exemption cleanly.

---

## 12. WHFIT Trust Expense Reporting

### Entities

* WHFIT Trust Expense
* Tax Information Statement
* Box 6 – Foreign Tax Paid

### Relationships

* **WHFIT Trust Expense**
  —(attributable_to)—> **Trust Income Holder (TIH)**

* **WHFIT Trust Expense**
  —(must_be_included_on)—> **Tax Information Statement**

* **WHFIT Trust Expense**
  —(not_required_to_be_included_in)—> **Box 6 – Foreign Tax Paid**

> Important: this is not "never in box 6"; it's specifically "not required" for this expense item — modeled as such.

---

## 13. Account Number Requirements

### Entities

* Account Number
* Recipient
* Box 11 – FATCA Filing Requirement
* Form 1099-DIV

### Relationships

**Trigger A: Multiple Accounts**

* **Account Number**
  —(required_when)—> **Multiple Accounts for Same Recipient**

**Trigger B: FATCA**

* **Account Number**
  —(required_when)—> **Box 11 Checked**

**Trigger C: General Encouragement**

* **Account Number**
  —(encouraged_for)—> **All Forms 1099-DIV**

### Normalized Rule Form

* **Rule: AccountNumberRequired_MultipleAccounts**
  —(when)—> **Recipient has Multiple Accounts**
  —(requires)—> **Account Number present**

* **Rule: AccountNumberRequired_FATCA**
  —(when)—> **Box 11 Checked**
  —(requires)—> **Account Number present**

* **Guideline: AccountNumberEncouraged**
  —(applies_to)—> **Form 1099-DIV**
  —(recommends)—> **Account Number present**

---

## 14. Section 1202 Epistemic Uncertainty

> This is subtle and matters for compliance reasoning: the RIC is making a "may qualify" determination based on incomplete downstream facts.

### Entities

* Recipient Holding Period (external / downstream fact)
* Section 1202 Statement
* Capital Gain Distribution (Box 2a)
* Section 1202 Exclusion Eligibility

### Relationships

* **Capital Gain Distribution**
  —(reported_in)—> **Box 2a**

* **Section 1202 Statement**
  —(required_when)—> **Capital Gain Distribution May Qualify for §1202 Exclusion**

* **May Qualify for §1202 Exclusion**
  —(qualification_depends_on)—> **Recipient Holding Period (external fact)**

* **RIC**
  —(cannot_deterministically_confirm)—> **Recipient Holding Period**
  —(therefore_uses)—> **May Qualify Determination**

### Epistemic Uncertainty Model

* **Rule: Section1202StatementTrigger_MayQualify**
  —(trigger_condition)—> **"may qualify" (not "does qualify")**
  —(depends_on_external_fact)—> **Recipient Holding Period**

---

## 15. Section 1223 Scope Limitation

> §1223 must not be modeled as applying generally to all exclusion tiers.

### Entities

* 75% Exclusion Tier
* 100% Exclusion Tier
* 50% Exclusion Tier
* 60% Exclusion Tier

### Relationships (Positive Scope)

* **IRC §1223**
  —(applies_to_acquisition_date_determination_for)—> **75% Exclusion Tier**

* **IRC §1223**
  —(applies_to_acquisition_date_determination_for)—> **100% Exclusion Tier**

### Relationships (Explicit Negative Scope)

* **IRC §1223**
  —(does_not_apply_to_acquisition_date_determination_for)—> **50% Exclusion Tier**

* **IRC §1223**
  —(does_not_apply_to_acquisition_date_determination_for)—> **60% Exclusion Tier**

> If you prefer monotonic-only ontologies, replace the "does_not_apply" edges with a constraint on the two apply-edges and omit the explicit negatives.

---

## 16. Reference Source Nodes

### General Instructions for Certain Information Returns

**Entity**

* General Instructions for Certain Information Returns

**Relationships**

* **Form 1099-DIV Instructions**
  —(incorporates_by_reference)—> **General Instructions for Certain Information Returns**

* **General Instructions for Certain Information Returns**
  —(covers_parts)—> **Part J | Part L | Part M | Part N**

> This improves traceability for procedural sections without bloating the core dividend ontology.

### TIN Matching System (Optional)

> Only worth modeling if integrating operational compliance workflows.

* **IRS TIN Matching System**
  —(provides)—> **TIN Validation Service**
  —(used_by)—> **Payer**

---

## 17. 2nd TIN Checkbox Effects

### Entities

* 2nd TIN Checkbox (form control on information return / account record context)
* IRS Notice Stream (or "IRS B-Notice Issuance" as process node)
* Account (or "Payee Account")

### Relationships

* **2nd TIN Checkbox**
  —(marked_for)—> **Account**

* **2nd TIN Checkbox (marked)**
  —(signals)—> **Second TIN Notification Received**

* **2nd TIN Checkbox (marked)**
  —(effect)—> **IRS Ceases Sending Further TIN Notices for Account**

### Rule Form

* **Rule: SecondTINCheckbox_StopNotices**
  —(when)—> **2nd TIN Checkbox is marked**
  —(effect)—> **IRS Notice Stream —(stops_for)—> Account**

---

## 18. Dividend-as-Interest Institution Classification

### Entities (Institution Types)

* Cooperative Bank
* Credit Union
* Domestic Building and Loan Association
* Domestic Savings and Loan Association
* Federal Savings and Loan Association
* Mutual Savings Bank

### Core Concept Node

* Dividend-Labeled Payment (the "so-called dividend" concept in the caution text)

### Relationships

* **Dividend-Labeled Payment**
  —(classified_as)—> **Interest**

* **Interest**
  —(reported_on)—> **Form 1099-INT**

### Source-Specific Conditional Issuer Typing

* **Dividend-Labeled Payment**
  —(classified_as_interest_if_from)—> **Cooperative Bank**

* **Dividend-Labeled Payment**
  —(classified_as_interest_if_from)—> **Credit Union**

* **Dividend-Labeled Payment**
  —(classified_as_interest_if_from)—> **Domestic Building and Loan Association**

* **Dividend-Labeled Payment**
  —(classified_as_interest_if_from)—> **Domestic Savings and Loan Association**

* **Dividend-Labeled Payment**
  —(classified_as_interest_if_from)—> **Federal Savings and Loan Association**

* **Dividend-Labeled Payment**
  —(classified_as_interest_if_from)—> **Mutual Savings Bank**

### Normalized Form (Less Edge Repetition)

* **IssuerCategory: DividendAsInterestInstitutions**
  —(includes)—> **(all six institution types above)**

* **Dividend-Labeled Payment**
  —(classified_as_interest_if_from_issuer_category)—> **IssuerCategory: DividendAsInterestInstitutions**

---

## 19. RIC Section 199A Pass-Through Rules

> This is the "RIC pays §199A dividends → certain shareholders meeting holding period requirements may treat as qualified REIT dividends for §199A" rule.

### Entities

* Section 199A Dividend (already defined)
* RIC Shareholder
* RIC Holding Period Requirement (constraint node; aligns to holding-period window framework)
* Qualified REIT Dividend (for §199A)

### Rule: Section199ADividend_RICShareholderHoldingPeriod

**Relationships**

* **Rule: Section199ADividend_RICShareholderHoldingPeriod**
  —(applies_when)—> **RIC Pays Section 199A Dividend**

* **RIC Shareholder**
  —(must_meet)—> **RIC Holding Period Requirement**

* **Rule: Section199ADividend_RICShareholderHoldingPeriod**
  —(if_recipient_meets)—> **RIC Holding Period Requirement**

* **Section 199A Dividend (paid by RIC)**
  —(may_be_treated_as)—> **Qualified REIT Dividend (for §199A)**

* **Rule: Section199ADividend_RICShareholderHoldingPeriod**
  —(enables_treatment_of)—> **Section 199A Dividend → Qualified REIT Dividend (for §199A)**

* **Rule: Section199ADividend_RICShareholderHoldingPeriod**
  —(governed_by)—> **Treas. Reg. §1.199A-3(d)**

### Limit Constraints

* **Section 199A Dividends (RIC may pay)**
  —(limited_to)—> **Qualified REIT Dividends Includible in RIC Taxable Income**
  —(reduced_by)—> **Properly Allocable Deductions**
  —(governed_by)—> **Treas. Reg. §1.199A-3(d)**
