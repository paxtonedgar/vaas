"""
Predicate Grammar: Rule Evaluation Expressions.

This module defines a minimal predicate grammar for rule conditions.
Predicates are compiled from string expressions and evaluated against FactSets.

Grammar:
    predicate := fact_ref COMPARATOR value
               | predicate AND predicate
               | predicate OR predicate
               | NOT predicate
               | atom_ref.field COMPARATOR value

    fact_ref := context.field | facts.category.field
    atom_ref := atom:atom_id
    COMPARATOR := == | != | > | >= | < | <=

Examples:
    "context.holding_days >= atom:holding:61_day.days"
    "facts.actor.is_us_person == True"
    "context.amount > 10 AND facts.issuer.is_pfic == False"
    "NOT facts.position.has_hedge"
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from vaas.core.atoms import ATOM_REGISTRY, SemanticAtom
from vaas.core.operators import FactSet


class Comparator(Enum):
    """Comparison operators for predicates."""
    EQ = "=="
    NE = "!="
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="


class BoolOperator(Enum):
    """Boolean operators for combining predicates."""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"


@dataclass
class PredicateAST:
    """Base class for predicate AST nodes."""
    pass


@dataclass
class ComparisonPredicate(PredicateAST):
    """A simple comparison: left COMPARATOR right."""
    left: str  # fact_ref or atom_ref.field
    comparator: Comparator
    right: Any  # value (could be str, int, float, bool)


@dataclass
class BooleanPredicate(PredicateAST):
    """Boolean combination of predicates."""
    operator: BoolOperator
    operands: List[PredicateAST]


@dataclass
class AtomReference:
    """Reference to an atom field: atom:atom_id.field."""
    atom_id: str
    field: str


@dataclass
class FactReference:
    """Reference to a fact: context.field or facts.category.field."""
    category: str  # "context" or fact category like "actor", "instrument"
    field: str


class PredicateParseError(Exception):
    """Error parsing predicate expression."""
    pass


class PredicateEvalError(Exception):
    """Error evaluating predicate."""
    pass


class PredicateComplexityError(Exception):
    """Error when predicate exceeds complexity limits."""
    pass


# =============================================================================
# COMPLEXITY LIMITS
# =============================================================================

# Maximum number of fact references in a single predicate
MAX_FACT_REFERENCES = 2

# Maximum nesting depth for boolean operators
MAX_NESTING_DEPTH = 1  # Only allow flat AND/OR, no nested (A AND (B OR C))


# =============================================================================
# PREDICATE PARSER
# =============================================================================

class PredicateParser:
    """
    Parses predicate strings into AST.

    Supports:
    - Simple comparisons: "context.x >= 10"
    - Atom references: "atom:holding:61_day.days"
    - Boolean operators: AND, OR, NOT
    - Parentheses for grouping
    """

    # Regex patterns
    COMPARATOR_PATTERN = re.compile(r"(==|!=|>=|<=|>|<)")
    ATOM_REF_PATTERN = re.compile(r"atom:([a-zA-Z0-9_:]+)\.([a-zA-Z_][a-zA-Z0-9_]*)")
    FACT_REF_PATTERN = re.compile(r"(context|facts)\.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)")
    BOOL_AND_PATTERN = re.compile(r"\bAND\b", re.IGNORECASE)
    BOOL_OR_PATTERN = re.compile(r"\bOR\b", re.IGNORECASE)
    BOOL_NOT_PATTERN = re.compile(r"\bNOT\b", re.IGNORECASE)

    def parse(self, expr: str) -> PredicateAST:
        """
        Parse a predicate expression into an AST.

        Args:
            expr: The predicate string to parse.

        Returns:
            A PredicateAST node representing the expression.

        Raises:
            PredicateParseError: If the expression is invalid.
        """
        expr = expr.strip()
        if not expr:
            raise PredicateParseError("Empty predicate expression")

        return self._parse_or(expr)

    def _parse_or(self, expr: str) -> PredicateAST:
        """Parse OR expressions (lowest precedence)."""
        # Split on OR (outside parentheses)
        parts = self._split_on_operator(expr, "OR")
        if len(parts) > 1:
            return BooleanPredicate(
                operator=BoolOperator.OR,
                operands=[self._parse_and(p.strip()) for p in parts]
            )
        return self._parse_and(expr)

    def _parse_and(self, expr: str) -> PredicateAST:
        """Parse AND expressions."""
        parts = self._split_on_operator(expr, "AND")
        if len(parts) > 1:
            return BooleanPredicate(
                operator=BoolOperator.AND,
                operands=[self._parse_not(p.strip()) for p in parts]
            )
        return self._parse_not(expr)

    def _parse_not(self, expr: str) -> PredicateAST:
        """Parse NOT expressions."""
        expr = expr.strip()
        if self.BOOL_NOT_PATTERN.match(expr):
            remainder = self.BOOL_NOT_PATTERN.sub("", expr, count=1).strip()
            return BooleanPredicate(
                operator=BoolOperator.NOT,
                operands=[self._parse_atom(remainder)]
            )
        return self._parse_atom(expr)

    def _parse_atom(self, expr: str) -> PredicateAST:
        """Parse atomic expressions (comparisons or parenthesized)."""
        expr = expr.strip()

        # Handle parentheses
        if expr.startswith("(") and expr.endswith(")"):
            return self._parse_or(expr[1:-1])

        # Parse comparison
        return self._parse_comparison(expr)

    def _parse_comparison(self, expr: str) -> ComparisonPredicate:
        """Parse a comparison expression: left COMPARATOR right."""
        match = self.COMPARATOR_PATTERN.search(expr)
        if not match:
            raise PredicateParseError(f"No comparator found in: {expr}")

        comparator_str = match.group(1)
        comparator = Comparator(comparator_str)

        left = expr[:match.start()].strip()
        right_str = expr[match.end():].strip()

        # Parse right side value
        right = self._parse_value(right_str)

        return ComparisonPredicate(
            left=left,
            comparator=comparator,
            right=right
        )

    def _parse_value(self, value_str: str) -> Any:
        """Parse a value string into a Python value."""
        value_str = value_str.strip()

        # Boolean
        if value_str.lower() == "true":
            return True
        if value_str.lower() == "false":
            return False

        # Integer
        try:
            return int(value_str)
        except ValueError:
            pass

        # Float
        try:
            return float(value_str)
        except ValueError:
            pass

        # String (quoted or unquoted)
        if value_str.startswith('"') and value_str.endswith('"'):
            return value_str[1:-1]
        if value_str.startswith("'") and value_str.endswith("'"):
            return value_str[1:-1]

        return value_str

    def _split_on_operator(self, expr: str, operator: str) -> List[str]:
        """Split expression on operator, respecting parentheses."""
        parts = []
        current = []
        paren_depth = 0
        i = 0
        pattern = re.compile(rf"\b{operator}\b", re.IGNORECASE)

        while i < len(expr):
            if expr[i] == "(":
                paren_depth += 1
                current.append(expr[i])
                i += 1
            elif expr[i] == ")":
                paren_depth -= 1
                current.append(expr[i])
                i += 1
            elif paren_depth == 0:
                match = pattern.match(expr[i:])
                if match:
                    parts.append("".join(current))
                    current = []
                    i += len(match.group())
                else:
                    current.append(expr[i])
                    i += 1
            else:
                current.append(expr[i])
                i += 1

        parts.append("".join(current))
        return [p for p in parts if p.strip()]


# =============================================================================
# PREDICATE EVALUATOR
# =============================================================================

class PredicateEvaluator:
    """
    Evaluates predicate AST against a FactSet.

    Resolves:
    - context.field → FactSet field values
    - atom:id.field → Atom registry lookups
    - Comparisons and boolean operations
    """

    def __init__(self, atom_registry=None):
        """
        Initialize evaluator.

        Args:
            atom_registry: Optional atom registry to use. Defaults to global.
        """
        self.atom_registry = atom_registry or ATOM_REGISTRY

    def evaluate(self, predicate: PredicateAST, facts: FactSet) -> bool:
        """
        Evaluate a predicate against a FactSet.

        Args:
            predicate: The predicate AST to evaluate.
            facts: The FactSet providing context values.

        Returns:
            True if the predicate is satisfied, False otherwise.

        Raises:
            PredicateEvalError: If evaluation fails.
        """
        if isinstance(predicate, ComparisonPredicate):
            return self._eval_comparison(predicate, facts)
        elif isinstance(predicate, BooleanPredicate):
            return self._eval_boolean(predicate, facts)
        else:
            raise PredicateEvalError(f"Unknown predicate type: {type(predicate)}")

    def _eval_comparison(self, pred: ComparisonPredicate, facts: FactSet) -> bool:
        """Evaluate a comparison predicate."""
        left_value = self._resolve_ref(pred.left, facts)
        right_value = pred.right

        # If right is also a reference, resolve it
        if isinstance(right_value, str) and (
            right_value.startswith("atom:") or
            right_value.startswith("context.") or
            right_value.startswith("facts.")
        ):
            right_value = self._resolve_ref(right_value, facts)

        return self._compare(left_value, pred.comparator, right_value)

    def _eval_boolean(self, pred: BooleanPredicate, facts: FactSet) -> bool:
        """Evaluate a boolean predicate."""
        if pred.operator == BoolOperator.AND:
            return all(self.evaluate(op, facts) for op in pred.operands)
        elif pred.operator == BoolOperator.OR:
            return any(self.evaluate(op, facts) for op in pred.operands)
        elif pred.operator == BoolOperator.NOT:
            if len(pred.operands) != 1:
                raise PredicateEvalError("NOT requires exactly one operand")
            return not self.evaluate(pred.operands[0], facts)
        else:
            raise PredicateEvalError(f"Unknown boolean operator: {pred.operator}")

    def _resolve_ref(self, ref: str, facts: FactSet) -> Any:
        """Resolve a reference string to a value."""
        ref = ref.strip()

        # Atom reference: atom:atom_id.field
        atom_match = PredicateParser.ATOM_REF_PATTERN.match(ref)
        if atom_match:
            atom_id = atom_match.group(1)
            field = atom_match.group(2)
            return self._get_atom_field(atom_id, field)

        # Context reference: context.field
        if ref.startswith("context."):
            field_path = ref[8:]  # Remove "context."
            return self._get_context_field(facts, field_path)

        # Facts reference: facts.category.field
        if ref.startswith("facts."):
            path = ref[6:]  # Remove "facts."
            return self._get_facts_field(facts, path)

        raise PredicateEvalError(f"Cannot resolve reference: {ref}")

    def _get_atom_field(self, atom_id: str, field: str) -> Any:
        """Get a field value from an atom."""
        atom = self.atom_registry.get_optional(atom_id)
        if atom is None:
            raise PredicateEvalError(f"Atom not found: {atom_id}")

        if not hasattr(atom, field):
            raise PredicateEvalError(f"Atom '{atom_id}' has no field '{field}'")

        return getattr(atom, field)

    def _get_context_field(self, facts: FactSet, field_path: str) -> Any:
        """Get a context field from the FactSet."""
        # Try holding facts first (most common for context)
        if facts.holding:
            if hasattr(facts.holding, field_path):
                return getattr(facts.holding, field_path)
            if field_path == "holding_days":
                return facts.effective_holding_days()

        # Try temporal facts
        if facts.temporal and hasattr(facts.temporal, field_path):
            return getattr(facts.temporal, field_path)

        # Try amount facts
        if facts.amount and hasattr(facts.amount, field_path):
            return getattr(facts.amount, field_path)

        raise PredicateEvalError(f"Context field not found: {field_path}")

    def _get_facts_field(self, facts: FactSet, path: str) -> Any:
        """Get a facts field from the FactSet."""
        parts = path.split(".", 1)
        category = parts[0]
        field = parts[1] if len(parts) > 1 else None

        # Map category to FactSet attribute
        category_map = {
            "actor": facts.actor,
            "instrument": facts.instrument,
            "holding": facts.holding,
            "amount": facts.amount,
            "temporal": facts.temporal,
        }

        obj = category_map.get(category)
        if obj is None:
            raise PredicateEvalError(f"Facts category not found: {category}")

        if field is None:
            return obj

        if not hasattr(obj, field):
            raise PredicateEvalError(f"Facts.{category} has no field '{field}'")

        return getattr(obj, field)

    def _compare(self, left: Any, comparator: Comparator, right: Any) -> bool:
        """Perform comparison operation."""
        try:
            if comparator == Comparator.EQ:
                return left == right
            elif comparator == Comparator.NE:
                return left != right
            elif comparator == Comparator.GT:
                return left > right
            elif comparator == Comparator.GE:
                return left >= right
            elif comparator == Comparator.LT:
                return left < right
            elif comparator == Comparator.LE:
                return left <= right
            else:
                raise PredicateEvalError(f"Unknown comparator: {comparator}")
        except TypeError as e:
            raise PredicateEvalError(f"Cannot compare {left} and {right}: {e}")


# =============================================================================
# PREDICATE COMPILER
# =============================================================================

class CompiledPredicate:
    """
    A compiled predicate ready for evaluation.

    Caches the parsed AST for efficient repeated evaluation.
    Enforces complexity limits to prevent predicates from becoming shadow rules.
    """

    def __init__(
        self,
        expression: str,
        parser: PredicateParser = None,
        evaluator: PredicateEvaluator = None,
        enforce_limits: bool = True,
    ):
        """
        Compile a predicate expression.

        Args:
            expression: The predicate string to compile.
            parser: Optional parser to use.
            evaluator: Optional evaluator to use.
            enforce_limits: If True, validate complexity limits.

        Raises:
            PredicateComplexityError: If predicate exceeds complexity limits.
        """
        self.expression = expression
        self._parser = parser or PredicateParser()
        self._evaluator = evaluator or PredicateEvaluator()
        self._ast = self._parser.parse(expression)

        if enforce_limits:
            self._validate_complexity()

    def _validate_complexity(self) -> None:
        """
        Validate predicate doesn't exceed complexity limits.

        Raises:
            PredicateComplexityError: If limits exceeded.
        """
        # Check fact reference count
        fact_count = self._count_fact_references(self._ast)
        if fact_count > MAX_FACT_REFERENCES:
            raise PredicateComplexityError(
                f"Predicate references {fact_count} facts (max {MAX_FACT_REFERENCES}). "
                f"Complex predicates should be decomposed into rules."
            )

        # Check nesting depth
        depth = self._measure_nesting_depth(self._ast)
        if depth > MAX_NESTING_DEPTH:
            raise PredicateComplexityError(
                f"Predicate has nesting depth {depth} (max {MAX_NESTING_DEPTH}). "
                f"Nested boolean logic should be decomposed into separate predicates."
            )

    def _count_fact_references(self, node: PredicateAST) -> int:
        """Count total fact/context references in predicate."""
        if isinstance(node, ComparisonPredicate):
            count = 0
            # Left side is always a reference
            if node.left.startswith("context.") or node.left.startswith("facts."):
                count += 1
            # Right side might be a reference too
            if isinstance(node.right, str) and (
                node.right.startswith("context.") or
                node.right.startswith("facts.") or
                node.right.startswith("atom:")
            ):
                count += 1
            return count
        elif isinstance(node, BooleanPredicate):
            return sum(self._count_fact_references(op) for op in node.operands)
        return 0

    def _measure_nesting_depth(self, node: PredicateAST, current_depth: int = 0) -> int:
        """Measure maximum nesting depth of boolean operators."""
        if isinstance(node, ComparisonPredicate):
            return current_depth
        elif isinstance(node, BooleanPredicate):
            if node.operator in (BoolOperator.AND, BoolOperator.OR):
                child_depths = [
                    self._measure_nesting_depth(op, current_depth + 1)
                    for op in node.operands
                ]
                return max(child_depths) if child_depths else current_depth
            else:  # NOT doesn't increase nesting
                return max(
                    self._measure_nesting_depth(op, current_depth)
                    for op in node.operands
                )
        return current_depth

    def evaluate(self, facts: FactSet) -> bool:
        """
        Evaluate this predicate against a FactSet.

        Args:
            facts: The FactSet to evaluate against.

        Returns:
            True if predicate is satisfied.
        """
        return self._evaluator.evaluate(self._ast, facts)

    @property
    def ast(self) -> PredicateAST:
        """Get the parsed AST."""
        return self._ast

    @property
    def fact_count(self) -> int:
        """Get number of fact references."""
        return self._count_fact_references(self._ast)

    @property
    def nesting_depth(self) -> int:
        """Get nesting depth."""
        return self._measure_nesting_depth(self._ast)


def compile_predicate(expression: str, enforce_limits: bool = True) -> CompiledPredicate:
    """
    Compile a predicate expression.

    Args:
        expression: The predicate string to compile.
        enforce_limits: If True, validate complexity limits.

    Returns:
        A CompiledPredicate ready for evaluation.

    Raises:
        PredicateComplexityError: If predicate exceeds complexity limits.
    """
    return CompiledPredicate(expression, enforce_limits=enforce_limits)


def evaluate_predicate(expression: str, facts: FactSet) -> bool:
    """
    Parse and evaluate a predicate in one step.

    For repeated evaluation, use compile_predicate() instead.

    Args:
        expression: The predicate string.
        facts: The FactSet to evaluate against.

    Returns:
        True if predicate is satisfied.
    """
    return compile_predicate(expression).evaluate(facts)
