"""Prediction explanations (SPEC section 27).

Every major prediction is decomposed into readable factors -- historical
traffic, rainfall, peak hour, weekday, season, location -- with the signed
contribution of each.

Two commitments shape the package. **Contributions are honest about what they
are**: when SHAP is available they are exact and additive, so the baseline plus
the contributions reconstructs the prediction; when it is not, they come from
NaN-occlusion, which ranks the inputs without decomposing anything, and every
rendering says which it is. And **nothing here claims causation**. The model
found associations in historical data; a factor that contributed to a
prediction did not necessarily bring the traffic about, and the vocabulary is
fixed so that a careless phrasing cannot creep in unnoticed.
"""

from urbansense.explainability.attribution import (
    DEFAULT_GLOBAL_SAMPLE,
    DEFAULT_PERMUTATION_REPEATS,
    AttributionMethod,
    Explanation,
    FactorAttributor,
    FactorContribution,
    GlobalImportance,
    global_importance,
)
from urbansense.explainability.factors import FactorGroups, load_factor_groups

__all__ = [
    "DEFAULT_GLOBAL_SAMPLE",
    "DEFAULT_PERMUTATION_REPEATS",
    "AttributionMethod",
    "Explanation",
    "FactorAttributor",
    "FactorContribution",
    "FactorGroups",
    "GlobalImportance",
    "global_importance",
    "load_factor_groups",
]
