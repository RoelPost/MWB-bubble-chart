"""
NOx Helper Functions and Constants

Reusable utilities for NOx emissions analysis including:
- Certification limits by stage and power band
- NOx per liter polynomial models
- TNO AUB coefficients
- Standard color palettes for plotting
"""

from typing import Optional, Callable
import numpy as np
import pandas as pd


# =============================================================================
# CERTIFICATION LIMITS - Power Bands and Stage Limits
# =============================================================================

POWER_BANDS = (
    ("lt_19", None, 19),
    ("19_37", 19, 37),
    ("37_56", 37, 56),
    ("56_75", 56, 75),
    ("75_130", 75, 130),
    ("130_560", 130, 560),
    ("gt_560", 560, None),
)

STAGE_LIMITS = {
    "stage-v": {
        "lt_19": 7.5,
        "19_37": 4.7,
        "37_56": 4.7,
        "56_75": 0.4,
        "75_130": 0.4,
        "130_560": 0.4,
        "gt_560": 3.5,
    },
    "stage-iv": {
        "56_75": 0.4,
        "75_130": 0.4,
        "130_560": 0.4,
    },
    "stage-iiib": {
        "37_56": 4.7,
        "56_75": 3.3,
        "75_130": 3.3,
        "130_560": 2.0,
    },
    "stage-iiia": {
        "lt_19": 7.5,
        "19_37": 4.7,
        "37_56": 4.7,
        "56_75": 4.0,
        "75_130": 4.0,
    },
    "stage-ii": {
        "lt_19": 8.0,
        "19_37": 7.0,
        "37_56": 7.0,
        "56_75": 6.0,
        "75_130": 6.0,
    },
}

STAGE_GROUP_POWER_BANDS = {
    "stage-ii": {
        "A": ["75_130"],
    },
    "stage-iiia": {
        "A": ["56_75"],
        "B": ["75_130"],
    },
    "stage-iiib": {
        "A": ["37_56", "56_75"],
        "B": ["75_130", "130_560"],
        "C": ["75_130", "130_560"],
    },
    "stage-iv": {
        "A": ["56_75"],
        "D": ["56_75", "75_130", "130_560"],
    },
    "stage-v": {
        "A": ["lt_19", "19_37", "37_56"],
        "D": ["56_75", "75_130", "130_560"],
        "B": ["gt_560"],
        "C": ["gt_560"],
    },
}


def power_band(power_kw: float) -> Optional[str]:
    """Map engine power to certification power band."""
    if pd.isna(power_kw):
        return None
    for band_name, lower, upper in POWER_BANDS:
        if lower is None and power_kw < upper:
            return band_name
        elif upper is None and power_kw >= lower:
            return band_name
        elif lower is not None and upper is not None and lower <= power_kw < upper:
            return band_name
    return None


def certification_limit(
    stage: str,
    power_kw: Optional[float] = None,
    *,
    machine_group: Optional[str] = None
) -> float:
    """
    Return NOx certification limit (g/kWh) x 4.

    Provide either power_kw (preferred, precise) or machine_group.

    Args:
        stage: Engine classification stage (e.g., "Stage-V", "Stage-IV")
        power_kw: Engine power in kW (preferred method)
        machine_group: Machine group letter (A, B, C, or D)

    Returns:
        Certification limit in g/kWh x 4

    Raises:
        ValueError: if inputs do not map to a unique certification limit.
    """
    if not stage or pd.isna(stage):
        raise ValueError("Stage is required to compute the certification limit.")

    # Normalise stage key: lowercase, replace spaces with hyphens
    stage_key = stage.strip().casefold().replace(" ", "-")
    limits = STAGE_LIMITS.get(stage_key)
    if not limits:
        raise ValueError(f"Unknown stage '{stage}'.")

    if power_kw is not None:
        band = power_band(power_kw)
        if band is None:
            raise ValueError(f"Power '{power_kw}' kW falls outside supported ranges.")

        base_limit = limits.get(band)
        if base_limit is None:
            raise ValueError(
                f"No certification limit configured for stage '{stage}' in band '{band}'."
            )
        return base_limit * 4

    if machine_group is None:
        raise ValueError("Provide either power_kw or machine_group to resolve the limit.")

    group_map = STAGE_GROUP_POWER_BANDS.get(stage_key)
    if not group_map:
        raise ValueError(
            f"Stage '{stage}' has no machine-group mappings; supply power_kw instead."
        )

    group_key = machine_group.strip().upper()
    bands = group_map.get(group_key)
    if not bands:
        raise ValueError(
            f"Machine group '{machine_group}' is not mapped for stage '{stage}'."
        )

    base_limits = []
    for band in bands:
        band_limit = limits.get(band)
        if band_limit is None:
            raise ValueError(
                f"Stage '{stage}' lacks a numeric limit for power band '{band}'."
            )
        base_limits.append(band_limit)

    return min(base_limits) * 4


def certification_limit_for_stage_group(stage_group: str) -> Optional[float]:
    """
    Return certification limit (g/kWh x 4) for a Stage+Groep label.

    Args:
        stage_group: Combined label like "Stage-V+D" or "Stage-IV+C"

    Returns:
        Certification limit or None if parsing fails
    """
    if not stage_group or "+" not in stage_group:
        return None

    stage_part, machine_group = [part.strip() for part in stage_group.split("+", 1)]
    if not stage_part or not machine_group:
        return None

    try:
        return certification_limit(stage_part, machine_group=machine_group)
    except ValueError:
        return None


# =============================================================================
# NOx POLYNOMIAL MODEL
# =============================================================================

STAGE_GROUP_UPPERBOUND = {
    "stage-ii": {"A": 18.75, "B": 140.0, "C": 110.0, "D": 220.0},
    "stage-iiia": {"A": 18.75, "B": 140.0, "C": 110.0, "D": 220.0},
    "stage-iiib": {"A": 14.0, "B": 140.0, "C": 110.0, "D": 220.0},
    "stage-iv": {"A": 12.0, "B": 140.0, "C": 110.0, "D": 220.0},
    "stage-v": {"A": 12.0, "B": 140.0, "C": 110.0, "D": 220.0},
}


def _clamp_adblue_pct(value: Optional[float], *, max_pct: float) -> float:
    """Clamp AdBlue percentage to reasonable range."""
    if value is None:
        return 6.0
    return max(3.0, min(value, max_pct))


def _machine_group_intercept(machine_group: str, adblue_pct: Optional[float]) -> float:
    """
    Calculate intercept for NOx polynomial model based on machine group and AdBlue %.

    Returns value in kg/L (will be converted to g/L in the polynomial function).
    """
    group = machine_group.strip().upper()
    adblue_is_set = adblue_pct is not None and not pd.isna(adblue_pct) and adblue_pct != 0

    if group == "A":
        if adblue_is_set:
            raise ValueError("AdBlue percentage applies only to machine groups C and D.")
        return 0.020
    elif group == "B":
        if adblue_is_set:
            raise ValueError("AdBlue percentage applies only to machine groups C and D.")
        return 0.015
    elif group == "C":
        capped_pct = _clamp_adblue_pct(adblue_pct, max_pct=4.0)
        return 0.025 - 0.46 * (capped_pct / 100)
    elif group == "D":
        capped_pct = _clamp_adblue_pct(adblue_pct, max_pct=7.0)
        return 0.033 - 0.46 * (capped_pct / 100)
    else:
        raise ValueError(f"Unsupported machine group: {group}")


def nox_per_liter_polynomial(
    stage: str,
    machine_group: str,
    *,
    adblue_pct: Optional[float] = None
) -> Callable[[float], float]:
    """
    Generate polynomial function for NOx/L based on motorbelasting.

    Returns a function that takes motorbelasting (x) and returns NOx/L (y) in kg/L.
    Model: y = intercept + 0.005 / (upperbound * motorbelasting)

    Note: Values are in kg/L. Multiply by 1000 to get g/L when plotting.

    Args:
        stage: Engine classification stage
        machine_group: Machine group letter (A, B, C, or D)
        adblue_pct: AdBlue percentage (only for groups C and D)

    Returns:
        Function that calculates NOx/L for a given motorbelasting value
    """
    if not stage or pd.isna(stage):
        raise ValueError("Stage is required to resolve the NOx/L polynomial.")
    if not machine_group:
        raise ValueError("Machine group is required to resolve the NOx/L polynomial.")

    # Normalise stage key: lowercase, replace spaces with hyphens
    stage_key = stage.strip().lower().replace(" ", "-")
    group_key = machine_group.strip().upper()

    stage_bounds = STAGE_GROUP_UPPERBOUND.get(stage_key)
    if not stage_bounds:
        raise ValueError(f"Stage '{stage}' has no configured NOx/L upperbound data.")
    upperbound = stage_bounds.get(group_key)
    if upperbound is None:
        raise ValueError(f"Machine group '{machine_group}' is not configured for stage '{stage}'.")

    intercept = _machine_group_intercept(group_key, adblue_pct)

    def polynomial(motorbelasting: float) -> float:
        if motorbelasting is None or pd.isna(motorbelasting):
            raise ValueError("Motorbelasting value is required.")
        if motorbelasting <= 0:
            raise ValueError("Motorbelasting must be greater than zero.")
        return intercept + 0.005 / (upperbound * motorbelasting)

    return polynomial


def model_fit_function(x, a, b, c):
    """
    Exponential decay model for curve fitting: y = a + b * exp(-c * x)

    Properties:
    - At x=0: y = a + b (maximum NOx/L)
    - As x→∞: y → a (horizontal asymptote)
    - Always positive when a, b, c > 0
    - Simple and easy to explain

    Args:
        x: Input value (motorbelasting, typically 0-1)
        a: Horizontal asymptote (NOx/L at high motorbelasting)
        b: Additional NOx at zero motorbelasting
        c: Decay rate

    Returns:
        Fitted y value (NOx per liter)
    """
    return a + b * np.exp(-c * x)


# =============================================================================
# TNO AUB COEFFICIENTS
# =============================================================================

TNO_AUB_COEFFICIENTS = {
    "A": {"Qb": 0.020, "Qu": 0.005, "Qa": None},      # Diesel with emission control (no SCR)
    "B": {"Qb": 0.015, "Qu": 0.005, "Qa": None},      # Diesel with emission hardware (no SCR)
    "C": {"Qb": 0.025, "Qu": 0.005, "Qa": -0.46},     # Diesel with SCR (NOx limit > 1 g/kWh)
    "D": {"Qb": 0.033, "Qu": 0.005, "Qa": -0.46},     # Diesel with advanced SCR (Stage IV/V)
}


def get_tno_coefficients(machine_group: str) -> Optional[dict]:
    """
    Get TNO AUB coefficients for a machine group.

    TNO AUB formula: NOx [kg] = Qb * liter_brandstof + Qu * draaiuren + Qa * liter_AdBlue
    Source: TNO 2021 R12305, Section 2.4 and Table 3

    Args:
        machine_group: Machine group letter (A, B, C, or D)

    Returns:
        Dictionary with Qb, Qu, Qa coefficients or None if not found
    """
    if not machine_group:
        return None
    return TNO_AUB_COEFFICIENTS.get(machine_group.strip().upper())


# =============================================================================
# COLOR PALETTES FOR PLOTTING
# =============================================================================

STAGE_GROUP_COLORS = {
    "Stage-V+D": "#2ecc40",
    "Stage-IV+D": "#1f77b4",
    "Stage-V+C": "#ff851b",
    "Stage-IV+C": "#b10dc9",
    "Stage-IIIB+C": "#e74c3c",
    "Stage-IIIB+B": "#9b59b6",
    "Stage-IIIA+A": "#3498db",
    "Stage-IIIA+B": "#1abc9c",
    "Stage-II+A": "#95a5a6",
}


def get_stage_group_color(stage_group: str, default_colors: list = None, index: int = 0) -> str:
    """
    Get consistent color for a Stage+Groep combination.

    Args:
        stage_group: Combined label like "Stage-V+D"
        default_colors: Fallback color list if stage_group not in palette
        index: Index into default_colors for fallback

    Returns:
        Hex color string
    """
    if stage_group in STAGE_GROUP_COLORS:
        return STAGE_GROUP_COLORS[stage_group]
    if default_colors:
        return default_colors[index % len(default_colors)]
    return "#333333"


# =============================================================================
# ADBLUE PERCENTAGES BY MACHINE GROUP
# =============================================================================

ADBLUE_PERCENTAGES = {
    "A": [0.0],       # No AdBlue (no SCR)
    "B": [0.0],       # No AdBlue (no SCR)
    "C": [3.0, 4.0],  # SCR with lower AdBlue consumption
    "D": [6.0, 7.0],  # SCR with higher AdBlue consumption
}


def get_adblue_percentages(machine_group: str) -> list:
    """
    Get typical AdBlue percentages for a machine group.

    Args:
        machine_group: Machine group letter (A, B, C, or D)

    Returns:
        List of AdBlue percentages to use for reference curves
    """
    if not machine_group:
        return []
    return ADBLUE_PERCENTAGES.get(machine_group.strip().upper(), [])
