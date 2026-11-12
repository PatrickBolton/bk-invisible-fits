"""Locations of the project's data and output directories.

The data directory defaults to <project root>/data, which works when the package is used
from the repository (the default for `uv sync`, an editable install). Set the environment
variable LIGHTS_DATA_DIR to use data stored elsewhere, e.g. with a non-editable install.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get("LIGHTS_DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()
if not (DATA_ROOT / "BtoKEmiss").is_dir():
    raise FileNotFoundError(
        f"lights data not found in {DATA_ROOT} (expected the BtoKEmiss and form_factors "
        f"directories). Set the LIGHTS_DATA_DIR environment variable to the data directory."
    )

BTOKEMISS_DIR = DATA_ROOT / "BtoKEmiss"  # experimental inputs (Belle II, BaBar)
FORM_FACTOR_DIR = DATA_ROOT / "form_factors"  # form factor coefficients and covariances
PLOTS_DIR = PROJECT_ROOT / "plots"  # figure output
