"""
Model catalog for adaptive-quant.

Provides curated sensitivity profiles built from community data
(steampunque, Unsloth KLD, Kaitchup).
"""

import json
import os

from ._registry import REGISTRY

_PROFILES_DIR = os.path.join(os.path.dirname(__file__), "profiles")


def list_models():
    """Return metadata for all cataloged models."""
    return list(REGISTRY)


def get_profile(model_name):
    """
    Load a full sensitivity profile by model name.

    Raises KeyError if model not found.
    """
    for entry in REGISTRY:
        if entry["name"].lower() == model_name.lower():
            profile_path = os.path.join(_PROFILES_DIR, entry["file"])
            with open(profile_path) as f:
                return json.load(f)

    available = [e["name"] for e in REGISTRY]
    raise KeyError(
        f"Model '{model_name}' not found in catalog. "
        f"Available: {', '.join(available)}"
    )


def search(query):
    """Fuzzy search by model name, family, or architecture."""
    query_lower = query.lower()
    results = []
    for entry in REGISTRY:
        searchable = f"{entry['name']} {entry['family']} {entry['architecture']} {entry['description']}"
        if query_lower in searchable.lower():
            results.append(entry)
    return results


def model_exists(model_name):
    """Check if a model is in the catalog."""
    return any(e["name"].lower() == model_name.lower() for e in REGISTRY)
