"""Chargement de la configuration Python."""
from __future__ import annotations

import importlib.util
from types import SimpleNamespace


def load_config(path: str = "configs/config.py") -> SimpleNamespace:
    """Charge un fichier de configuration Python et retourne son objet CFG.

    Args:
        path: Chemin vers le fichier config .py.

    Returns:
        L'objet ``CFG`` défini dans le fichier de configuration.
    """
    spec = importlib.util.spec_from_file_location("_config", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.CFG
