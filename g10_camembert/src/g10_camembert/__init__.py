"""
G10 — Fine-tuning CamemBERT sur Allociné
Protocole P02 : Régularisation & Généralisation

Auteurs : Groupe G10
Version : 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Groupe G10"
__description__ = "Fine-tuning CamemBERT — Régularisation & Généralisation (P02)"

from g10_camembert.utils.seed import set_seed

__all__ = ["set_seed", "__version__"]
