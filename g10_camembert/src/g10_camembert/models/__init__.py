"""Module models : chargement CamemBERT configurable."""
from g10_camembert.models.camembert import get_device, load_camembert, load_tokenizer
__all__ = ["load_camembert", "load_tokenizer", "get_device"]
