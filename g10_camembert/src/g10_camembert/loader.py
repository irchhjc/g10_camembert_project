"""
Chargement et sous-échantillonnage équilibré du dataset Allociné.

Le sous-échantillonnage est nécessaire pour l'adaptation matérielle
(contraintes GPU/CPU). L'équilibre inter-classes garantit la validité
du F1-score macro comme métrique principale.
"""

import random
from typing import Any

from datasets import DatasetDict, load_dataset
from loguru import logger


def load_allocine() -> DatasetDict:
    """
    Charge le dataset Allociné depuis Hugging Face Hub.

    Returns:
        DatasetDict avec les splits train/validation/test.
    """
    logger.info("Chargement du dataset Allociné depuis HuggingFace Hub...")
    dataset = load_dataset("allocine")
    logger.info(
        f"Dataset chargé : train={len(dataset['train']):,} | "
        f"val={len(dataset['validation']):,} | "
        f"test={len(dataset['test']):,}"
    )
    return dataset


def balanced_subsample(
    dataset_split: Any,
    n_per_class: int,
    label_col: str = "label",
    seed: int = 42,
) -> list[dict]:
    """
    Retourne un sous-ensemble équilibré de n_per_class exemples par classe.

    L'équilibre est crucial pour que le F1-score macro soit informatif.
    Le sous-échantillonnage est stratifié : chaque classe est représentée
    de manière égale, évitant les biais de distribution.

    Args:
        dataset_split: Split HuggingFace à sous-échantillonner.
        n_per_class:   Nombre d'exemples par classe.
        label_col:     Nom de la colonne d'étiquettes.
        seed:          Graine aléatoire pour reproductibilité.

    Returns:
        Liste de dictionnaires mélangés aléatoirement.
    """
    random.seed(seed)

    # Groupement par classe
    by_class: dict[int, list] = {}
    for example in dataset_split:
        label = example[label_col]
        by_class.setdefault(label, []).append(example)

    # Sous-échantillonnage par classe
    result = []
    for label, examples in sorted(by_class.items()):
        n = min(n_per_class, len(examples))
        selected = random.sample(examples, n)
        result.extend(selected)
        logger.debug(f"  Classe {label} : {n} exemples sélectionnés / {len(examples)}")

    random.shuffle(result)
    logger.info(
        f"Sous-ensemble créé : {len(result)} exemples "
        f"({n_per_class}/classe × {len(by_class)} classes)"
    )
    return result


def prepare_splits(
    dataset: DatasetDict,
    n_train: int = 500,
    n_val: int = 150,
    n_test: int = 150,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Prépare les trois splits avec sous-échantillonnage équilibré.

    Args:
        dataset:  DatasetDict complet Allociné.
        n_train:  Exemples par classe pour l'entraînement.
        n_val:    Exemples par classe pour la validation.
        n_test:   Exemples par classe pour le test.
        seed:     Graine aléatoire.

    Returns:
        Tuple (train_samples, val_samples, test_samples).
    """
    logger.info("Préparation des splits sous-échantillonnés...")
    train = balanced_subsample(dataset["train"], n_train, seed=seed)
    val = balanced_subsample(dataset["validation"], n_val, seed=seed)
    test = balanced_subsample(dataset["test"], n_test, seed=seed)
    return train, val, test
