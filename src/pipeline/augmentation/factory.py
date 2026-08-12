from __future__ import annotations

from src.pipeline.augmentation.base import BaseAugmenter


def augmenter_factory(cfg: dict) -> BaseAugmenter | None:
    """Instancia o augmenter correto a partir do bloco 'augmentation' do YAML."""
    method = cfg.get("method", "none")
    percentile = cfg.get("extreme_percentile", 90.0)
    multiplier = cfg.get("multiplier", 3.0)

    if method == "none":
        return None

    if method == "extreme_gan":
        from src.pipeline.augmentation.extreme_gan_augmenter import (
            ExGANAugmenter,
        )
        return ExGANAugmenter(
            extreme_percentile=percentile,
            multiplier=multiplier,
            **cfg.get("extreme_gan", {}),
        )

    if method == "extreme_diffusion":
        from src.pipeline.augmentation.extreme_diffusion_augmenter import (
            ExtremeDiffusionAugmenter,
        )
        return ExtremeDiffusionAugmenter(
            extreme_percentile=percentile,
            multiplier=multiplier,
            **cfg.get("extreme_diffusion", {}),
        )

    raise ValueError(
        f"Augmenter desconhecido: '{method}'. "
        "Opcoes: 'extreme_gan', 'extreme_diffusion', 'none'."
    )
