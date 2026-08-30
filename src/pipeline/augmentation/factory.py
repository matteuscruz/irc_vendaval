from __future__ import annotations

from src.pipeline.augmentation.base import BaseAugmenter


def augmenter_factory(
    cfg: dict, checkpoint_dir: str | None = None,
) -> BaseAugmenter | None:
    """Instancia o augmenter correto a partir do bloco 'augmentation' do YAML.

    `checkpoint_dir`: default de checkpoint/resume pro augmenter (ver
    src/pipeline/augmentation/checkpoint.py) — usado só se o YAML não definir
    `checkpoint_dir` explicitamente no sub-bloco do método (extreme_gan/
    extreme_diffusion), para o caller (cluster_lstm.py) poder apontar pra um
    diretório específico do experimento sem precisar editar o YAML.
    """
    method = cfg.get("method", "none")
    percentile = cfg.get("extreme_percentile", 90.0)
    multiplier = cfg.get("multiplier", 3.0)

    if method == "none":
        return None

    if method == "extreme_gan":
        from src.pipeline.augmentation.extreme_gan_augmenter import (
            ExGANAugmenter,
        )
        extra = dict(cfg.get("extreme_gan", {}))
        extra.setdefault("checkpoint_dir", checkpoint_dir)
        return ExGANAugmenter(
            extreme_percentile=percentile,
            multiplier=multiplier,
            **extra,
        )

    if method == "extreme_diffusion":
        from src.pipeline.augmentation.extreme_diffusion_augmenter import (
            ExtremeDiffusionAugmenter,
        )
        extra = dict(cfg.get("extreme_diffusion", {}))
        extra.setdefault("checkpoint_dir", checkpoint_dir)
        return ExtremeDiffusionAugmenter(
            extreme_percentile=percentile,
            multiplier=multiplier,
            **extra,
        )

    raise ValueError(
        f"Augmenter desconhecido: '{method}'. "
        "Opcoes: 'extreme_gan', 'extreme_diffusion', 'none'."
    )
