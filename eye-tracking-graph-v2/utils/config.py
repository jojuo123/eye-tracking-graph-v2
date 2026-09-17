"""Lightweight YAML config loading with attribute-style access."""

import os

import yaml


class Config(dict):
    """A dict that also allows attribute access (`cfg.model.type` == `cfg["model"]["type"]`).

    Nested dicts are recursively wrapped into `Config` as well.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict) and not isinstance(v, Config):
                self[k] = Config(v)
            elif isinstance(v, list):
                self[k] = [Config(item) if isinstance(item, dict) else item for item in v]

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as e:
            raise AttributeError(key) from e


def config_to_dict(cfg) -> dict:
    """Recursively convert a Config (or nested dict/list) to plain Python dicts/lists."""
    if isinstance(cfg, dict):
        return {k: config_to_dict(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [config_to_dict(v) for v in cfg]
    return cfg


def load_config(path: str) -> Config:
    """Loads a YAML config, merging in any files listed under a top-level `includes:` key
    first -- so a big config can be assembled from several smaller, independently-readable
    ones (e.g. one experiment folder's `data.yaml`, `model.yaml`, `trainer.yaml`, ...)
    instead of duplicating their content.

    `includes` is a list of paths, resolved relative to `path`'s own directory (or
    absolute), loaded and shallow-merged in order -- later includes override earlier ones,
    and the including file's own top-level keys override every include. Nested (an included
    file can itself have `includes`) and existing configs are unaffected, since none of them
    have this key.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    includes = raw.pop("includes", None)
    merged = {}
    if includes:
        base_dir = os.path.dirname(path)
        for include_path in includes:
            if not os.path.isabs(include_path):
                include_path = os.path.join(base_dir, include_path)
            merged.update(load_config(include_path))
    merged.update(raw)
    return Config(merged)
