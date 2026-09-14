"""Lightweight YAML config loading with attribute-style access."""

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


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(raw or {})
