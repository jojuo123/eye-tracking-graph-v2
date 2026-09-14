"""Generic name -> class registry used to build models, datasets, losses, etc. from config dicts.

Usage:
    MODELS = Registry("models")

    @MODELS.register("my_model")
    class MyModel(BaseModel):
        ...

    model = MODELS.build({"type": "my_model", "in_dim": 32})
"""


class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def __contains__(self, key):
        return key in self._module_dict

    def __repr__(self):
        return f"Registry(name={self._name!r}, items={list(self._module_dict)})"

    def register(self, name=None):
        """Decorator: `@registry.register()` or `@registry.register('custom_name')`."""

        def _register(cls_or_fn):
            key = name or cls_or_fn.__name__
            if key in self._module_dict:
                raise KeyError(f"'{key}' is already registered in '{self._name}' registry")
            self._module_dict[key] = cls_or_fn
            return cls_or_fn

        return _register

    def get(self, name):
        if name not in self._module_dict:
            raise KeyError(
                f"'{name}' not found in '{self._name}' registry. "
                f"Available: {sorted(self._module_dict.keys())}"
            )
        return self._module_dict[name]

    def build(self, cfg, **kwargs):
        """Build an instance from a config dict. `cfg['type']` selects the registered class,
        the remaining keys (plus any extra `kwargs`) are passed as constructor arguments."""
        if isinstance(cfg, str):
            cfg = {"type": cfg}
        cfg = dict(cfg)
        name = cfg.pop("type")
        cls_or_fn = self.get(name)
        cfg.update(kwargs)
        return cls_or_fn(**cfg)
