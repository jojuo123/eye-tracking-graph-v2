from utils.registry import Registry

TRAINERS = Registry("trainers")


def build_trainer(full_cfg):
    """Build a trainer from the full experiment config. `full_cfg["trainer"]["type"]`
    selects which registered `BaseTrainer` subclass to instantiate; the whole
    `full_cfg` is then passed to its constructor (trainers pull out the pieces
    they need: model/data/optimizer/scheduler/trainer sections)."""
    trainer_type = full_cfg["trainer"]["type"]
    trainer_cls = TRAINERS.get(trainer_type)
    return trainer_cls(full_cfg)


from trainers.base_trainer import BaseTrainer  # noqa: E402
from trainers.example_trainer import ExampleTrainer  # noqa: E402

TRAINERS.register("base_trainer")(BaseTrainer)
TRAINERS.register("example_trainer")(ExampleTrainer)

__all__ = ["TRAINERS", "build_trainer", "BaseTrainer", "ExampleTrainer"]
