from .few_shot_engine import FewShotEngine
from .sweep_engine import SweepEngine
from .train_engine import Engine


def build_engine(engine_name: str):
    if engine_name == "engine":
        return Engine
    elif engine_name == "sweep_engine":
        return SweepEngine
    elif engine_name == "few_shot_engine":
        return FewShotEngine
    else:
        raise ValueError(f"Unknown engine: {engine_name}")
