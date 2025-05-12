from dataclasses import dataclass
from typing import Self, Any
from omegaconf import OmegaConf

@dataclass
class BaseConfig:
    @classmethod
    def create(cls, *args: Any) -> Self:
        schema = OmegaConf.structured(cls)
        config = OmegaConf.merge(schema, *args)
        result: Self = OmegaConf.to_object(config)  # type: ignore
        return result

    @classmethod
    def from_cli(cls) -> Self:
        return cls.create(OmegaConf.from_cli())
