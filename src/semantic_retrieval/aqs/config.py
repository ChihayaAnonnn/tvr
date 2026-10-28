"""Strict, serializable AQS choices; no scientific hyperparameter defaults."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AqsConfig:
    enabled: bool
    negatives: int
    weight: float

    def __post_init__(self):
        if (type(self.enabled) is not bool or type(self.negatives) is not int
                or self.negatives < 1 or type(self.weight) not in (float, int)
                or not math.isfinite(self.weight) or self.weight < 0):
            raise ValueError('invalid AQS configuration')
        object.__setattr__(self, 'weight', float(self.weight))

    @property
    def active(self):
        return self.enabled and self.weight > 0

    def recipe(self):
        return dict(schema='uatvr-aqs-v1', enabled=self.enabled, negatives=self.negatives,
                    weight=self.weight, difficulty_epsilon=1e-8, selection_dtype='fp32',
                    tie_break='canonical_video_id', aux_direction='t2v',
                    batching='unique_video_all_captions_v1')

    @classmethod
    def from_recipe(cls, recipe):
        if not isinstance(recipe, dict):
            raise ValueError('invalid AQS recipe')
        try:
            result = cls(recipe['enabled'], recipe['negatives'], recipe['weight'])
        except (KeyError, TypeError) as exc:
            raise ValueError('invalid AQS recipe') from exc
        if recipe != result.recipe():
            raise ValueError('AQS recipe fields or constants mismatch')
        return result
