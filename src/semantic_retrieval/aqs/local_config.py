"""Versioned spatial-query semantics, without scientific parameter defaults."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LocalQueryConfig:
    enabled: bool
    beta: float
    tau: float

    def __post_init__(self):
        if (type(self.enabled) is not bool
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in (self.beta, self.tau))
                or self.beta < 0 or self.tau <= 0 or self.tau < 1e-38):
            raise ValueError('invalid local query configuration')
        object.__setattr__(self, 'beta', float(self.beta))
        object.__setattr__(self, 'tau', float(self.tau))

    @property
    def active(self):
        return self.enabled and self.beta > 0

    def recipe(self):
        return dict(schema='uatvr-aqs-local-query-v1', enabled=self.enabled, beta=self.beta, tau=self.tau,
                    windows='grid7-window3-stride2-v1', queries=9, heads=8, layers=1,
                    dropout=0.0, ffn=False, aggregation='uniform-lexical', epsilon=1e-6,
                    text_source='clip_pre_global_adapter', dtype='fp32')

    @classmethod
    def from_recipe(cls, recipe):
        try:
            if not isinstance(recipe, dict): raise ValueError('invalid local recipe')
            result = cls(recipe['enabled'], recipe['beta'], recipe['tau'])
        except (KeyError, TypeError) as exc:
            raise ValueError('invalid local recipe') from exc
        expected = result.recipe()
        if (recipe != expected or any(type(recipe[k]) is not type(v) for k,v in expected.items())):
            raise ValueError('local recipe fields or constants mismatch')
        return result
