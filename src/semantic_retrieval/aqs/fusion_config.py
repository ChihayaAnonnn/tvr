"""Versioned scoring semantics, separate from the immutable AQS recipe."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FusionConfig:
    enabled: bool
    eta: float

    def __post_init__(self):
        if (type(self.enabled) is not bool or type(self.eta) not in (int,float)
                or not math.isfinite(self.eta) or self.eta < 0):
            raise ValueError('invalid KL fusion configuration')
        object.__setattr__(self,'eta',float(self.eta))

    @property
    def active(self):
        return self.enabled and self.eta > 0

    def recipe(self):
        return dict(schema='uatvr-aqs-kl-v1',enabled=self.enabled,eta=self.eta,
            metric='symmetric_diagonal_gaussian_kl',normalization='per_dimension',
            parameterization='legacy_logstd_x2',logvar_min=-8.0,logvar_max=4.0,
            dtype='fp32',legacy_vib='sample_mean_logsigma_unmodified_v1')

    @classmethod
    def from_recipe(cls,recipe):
        try:
            if not isinstance(recipe,dict): raise ValueError('invalid fusion recipe')
            result=cls(recipe['enabled'],recipe['eta'])
        except (KeyError,TypeError) as exc:
            raise ValueError('invalid fusion recipe') from exc
        if recipe != result.recipe() or any(type(recipe[k]) is not type(v) for k,v in result.recipe().items()):
            raise ValueError('fusion recipe fields or constants mismatch')
        return result
