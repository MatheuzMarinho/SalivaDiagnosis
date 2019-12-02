from ga.ga_config import GaConfig
from pso.pso_config import PsoConfig


def build_config(typ, config_content):
    if typ == 1:
        return GaConfig(config_content)
    elif typ == 2:
        return PsoConfig(config_content)
