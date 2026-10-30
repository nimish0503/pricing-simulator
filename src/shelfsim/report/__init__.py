"""make report: deploy export, then the findings report."""

from shelfsim.config import ClientConfig, ModelingConfig
from shelfsim.deploy import export
from shelfsim.report.build import build


def run(cfg: ClientConfig, mcfg: ModelingConfig) -> None:
    export(cfg)
    build(cfg, mcfg)
