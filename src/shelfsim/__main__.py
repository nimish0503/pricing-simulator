"""Command line for the pipeline: python -m shelfsim <step> --client <client>."""

import argparse
import logging
import time

from shelfsim.config import load_client, load_modeling

STEPS = ("onboard", "cluster", "model", "backtest", "simulate", "report")


def main() -> None:
    parser = argparse.ArgumentParser(prog="shelfsim")
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--client", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for noisy in ("pymc", "pytensor", "arviz"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg, mcfg = load_client(args.client), load_modeling()
    started = time.perf_counter()
    match args.step:
        case "onboard":
            from shelfsim.onboarding import onboard

            onboard(cfg, mcfg)
        case "cluster":
            from shelfsim.segmentation import run

            run(cfg, mcfg)
        case "model":
            from shelfsim.demand import run

            run(cfg, mcfg)
        case "backtest":
            from shelfsim.evaluation import run

            run(cfg, mcfg)
        case "simulate":
            from shelfsim.choice import run

            run(cfg, mcfg)
        case "report":
            from shelfsim.report import run

            run(cfg, mcfg)
    logging.info(
        "%s for %s finished in %.0fs", args.step, args.client, time.perf_counter() - started
    )


if __name__ == "__main__":
    main()
