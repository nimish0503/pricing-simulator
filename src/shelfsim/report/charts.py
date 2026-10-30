"""Matplotlib figures embedded in the HTML reports as base64 PNGs."""

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2f5d8a", "#c2662d", "#4f8a4f", "#8a4f7d", "#b0a03a", "#3a9bb0", "#7d7d7d", "#a33b3b"]


def to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def figure(width: float = 7, height: float = 4.2):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax
