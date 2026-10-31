from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from shelfsim.deploy import DEPLOY_DIR

APP = Path(__file__).parents[1] / "app" / "streamlit_app.py"
PAGES = [
    "views/home.py",
    "views/prices.py",
    "views/promos.py",
    "views/try_price.py",
    "views/simulator.py",
    "views/elasticities.py",
    "views/compare.py",
]

pytestmark = pytest.mark.skipif(
    not any(DEPLOY_DIR.glob("*/summary.json")), reason="no deploy outputs"
)


CLIENTS = sorted(p.parent.name for p in DEPLOY_DIR.glob("*/summary.json"))


@pytest.mark.parametrize("client", CLIENTS)
@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_errors(page, client):
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.session_state["client"] = client
    app.run()
    app.switch_page(page).run()
    assert not app.exception, [e.message for e in app.exception]
