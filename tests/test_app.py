from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from shelfsim.deploy import DEPLOY_DIR

APP = Path(__file__).parents[1] / "app" / "streamlit_app.py"
PAGES = ["pages/overview.py", "pages/elasticities.py", "pages/simulator.py", "pages/optimizer.py"]

pytestmark = pytest.mark.skipif(
    not any(DEPLOY_DIR.glob("*/summary.json")), reason="no deploy outputs"
)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_errors(page):
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.switch_page(page).run()
    assert not app.exception, [e.message for e in app.exception]
