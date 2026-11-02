from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/corpus"))
from validate import Catalog


@pytest.fixture(scope="session")
def catalog():
    return Catalog().validate()
