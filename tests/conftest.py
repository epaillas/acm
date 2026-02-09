import os
from pathlib import Path

import pytest
import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_URL = "https://github.com/epaillas/acm/releases/download/data_test_v1/"

# Add here the data files you want to download for the tests from DATA_URL
DATA_FILES = ["tpcf.npy", "tpcf.ckpt"]

print("conftest.py: Configuration des tests")

os.environ["ACM_TEST_DATA"] = str(DATA_DIR)


@pytest.fixture(scope="session", autouse=True)
def download_test_data():
    DATA_DIR.mkdir(exist_ok=True)
    for filename in DATA_FILES:
        pn_data = DATA_DIR / filename
        if not pn_data.exists():
            print(f"\nDownloading {DATA_URL}{filename}")
            response = requests.get(DATA_URL + filename)
            response.raise_for_status()
            pn_data.write_bytes(response.content)


@pytest.fixture(scope="session")
def PYTEST_data():
    return DATA_DIR
