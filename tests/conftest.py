import os
from pathlib import Path
import runpy
import pytest


@pytest.fixture(scope="session", autouse=True)
def ensure_example_data():
    project_root = Path(__file__).resolve().parents[1]
    examples = project_root / 'examples'

    experimental_pkl = examples / 'experimental.pkl'
    lookup_pkl = examples / 'lookup.pkl'
    distances_csv = examples / 'distances.csv'

    if not (experimental_pkl.exists() and lookup_pkl.exists() and distances_csv.exists()):
        cwd = os.getcwd()
        os.chdir(project_root)
        try:
            runpy.run_path(str(examples / 'prepare_data.py'), run_name='__main__')
        finally:
            os.chdir(cwd)


