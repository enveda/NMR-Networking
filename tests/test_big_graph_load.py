from pathlib import Path
import pytest

from nmr_networking import GraphBuilder


def test_load_big_graph_or_skip():
    path = Path('/home/cailum.stienstra/HSQC_Models/Networking_HSQC/demo/storage/MolecularNetwork_Paper_COCONUT_LOTUS.pkl')
    if not path.exists():
        pytest.skip('Big paper graph not present on this machine')

    b = GraphBuilder()
    b.load_graph(str(path))
    assert b.graph is not None
    # Ensure graph has a reasonable size
    assert b.graph.number_of_nodes() > 1000
    assert b.graph.number_of_edges() > 1000