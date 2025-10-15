"""Tests for GO prior construction utilities."""

import numpy as np
import pandas as pd

from utils.go_prior import Go_Prior


def test_go_prior_builds_aspect_specific_graphs(tmp_path):
    obo_text = """
    [Term]
    id: GO:0000001
    name: bp root
    namespace: biological_process

    [Term]
    id: GO:0000002
    name: bp child one
    namespace: biological_process
    is_a: GO:0000001 ! bp root

    [Term]
    id: GO:0000003
    name: bp branch parent
    namespace: biological_process
    is_a: GO:0000001 ! bp root

    [Term]
    id: GO:0000008
    name: bp child two
    namespace: biological_process
    is_a: GO:0000003 ! bp branch parent
    relationship: part_of GO:0000001 ! bp root

    [Term]
    id: GO:0000004
    name: mf root
    namespace: molecular_function

    [Term]
    id: GO:0000005
    name: mf child
    namespace: molecular_function
    is_a: GO:0000004 ! mf root

    [Term]
    id: GO:0000006
    name: cc root
    namespace: cellular_component

    [Term]
    id: GO:0000007
    name: cc child
    namespace: cellular_component
    relationship: part_of GO:0000006 ! cc root
    """

    obo_file = tmp_path / "mock_go.obo"
    obo_file.write_text(obo_text.strip(), encoding="utf-8")

    df = pd.DataFrame(
        {
            "protein_id": ["P1", "P2"],
            "go_terms": [
                "['GO:0000002', 'GO:0000008', 'GO:0000005']",
                "GO:0000007;GO:0000002",
            ],
        }
    )
    csv_path = tmp_path / "train_split.csv"
    df.to_csv(csv_path, index=False)

    priors = Go_Prior(
        obo_path=obo_file,
        train_split_csv=csv_path,
        top_k_mf=1,
        top_k_bp=2,
        top_k_cc=1,
    )

    bp = priors["BP"]
    assert bp.terms == [
        "GO:0000001",
        "GO:0000002",
        "GO:0000003",
        "GO:0000008",
    ]
    assert bp.index_map["GO:0000003"] == 2
    assert np.allclose(bp.adjacency[1], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(
        bp.adjacency[3],
        np.array([0.5, 0.0, 0.5, 0.0], dtype=np.float32),
    )
    assert np.allclose(bp.adjacency[0], np.zeros(4, dtype=np.float32))

    mf = priors["MF"]
    assert mf.terms == ["GO:0000004", "GO:0000005"]
    expected_mf = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert np.allclose(mf.adjacency, expected_mf)

    cc = priors["CC"]
    assert cc.terms == ["GO:0000006", "GO:0000007"]
    expected_cc = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert np.allclose(cc.adjacency, expected_cc)
