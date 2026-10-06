import numpy as np
import pytest

from eval.statistics import holm_correction, paired_comparison


def test_holm():
    assert np.allclose(holm_correction([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_paired_effect_and_ties():
    base = np.arange(1, 21, dtype=float)
    result = paired_comparison(base, base*0.9)
    assert result["rank_biserial_improvement"] == 1
    assert result["pvalue"] < 0.01
    assert paired_comparison(base, base)["pvalue"] == 1
    assert paired_comparison(base, base*1.1)["rank_biserial_improvement"] == -1
    with pytest.raises(ValueError):
        paired_comparison(base, base[:2])
