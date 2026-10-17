"""加权联合后验、官方兼容性及单例/批处理的一致性测试。"""
import numpy as np
import pytest

from substain_sustain.posterior import (
    infer_posterior, normalize_weighted_likelihood, sample_indices, summarize_joint,
)

pytestmark = pytest.mark.filterwarnings("ignore:Conversion of an array with ndim > 0 to a scalar:DeprecationWarning:pySuStaIn")


def test_weighted_joint_has_expected_prior():
    likelihood = np.array([[[0.8, 0.2], [0.2, 0.8]]])
    joint = normalize_weighted_likelihood(likelihood, np.array([0.9, 0.1]))
    np.testing.assert_allclose(joint, [[[0.72, 0.02], [0.18, 0.08]]])
    summary = summarize_joint(joint)
    np.testing.assert_allclose(summary["posterior_subtype"], [[0.9, 0.1]])
    assert summary["stage"][0] == 0
    assert summary["subtype"][0] == 0  # 报告层未分化；原始模型标签仍为 0-based。
    assert summary["stage_conditional_probability"][0] == pytest.approx(0.8)


def test_average_posteriors_is_not_normalized_average_likelihood():
    first = np.array([[[9.0], [1.0]]])
    second = np.array([[[0.1], [0.9]]])
    expected = (normalize_weighted_likelihood(first, [1]) + normalize_weighted_likelihood(second, [1])) / 2
    incorrect = normalize_weighted_likelihood((first + second) / 2, [1])
    np.testing.assert_allclose(expected, [[[0.5], [0.5]]])
    assert not np.allclose(expected, incorrect)


def test_zero_likelihood_is_rejected():
    with pytest.raises(FloatingPointError, match="均为零"):
        normalize_weighted_likelihood(np.zeros((1, 3, 2)), np.array([0.5, 0.5]))


def test_sample_selection_preserves_burnin_and_unique_draws():
    np.testing.assert_array_equal(sample_indices(10, 1000, 0.2), np.arange(2, 10))
    with pytest.raises(ValueError):
        sample_indices(10, 2, 1.0)


def test_real_official_engine_weighted_marginals_and_batching(tmp_path):
    pytest.importorskip("pySuStaIn")
    from pySuStaIn.ZscoreSustain import ZscoreSustain

    data = np.array([[0.0, 0.2, -0.2], [1.1, 0.2, 0.5], [2.0, 1.2, 0.8]])
    engine = ZscoreSustain(data, np.ones((3, 1)), np.repeat(3.0, 3), ["A", "B", "C"],
                          1, 2, 4, str(tmp_path), "test", False, seed=7)
    sequences = np.repeat(np.array([[[0], [1], [2]], [[2], [1], [0]]]), 4, axis=2)
    fractions = np.array([[0.9, 0.7, 0.8, 0.6], [0.1, 0.3, 0.2, 0.4]])
    combined = infer_posterior(engine, data, sequences, fractions, n_samples=4, burn_in_fraction=0, batch_size=3)
    batched = infer_posterior(engine, data, sequences, fractions, n_samples=4, burn_in_fraction=0, batch_size=2)
    single = infer_posterior(engine, data[:1], sequences, fractions, n_samples=4, burn_in_fraction=0, batch_size=1)
    np.testing.assert_allclose(combined["posterior_joint"], batched["posterior_joint"])
    np.testing.assert_allclose(combined["posterior_joint"][:1], single["posterior_joint"])
    np.testing.assert_allclose(combined["posterior_subtype"], combined["official_prob_subtype"])
    assert not np.allclose(combined["posterior_joint"], combined["official_prob_subtype_stage"])
