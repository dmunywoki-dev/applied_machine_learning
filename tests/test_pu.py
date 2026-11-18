import numpy as np
from scipy import sparse

from mthl.pu import NnPULogistic


def test_nnpu_learner_fits_and_scores():
    rng = np.random.default_rng(7)
    positive = rng.normal(1.0, 0.4, size=(12, 4))
    unlabelled = rng.normal(0.0, 1.0, size=(60, 4))
    X = sparse.csr_matrix(np.vstack([positive, unlabelled]))
    y = np.r_[np.ones(len(positive), dtype=int), np.zeros(len(unlabelled), dtype=int)]
    model = NnPULogistic(positive_prior=0.15, max_iter=100).fit(X, y)
    scores = model.decision_function(X)
    assert scores.shape == (len(y),)
    assert np.isfinite(scores).all()
