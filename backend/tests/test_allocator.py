# test_allocator.py
# PyTest/Unittest hybrid test suite for a portfolio allocator.
# It validates: sum-to-1, box bounds, no-short (if lb>=0), sector caps,
# inverse-vol monotonicity, mean-variance sanity, and stability.
#
# How to run:
#   pytest -q test_allocator.py
#
# Plug in your own allocator by implementing functions on `Allocator`
# or swap the calls in the tests to your module.

from __future__ import annotations
import numpy as np
import unittest
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

# -----------------------------
# Test scaffolding / helpers
# -----------------------------
np.set_printoptions(precision=5, suppress=True)

@dataclass
class Bounds:
    lb: float = 0.0
    ub: float = 1.0

@dataclass
class SectorLimit:
    limits: Dict[int, float]  # sector_id -> max weight

def _normalize(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = float(w.sum())
    return w / (s if abs(s) > eps else 1.0)

def random_cov(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    Sigma = A @ A.T
    d = np.sqrt(np.diag(Sigma))
    Dinv = np.diag(1 / np.maximum(d, 1e-8))
    R = Dinv @ Sigma @ Dinv  # correlation-like
    vols = rng.uniform(0.1, 0.4, size=n)
    return np.diag(vols) @ R @ np.diag(vols)

def random_mu(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.05, 0.15, size=n)

def sector_map(n: int, k: int, seed: int = 2) -> List[int]:
    rng = np.random.default_rng(seed)
    return list(rng.integers(0, k, size=n))

# -----------------------------
# Minimal reference Allocator (replace with yours)
# -----------------------------
class Allocator:
    @staticmethod
    def equal_weight(n: int, bounds: Bounds = Bounds()) -> np.ndarray:
        w = np.ones(n) / n
        return np.clip(w, bounds.lb, bounds.ub) / np.clip(w, bounds.lb, bounds.ub).sum()

    @staticmethod
    def inverse_vol(Sigma: np.ndarray, bounds: Bounds = Bounds()) -> np.ndarray:
        vols = np.sqrt(np.clip(np.diag(Sigma), 1e-12, None))
        inv = 1.0 / vols
        w = inv / inv.sum()
        w = np.clip(w, bounds.lb, bounds.ub)
        return _normalize(w)

    @staticmethod
    def mean_variance(
        mu: np.ndarray,
        Sigma: np.ndarray,
        risk_aversion: float = 5.0,
        bounds: Bounds = Bounds(),
        sector_limit: Optional[SectorLimit] = None,
        sectors: Optional[Sequence[int]] = None,
        ridge: float = 1e-6,
        step: float = 0.01,
        iters: int = 1500,
    ) -> np.ndarray:
        n = len(mu)
        Q = risk_aversion * Sigma + ridge * np.eye(n)
        w = np.ones(n) / n

        def project(w: np.ndarray) -> np.ndarray:
            w = np.clip(w, bounds.lb, bounds.ub)
            if sector_limit and sectors is not None:
                caps = sector_limit.limits
                # scale any sector that is over the cap (couple passes)
                for _ in range(4):
                    fixed = True
                    for sid, cap in caps.items():
                        idx = [i for i, s in enumerate(sectors) if s == sid]
                        if not idx:
                            continue
                        tot = float(w[idx].sum())
                        if tot > cap + 1e-12:
                            w[idx] *= cap / tot
                            fixed = False
                    if fixed:
                        break
            w = _normalize(w)
            w = np.clip(w, bounds.lb, bounds.ub)
            return _normalize(w)

        for _ in range(iters):
            grad = Q @ w - mu
            w = project(w - step * grad)
        return w

# -----------------------------
# Tests
# -----------------------------
class TestAllocator(unittest.TestCase):
    def setUp(self):
        self.n = 20
        self.Sigma = random_cov(self.n, seed=42)
        self.mu = random_mu(self.n, seed=43)

    def test_equal_weight_bounds_and_sum(self):
        b = Bounds(0.0, 0.1)
        w = Allocator.equal_weight(self.n, b)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=7)
        self.assertTrue(np.all(w >= b.lb - 1e-12))
        self.assertTrue(np.all(w <= b.ub + 1e-12))
        self.assertAlmostEqual(float(w.max()), 1.0 / self.n, places=7)

    def test_inverse_vol_monotonicity(self):
        w = Allocator.inverse_vol(self.Sigma, Bounds(0, 1))
        vols = np.sqrt(np.diag(self.Sigma))
        i_min, i_max = int(np.argmin(vols)), int(np.argmax(vols))
        self.assertGreater(w[i_min], w[i_max])  # lower vol => higher weight
        self.assertAlmostEqual(float(w.sum()), 1.0, places=7)

    def test_mv_basic_sanity(self):
        w = Allocator.mean_variance(self.mu, self.Sigma, risk_aversion=5.0, bounds=Bounds(0, 1), ridge=1e-5)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=7)
        self.assertTrue(np.all(w >= -1e-10))
        hi, lo = int(np.argmax(self.mu)), int(np.argmin(self.mu))
        self.assertGreaterEqual(w[hi], w[lo])

    def test_box_constraints(self):
        b = Bounds(0.02, 0.12)
        w = Allocator.mean_variance(self.mu, self.Sigma, bounds=b, ridge=1e-4)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=7)
        self.assertTrue(np.all(w >= b.lb - 1e-9))
        self.assertTrue(np.all(w <= b.ub + 1e-9))

    def test_sector_caps(self):
        sectors = sector_map(self.n, k=4, seed=7)
        caps = {0: 0.35, 1: 0.40, 2: 0.30, 3: 0.50}
        w = Allocator.mean_variance(
            self.mu, self.Sigma,
            bounds=Bounds(0, 0.25),
            sector_limit=SectorLimit(caps),
            sectors=sectors,
            ridge=1e-5, step=0.02, iters=1500
        )
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)
        for sid, cap in caps.items():
            idx = [i for i, s in enumerate(sectors) if s == sid]
            if idx:
                self.assertLessEqual(float(w[idx].sum()), cap + 1e-6)

    def test_stability_small_perturbations(self):
        w0 = Allocator.mean_variance(self.mu, self.Sigma, ridge=1e-4)
        w1 = Allocator.mean_variance(self.mu + 1e-4, self.Sigma + np.eye(self.n) * 1e-4, ridge=1e-4)
        self.assertAlmostEqual(float(w1.sum()), 1.0, places=7)
        self.assertLess(float(np.linalg.norm(w1 - w0)), 0.25)

    def test_degenerate_sigma_ok_with_ridge(self):
        S = self.Sigma.copy()
        S[:, 0] = S[:, 1]
        S[0, :] = S[1, :]
        w = Allocator.mean_variance(self.mu, S, ridge=1e-3)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=7)

# -----------------------------
# PyTest entrypoint
# -----------------------------
def test_pytest_bridge():
    # Allows running with pytest without -k
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestAllocator)
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    assert result.wasSuccessful(), "Allocator tests failed"