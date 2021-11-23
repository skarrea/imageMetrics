import pytest
from imageMetrics import metrics
import numpy as np


class TestRelativeErrorImage:
    def testEqualImages(self):
        assert np.array_equal(metrics.RelativeErrorImage(
            np.ones(5), np.ones(5), epsilon=0
        ), np.zeros(5))

    def testTrueBigger(self):
        assert np.array_equal(metrics.RelativeErrorImage(
            2*np.ones(5), np.ones(5), epsilon=0
        ), -0.5*np.ones(5))

    def testPredictedBigger(self):
        assert np.array_equal(metrics.RelativeErrorImage(
            np.ones(5), 2*np.ones(5), epsilon=0
        ), np.ones(5))

    def testBothZero(self):
        assert np.array_equal(metrics.RelativeErrorImage(
            np.zeros(5), np.zeros(5)
        ), np.zeros(5))        

    def testEpsilon(self):
        assert np.allclose(metrics.RelativeErrorImage(
            2*np.ones(5), np.ones(5)
        ), -0.5*np.ones(5))

    def testMask(self):
        mask = np.ones(5)
        mask[0], mask[-1] = 0, 0
        target = -.5*np.ones(5)
        target[0], target[-1] = 0, 0
        assert np.allclose(metrics.RelativeErrorImage(
            2*np.ones(5), np.ones(5), mask
        ), target)
