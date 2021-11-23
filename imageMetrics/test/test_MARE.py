from numpy.testing._private.utils import assert_equal
import pytest
from imageMetrics import metrics
import numpy as np


class TestRelativeErrorImage:
    def testEqualImages(self):
        assert metrics.MARE(np.ones(5), np.ones(5), epsilon=0) == 0

    def testTrueBigger(self):
        assert np.array_equal(metrics.MARE(
            2*np.ones(5), np.ones(5), epsilon=0
        ), 0.5)

    def testPredictedBigger(self):
        assert np.array_equal(metrics.MARE(
            np.ones(5), 2*np.ones(5), epsilon=0
        ), 1)

    def testBothZero(self):
        assert np.array_equal(metrics.MARE(
            np.zeros(5), np.zeros(5)
        ), 0)        

    def testEpsilon(self):
        assert np.allclose(metrics.MARE(
            2*np.ones(5), np.ones(5)
        ), 0.5)

    def testMask(self):
        mask = np.ones(5)
        mask[0], mask[-1] = 0, 0
        target = .5
        assert np.allclose(metrics.MARE(
            2*np.ones(5), np.ones(5), mask
        ), target)

    def testMask2(self):
        mask = np.ones(5)
        mask[0], mask[-1] = 0, 0
        im_true = np.linspace(1,5,5)
        im_test = np.array([-4, -2, 1, 6, -300])
        MARE_val = metrics.MARE(
            im_true, im_test, mask
        )
        print(MARE_val)
        assert np.allclose(MARE_val, 19/18)