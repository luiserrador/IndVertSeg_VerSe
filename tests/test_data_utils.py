import unittest
from utils.data_utils import *


class MyTestCase(unittest.TestCase):
    def testCalcCentr(self):

        xx, yy, zz = np.mgrid[:50, :50, :50]
        circle_data = (xx - 25) ** 2 + (yy - 25) ** 2 + (zz - 25) ** 2
        circle_data = np.logical_and(circle_data < 25, circle_data > 0, dtype=np.float)

        cir_centr = calc_centr(np.where(circle_data == 1))

        assert np.array_equal(cir_centr, [25, 25, 25])

    def testCalcCentrVert(self):

        xx, yy, zz = np.mgrid[:50, :50, :50]
        circle_data = (xx - 25) ** 2 + (yy - 25) ** 2 + (zz - 25) ** 2
        circle_data = np.logical_and(circle_data < 25, circle_data > 0, dtype=np.float)

        cir_centr = calc_centr_vertebras(circle_data, 1)

        assert np.array_equal(cir_centr, [25, 25, 25])

    def testCalcDistMap(self):

        radius = 5

        xx, yy, zz = np.mgrid[:50, :50, :50]
        circle_data = (xx - 25) ** 2 + (yy - 25) ** 2 + (zz - 25) ** 2
        circle_data = np.logical_and(circle_data < radius**2, circle_data > 0, dtype=np.float)
        circle_data[25, 25, 25] = 1

        dist_map = calc_dist_map(circle_data)

        center_value = dist_map[25, 25, 25]

        assert center_value == 5.0

    def testNormalizeVolStd(self):

        data = (np.random.normal(size=[50, 50, 50]) * 5) + 30

        assert abs(np.mean(data.ravel())) > 25
        assert abs(np.std(data.ravel())) > 4

        new_data = normalize_vol_std(data)

        assert abs(np.mean(new_data.ravel())) < 1e-10
        assert abs(np.std(new_data.ravel())) - 1 < 1e-10

    def testNormalizeVolMaxMin(self):

        data = (np.random.normal(size=[50, 50, 50]) * 5) + 30
        new_data = normalize_vol_max_min(data, np.max(data), np.min(data))

        assert abs(np.mean(new_data.ravel())) < 0.05
        assert np.max(new_data) - 1 < 1e-4
        assert abs(np.min(new_data)) - 1 < 1e-4


if __name__ == '__main__':
    unittest.main()
