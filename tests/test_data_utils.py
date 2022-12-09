"""test_data_utils.py: Tests on data_utils.py."""

__author__ = "Luis Serrador"

import unittest

from utils.data_utils import *


class MyTestCase(unittest.TestCase):
    def testCalcCentr(self):
        """Test calc_centr"""

        xx, yy, zz = np.mgrid[:50, :50, :50]
        sphere_data = (xx - 25) ** 2 + (yy - 25) ** 2 + (zz - 25) ** 2
        sphere_data = np.logical_and(sphere_data < 25, sphere_data > 0, dtype=np.float)

        cir_centr = calc_centr(np.where(sphere_data == 1))

        assert np.array_equal(cir_centr, [25, 25, 25])

    def testCalcCentrVert(self):
        """Test calc_centr_vert"""

        xx, yy, zz = np.mgrid[:50, :50, :50]
        sphere_data = (xx - 25) ** 2 + (yy - 25) ** 2 + (zz - 25) ** 2
        sphere_data = np.logical_and(sphere_data < 25, sphere_data > 0, dtype=np.float)

        cir_centr = calc_centr_vertebras(sphere_data, 1)

        assert np.array_equal(cir_centr, [25, 25, 25])

    def testCalcDistMap(self):
        """Test calc_dist_map"""

        radius = 5

        xx, yy, zz = np.mgrid[:50, :50, :50]
        sphere_data = (xx - 25) ** 2 + (yy - 25) ** 2 + (zz - 25) ** 2
        sphere_data = np.logical_and(sphere_data < radius**2, sphere_data > 0, dtype=np.float)
        sphere_data[25, 25, 25] = 1

        dist_map = calc_dist_map(sphere_data)

        center_value = dist_map[25, 25, 25]

        assert center_value == 5.0

    def testNormalizeVolStd(self):
        """Test normalize_vol_std"""

        np.random.seed(1)

        data = (np.random.normal(size=[50, 50, 50]) * 5) + 30

        assert abs(np.mean(data.ravel())) > 25
        assert abs(np.std(data.ravel())) > 4

        new_data = normalize_vol_std(data)

        assert abs(np.mean(new_data.ravel())) < 1e-10
        assert abs(np.std(new_data.ravel())) - 1 < 1e-10

    def testNormalizeVolMaxMin(self):
        """Test normalize_vol_max_min"""

        np.random.seed(1)

        data = (np.random.normal(size=[50, 50, 50]) * 5) + 30
        new_data = normalize_vol_max_min(data, np.max(data), np.min(data))

        assert abs(np.mean(new_data.ravel())) < 0.1
        assert np.max(new_data) - 1 < 1e-4
        assert abs(np.min(new_data)) - 1 < 1e-4

    def testRandMulShiVox(self):
        """Test rand_mult_shi_vox"""

        np.random.seed(1)

        data = np.random.normal(size=[50, 50, 50])
        new_data = rand_mul_shi_vox(data)

        assert abs(np.mean(new_data.ravel())) > 0.05
        assert np.std(new_data.ravel()) < 0.8

    def testFlipVol(self):
        """Test flip_vol"""
        
        radius = 10

        xx, yy, zz = np.mgrid[:50, :50, :50]
        sphere_data = (xx - 15) ** 2 + (yy - 15) ** 2 + (zz - 15) ** 2
        sphere_data = np.logical_and(sphere_data < radius**2, sphere_data > 0, dtype=np.float)
        sphere_data[15, 15, 15] = 1
        
        new_sphere = flip_vol(sphere_data, sphere_data, sphere_data)
        distance_border1 = calc_dist_map(new_sphere[0])[15, 34, 15]
        distance_border2 = calc_dist_map(new_sphere[1])[15, 34, 15]
        distance_border3 = calc_dist_map(new_sphere[2])[15, 34, 15]

        assert distance_border1 == radius
        assert distance_border2 == radius
        assert distance_border3 == radius

    def testZoomZ(self):
        """Test zoom_z"""

        np.random.seed(1)

        radius = 20

        xx, yy, zz = np.mgrid[:50, :50, :50]
        sphere_data = (xx - 25) ** 2 + (yy - 25) ** 2 + (zz - 25) ** 2
        sphere_data = np.logical_and(sphere_data < radius**2, sphere_data > 0, dtype=np.float)
        sphere_data[25, 25, 25] = 1

        new_sphere = zoom_z(sphere_data, sphere_data, sphere_data)

        new_coordinates_sphere = np.where(new_sphere[0] > 0.0)
        new_coordinates_sphere1 = np.where(new_sphere[1] > 0.0)
        new_coordinates_sphere2 = np.where(new_sphere[2] > 0.0)
        new_height_sphere = np.max(new_coordinates_sphere[2]) - np.min(new_coordinates_sphere[2])
        new_width_sphere = np.max(new_coordinates_sphere[1]) - np.min(new_coordinates_sphere[1])
        new_depth_sphere = np.max(new_coordinates_sphere[0]) - np.min(new_coordinates_sphere[0])
        new_height_sphere1 = np.max(new_coordinates_sphere1[2]) - np.min(new_coordinates_sphere1[2])
        new_width_sphere1 = np.max(new_coordinates_sphere1[1]) - np.min(new_coordinates_sphere1[1])
        new_depth_sphere1 = np.max(new_coordinates_sphere1[0]) - np.min(new_coordinates_sphere1[0])
        new_height_sphere2 = np.max(new_coordinates_sphere2[2]) - np.min(new_coordinates_sphere2[2])
        new_width_sphere2 = np.max(new_coordinates_sphere2[1]) - np.min(new_coordinates_sphere2[1])
        new_depth_sphere2 = np.max(new_coordinates_sphere2[0]) - np.min(new_coordinates_sphere2[0])

        assert new_height_sphere == 25
        assert new_height_sphere1 == new_height_sphere2 == 26
        assert new_width_sphere == 40
        assert new_width_sphere1 == new_width_sphere2 == 42
        assert new_depth_sphere == new_depth_sphere1 == new_depth_sphere2 == 36

    def testRotate3D(self):
        """Test rotate3D"""

        np.random.seed(1)

        radius = 10

        xx, yy, zz = np.mgrid[:50, :50, :50]
        sphere_data = (xx - 15) ** 2 + (yy - 15) ** 2 + (zz - 15) ** 2
        sphere_data = np.logical_and(sphere_data < radius**2, sphere_data > 0, dtype=np.float)
        sphere_data[15, 15, 15] = 1

        new_sphere = rotate3D(sphere_data, sphere_data, sphere_data)

        centr_sphere = calc_centr_vertebras(new_sphere[0], 1)
        centr_sphere1 = calc_centr_vertebras(new_sphere[1], 1)
        centr_sphere2 = calc_centr_vertebras(new_sphere[2], 1)

        new_centr = [25, 28, 19]

        assert np.array_equal(centr_sphere, new_centr)

        new_centr = [25, 27, 19]

        assert np.array_equal(centr_sphere1, new_centr)
        assert np.array_equal(centr_sphere2, new_centr)

    def testGaussNoise(self):
        """Tets gauss_noise"""

        np.random.seed(30)

        radius = 10

        xx, yy, zz = np.mgrid[:50, :50, :50]
        sphere_data = (xx - 25) ** 2 + (yy - 25) ** 2 + (zz - 25) ** 2
        sphere_data = np.logical_and(sphere_data < radius**2, sphere_data > (radius//2)**2).astype(np.float)
        sphere_data = normalize_vol_max_min(sphere_data, np.max(sphere_data), np.min(sphere_data))
        new_data = gauss_noise(sphere_data)

    def testGaussBlur(self):
        """Tets gauss_blur"""

        np.random.seed(30)

        radius = 10

        xx, yy, zz = np.mgrid[:50, :50, :50]
        sphere_data = (xx - 25) ** 2 + (yy - 25) ** 2 + (zz - 25) ** 2
        sphere_data = np.logical_and(sphere_data < radius**2, sphere_data > (radius//2)**2).astype(np.float)
        sphere_data = normalize_vol_max_min(sphere_data, np.max(sphere_data), np.min(sphere_data))
        new_data = gauss_blur(sphere_data)

    def testRollImgs(self):
        """Tets clean_memory and roll_imgs"""

        np.random.seed(2**32-50)

        radius = 25

        xx, yy, zz = np.mgrid[:256, :256, :256]
        sphere_data = (xx - 128) ** 2 + (yy - 128) ** 2 + (zz - 128) ** 2
        sphere_data = np.logical_and(sphere_data < radius**2, sphere_data > (radius//2)**2).astype(np.float)
        memory = sphere_data
        nb_1s = len(np.where(sphere_data == 1)[0])

        new_data = roll_imgs(sphere_data, memory, sphere_data, [64, 64, 64], nb_1s)

        memory = clean_memory(memory)

        new_data = roll_imgs(sphere_data, memory, sphere_data, [64, 64, 64], nb_1s)


if __name__ == '__main__':
    unittest.main()
