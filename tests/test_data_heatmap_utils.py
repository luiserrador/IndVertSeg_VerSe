import unittest

from utils.data_heatmap_utils import *
from utils.data_utils import calc_dist_map, calc_centr_vertebras, normalize_vol_max_min


class MyTestCase(unittest.TestCase):
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
        sphere_data = np.logical_and(sphere_data < radius**2, sphere_data > 0, dtype=np.float).astype(np.float)
        sphere_data[15, 15, 15] = 1

        new_sphere = flip_vol(sphere_data, sphere_data)
        distance_border1 = calc_dist_map(new_sphere[0])[15, 34, 15]
        distance_border2 = calc_dist_map(new_sphere[1])[15, 34, 15]

        assert distance_border1 == distance_border2 == radius

    def testZoomZ(self):
        """Test zoom_z"""

        np.random.seed(1)

        radius = 10

        xx, yy, zz = np.mgrid[:64, :64, :128]
        sphere_data = (xx - 32) ** 2 + (yy - 32) ** 2 + (zz - 64) ** 2
        sphere_data = np.logical_and(sphere_data < radius**2, sphere_data > 0, dtype=np.float).astype(np.float)
        sphere_data[32, 32, 64] = 1

        new_sphere = zoom_z(sphere_data, sphere_data)

        new_coordinates_sphere = np.where(new_sphere[0] > 0.45)
        new_coordinates_sphere1 = np.where(new_sphere[1] > 0.45)
        new_height_sphere = np.max(new_coordinates_sphere[2]) - np.min(new_coordinates_sphere[2])
        new_width_sphere = np.max(new_coordinates_sphere[1]) - np.min(new_coordinates_sphere[1])
        new_depth_sphere = np.max(new_coordinates_sphere[0]) - np.min(new_coordinates_sphere[0])
        new_height_sphere1 = np.max(new_coordinates_sphere1[2]) - np.min(new_coordinates_sphere1[2])
        new_width_sphere1 = np.max(new_coordinates_sphere1[1]) - np.min(new_coordinates_sphere1[1])
        new_depth_sphere1 = np.max(new_coordinates_sphere1[0]) - np.min(new_coordinates_sphere1[0])

        assert new_height_sphere == new_height_sphere1 == 17
        assert new_width_sphere == new_width_sphere1 == 19
        assert new_depth_sphere == new_depth_sphere1 == 17

    def testRotate3D(self):
        """Test rotate3D"""

        np.random.seed(1)

        radius = 10

        xx, yy, zz = np.mgrid[:50, :50, :50]
        sphere_data = (xx - 15) ** 2 + (yy - 15) ** 2 + (zz - 15) ** 2
        sphere_data = np.logical_and(sphere_data < radius**2, sphere_data > 0, dtype=np.float).astype(np.float)
        sphere_data[15, 15, 15] = 1

        new_sphere, new_sphere1 = rotate3D(sphere_data, sphere_data)
        new_sphere[new_sphere > 0.45] = 1.0
        new_sphere1[new_sphere1 > 0.45] = 1.0
        centr_sphere = calc_centr_vertebras(new_sphere, 1)
        centr_sphere1 = calc_centr_vertebras(new_sphere1, 1)

        new_centr = [14, 16, 15]

        assert np.array_equal(centr_sphere, new_centr)
        assert np.array_equal(centr_sphere1, new_centr)


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

        xx, yy, zz = np.mgrid[:64, :64, :128]
        sphere_data = (xx - 32) ** 2 + (yy - 32) ** 2 + (zz - 64) ** 2
        sphere_data = np.logical_and(sphere_data < radius**2, sphere_data > (radius//2)**2).astype(np.float)

        new_data = roll_imgs(sphere_data, sphere_data)

    def TestAugmentatData(self):
        """Test augment_data"""

        np.random.seed(2**32-50)

        radius = 25

        xx, yy, zz = np.mgrid[:64, :64, :128]
        sphere_data = (xx - 32) ** 2 + (yy - 32) ** 2 + (zz - 64) ** 2
        sphere_data = np.logical_and(sphere_data < radius**2, sphere_data > (radius//2)**2).astype(np.float)

        new_data = augment_data(sphere_data, sphere_data)


if __name__ == '__main__':
    unittest.main()
