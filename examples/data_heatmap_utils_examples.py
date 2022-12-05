"""data_heatmap_utils_examples.py: Examples for data_heatmap_utils.py functions."""

__author__ = "Luis Serrador"

from utils.data_heatmap_utils import *
import matplotlib.pyplot as plt


def rand_mul_shi_vox_example():
    """rand_mul_shi_vox example"""

    np.random.seed(1)

    data = np.random.normal(size=[50, 50, 50])
    new_data = rand_mul_shi_vox(data)

    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1)
    ax.hist(data.ravel(), bins=800)
    ax.set_title('Original data')
    ax.text(-4, 599, 'Mean: {}'.format(np.around(np.mean(data.ravel()), decimals=2)))
    ax.text(-4, 579, 'Std: {}'.format(np.around(np.std(data.ravel()), decimals=2)))
    ax.text(-4, 559, 'Max: {}'.format(np.around(np.max(data.ravel()), decimals=2)))
    ax.text(-4, 539, 'Min: {}'.format(np.around(np.min(data.ravel()), decimals=2)))

    ax1 = fig.add_subplot(1, 2, 2)
    ax1.hist(new_data.ravel(), bins=800)
    ax1.set_title('Randomly multiplied and shifted data')
    ax1.text(-3, 599, 'Mean: {}'.format(np.around(np.mean(new_data.ravel()), decimals=2)))
    ax1.text(-3, 579, 'Std: {}'.format(np.around(np.std(new_data.ravel()), decimals=2)))
    ax1.text(-3, 559, 'Max: {}'.format(np.around(np.max(new_data.ravel()), decimals=2)))
    ax1.text(-3, 539, 'Min: {}'.format(np.around(np.min(new_data.ravel()), decimals=2)))

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()


def flip_vol_example():
    """flip_vol example"""
    radius = 10

    xx, yy, zz = np.mgrid[:50, :50, :50]
    sphere_data = (xx - 15) ** 2 + (yy - 15) ** 2 + (zz - 15) ** 2
    sphere_data = np.logical_and(sphere_data < radius ** 2, sphere_data > 0, dtype=np.float)
    sphere_data[15, 15, 15] = 1

    new_sphere = flip_vol(sphere_data, sphere_data)

    x, y, z = np.where(sphere_data == 1)
    new_x, new_y, new_z = np.where(new_sphere[0] == 1)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(x, y, z, color='red')
    ax.axes.set_xlim3d(0, 50)
    ax.axes.set_ylim3d(0, 50)
    ax.axes.set_zlim3d(0, 50)
    ax.set_title('Original sphere')
    ax.view_init(elev=180, azim=0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.scatter(new_x, new_y, new_z, color='red')
    ax1.axes.set_xlim3d(0, 50)
    ax1.axes.set_ylim3d(0, 50)
    ax1.axes.set_zlim3d(0, 50)
    ax1.set_title('Fliped volume')
    ax1.view_init(elev=180, azim=0)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()


def zoom_z_example():
    """zoom_z example"""

    np.random.seed(1)

    radius = 20

    xx, yy, zz = np.mgrid[:64, :64, :128]
    sphere_data = (xx - 25) ** 2 + (yy - 25) ** 2 + (zz - 25) ** 2
    sphere_data = np.logical_and(sphere_data < radius**2, sphere_data > 0).astype(np.float)
    sphere_data[25, 25, 25] = 1

    coordinates_sphere = np.where(sphere_data > 0.0)
    height_sphere = np.max(coordinates_sphere[2]) - np.min(coordinates_sphere[2])
    width_sphere = np.max(coordinates_sphere[1]) - np.min(coordinates_sphere[1])
    depth_sphere = np.max(coordinates_sphere[0]) - np.min(coordinates_sphere[0])

    new_sphere = zoom_z(sphere_data, sphere_data)

    x, y, z = np.where(sphere_data > 0.45)
    new_x, new_y, new_z = np.where(new_sphere[0] > 0.45)

    new_coordinates_sphere = np.where(new_sphere[0] > 0.45)
    new_height_sphere = np.max(new_coordinates_sphere[2]) - np.min(new_coordinates_sphere[2])
    new_width_sphere = np.max(new_coordinates_sphere[1]) - np.min(new_coordinates_sphere[1])
    new_depth_sphere = np.max(new_coordinates_sphere[0]) - np.min(new_coordinates_sphere[0])

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(x, y, z, color='red')
    ax.axes.set_xlim3d(0, 64)
    ax.axes.set_ylim3d(0, 64)
    ax.axes.set_zlim3d(0, 128)
    ax.set_title('Original sphere')
    ax.view_init(elev=180, azim=0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.text(0, 50, 10, 'Height: {}'.format(height_sphere))
    ax.text(0, 50, 15, 'Width:  {}'.format(width_sphere))
    ax.text(0, 50, 20, 'Depth:  {}'.format(depth_sphere))

    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.scatter(new_x, new_y, new_z, color='red')
    ax1.axes.set_xlim3d(0, 64)
    ax1.axes.set_ylim3d(0, 64)
    ax1.axes.set_zlim3d(0, 128)
    ax1.set_title('Zoomed volume')
    ax1.view_init(elev=180, azim=0)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.text(0, 50, 10, 'Height: {}'.format(new_height_sphere))
    ax1.text(0, 50, 15, 'Width:  {}'.format(new_width_sphere))
    ax1.text(0, 50, 20, 'Depth:  {}'.format(new_depth_sphere))

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()


if __name__ == '__main__':
    # rand_mul_shi_vox_example()
    # flip_vol_example()
    zoom_z_example()