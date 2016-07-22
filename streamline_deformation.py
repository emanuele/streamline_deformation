"""Example of deformation of an entire tractogram using a given
deformation field.

This code requires the package 'regulargrid' in order to run:
  https://github.com/JohannesBuchner/regulargrid
"""

import numpy as np
import nibabel
from regulargrid.cartesiangrid import CartesianGrid


if __name__ == '__main__':
    subject = '124422'
    filename_deformation = 'HCP/' + subject + \
                           '/nonlinear_acpc_dc2standard.nii.gz'
    filename_trk = 'HCP/' + subject + \
                   '/Diffusion/DTI/tracks_dti_1M.trk'

    print("Loading defomation file: %s" % filename_deformation)
    deformation_img = nibabel.load(filename_deformation)
    deformation_data = deformation_img.get_data()
    grid_size = deformation_img.header.get_zooms()
    grid_shape = deformation_img.get_shape()
    maxs = (grid_size * np.array(grid_shape))[:3]
    limits = zip((0.0, 0.0, 0.0), maxs)
    print("The volume has the following limits: %s" % limits)

    print("Creating the Cartesian deformation grid.")
    grid_x = CartesianGrid(limits, values=deformation_data[:, :, 0])
    grid_y = CartesianGrid(limits, values=deformation_data[:, :, 1])
    grid_z = CartesianGrid(limits, values=deformation_data[:, :, 2])

    print("Computing the deformation of one point.")
    test_point = ([120.0], [120.0], [120.0])
    print("test point: %s" % (test_point, ))
    delta_x = grid_x(*test_point)
    delta_y = grid_y(*test_point)
    delta_z = grid_z(*test_point)
    print("deformation: %s" % (np.array([delta_x, delta_y, delta_z]), ))

    from time import time
    size = 1000
    print("Estimating time for deforming %d points." % size)
    tmp = np.random.uniform(low=0.0, high=100, size=(size, 3))
    points = (tmp[:, 0], tmp[:, 1], tmp[:, 2])
    t0 = time()
    delta_x = grid_x(*points)
    delta_y = grid_y(*points)
    delta_z = grid_z(*points)
    print("%f sec." % (time() - t0))

    print("Loading tracogram: %s" % filename_trk)
    streamlines, header = nibabel.trackvis.read(filename_trk)
    streamlines_deformed = []
    print("Deforming tractogram.")
    t0 = time()
    for i, x in enumerate(streamlines):
        if i % 1000 == 0:
            print(i)

        s = x[0]
        delta_x = grid_x(s[:, 0], s[:, 1], s[:, 2])
        delta_y = grid_y(s[:, 0], s[:, 1], s[:, 2])
        delta_z = grid_z(s[:, 0], s[:, 1], s[:, 2])
        s_deformed = s + np.vstack([delta_x, delta_y, delta_z]).T
        streamlines_deformed.append((s_deformed, None, None))

    print("%f sec." % (time() - t0))
    
    filename_trk_deformed = filename_trk[:-4] + '_deformed.trk'
    print("Saving deformed tractogram: %s" % filename_trk_deformed)
    nibabel.trackvis.write(filename_trk_deformed, streamlines_deformed, header)
