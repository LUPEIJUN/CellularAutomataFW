import numpy as np
import pylab as pl
from sklearn.cluster import KMeans
from osgeo import gdal
from sklearn import preprocessing

def k_means_model(im_data_land2010, file_allfactor_2010, im_height, im_width, output_path):
    cluster_data = np.loadtxt(file_allfactor_2010)
    estimator = KMeans(n_clusters=16)
    labels = estimator.fit_predict(cluster_data)

    cluster_result = np.full((im_height, im_width), 100)
    valid_pixels = [
        (row, col)
        for row in range(12, im_height - 12)
        for col in range(12, im_width - 12)
        if im_data_land2010[row][col] != 0
    ]

    for index, (row, col) in enumerate(valid_pixels):
        cluster_result[row][col] = labels[index]

    np.savetxt(output_path, cluster_result, fmt='%s', newline='\n')
    return cluster_result