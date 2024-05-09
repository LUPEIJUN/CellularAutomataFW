import numpy as np
import random
from osgeo import gdal
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

# Function to read and process a TIFF file using GDAL
def read_tiff(file_path):
    data = gdal.Open(file_path)
    im_width = data.RasterXSize
    im_height = data.RasterYSize
    im_data = data.ReadAsArray(0, 0, im_width, im_height)
    return im_data, im_height, im_width

# Function to load features and neighborhood data
def _load_samples(all_factors_path, neighbor_path):
    all_factors = np.loadtxt(all_factors_path)
    neighbor_data = np.loadtxt(neighbor_path)
    return np.concatenate((all_factors, neighbor_data), axis=1)

# Read land use data
def Data_preparation_and_split(file_land2005, file_land2010, file_land2015, file_AllFactor_2005, file_AllFactor_2010, file_neighbor_2005, file_neighbor_2010 ):
    im_data_land2005, im_height, im_width = read_tiff(file_land2005)
    im_data_land2010, _, _ = read_tiff(file_land2010)
    im_data_land2015, _, _ = read_tiff(file_land2015)
    
    # Load samples data 
    sample_2005 = _load_samples(file_AllFactor_2005, file_neighbor_2005)
    sample_2010 = _load_samples(file_AllFactor_2010, file_neighbor_2010)

    # Normalize samples
    scaler = MinMaxScaler()
    sample_2005_norm = scaler.fit_transform(sample_2005)
    sample_2010_norm = scaler.transform(sample_2010)

    # Generate labels
    label_2005 = np.array([im_data_land2010[row][col] for row in range(12, im_height - 12) for col in range(12, im_width - 12) if im_data_land2005[row][col] != 0]).reshape(-1, 1)
    label_2010 = np.array([im_data_land2015[row][col] for row in range(12, im_height - 12) for col in range(12, im_width - 12) if im_data_land2010[row][col] != 0]).reshape(-1, 1)


    # Randomly sample 20% of the data 
    ann_data, _, ann_data_label, _ = train_test_split(
        sample_2005_norm, label_2005, test_size=0.8, stratify=label_2005, random_state=42)

    # Split the 20% sampled data into training (70%) and testing (30%) sets
    ann_data_train, ann_data_test, ann_data_train_label, ann_data_test_label = train_test_split(
        ann_data, ann_data_label, test_size=0.3, stratify=ann_data_label, random_state=42)

    # One-hot encode labels
    num_classes = 7
    ann_data_train_label = to_categorical(ann_data_train_label, num_classes=num_classes)
    ann_data_test_label = to_categorical(ann_data_test_label, num_classes=num_classes)

    return ann_data_train, ann_data_test, ann_data_train_label, ann_data_test_label ,label_2005, label_2010,sample_2010_norm

