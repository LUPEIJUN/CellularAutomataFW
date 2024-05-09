from data_preparation_and_split import Data_preparation_and_split,read_tiff
from cellular_automata_dl.ANN_CA import ANN_model_train, ANN_model_predict
from MLDL.LR  import LR_model_train,  LR_model_predict
from MLDL.RF  import RF_model_train,  RF_model_predict
from MLDL.SVM import SVM_model_train, SVM_model_predict
from MLDL.CNN import CNN_model_train, CNN_model_predict
from Evaluation.Multilabel_Precision_Recal import Confusion_Cal,Multilabel_Precision_Recall
from Evaluation.FoM import FoM_Cal
from Cluster.k_means import k_means_model
from Cluster.SOM import SOM_model

import numpy as np
def run():
    
    file_land2000 = '/input/Landuse/2000_final.tif'
    file_land2005 = '/input/Landuse/2005_final.tif'
    file_land2010 = '/input/Landuse/2010_final.tif'
    file_land2015 = '/input/Landuse/2015_final.tif'
    file_AllFactor_2005= '/input/Factor/2010_Factors_all.txt'
    file_AllFactor_2010= '/input/Factor/2010_Factors_all.txt'
    file_neighbor_2005 = '/input/Neighbor/2010_Neighbor_raster.txt'
    file_neighbor_2010 = '/input/Neighbor/2010_Neighbor_raster.txt'

    file_out_ann = '/output/MLDL/ANN_CA.txt'
    file_out_lr = '/output/MLDL/LR_CA.txt'
    file_out_rf = '/output/MLDL/RF_CA.txt'
    file_out_svm = '/output/MLDL/SVM_CA.txt'
    file_out_cnn = '/output/MLDL/CNN_CA.txt'

    file_out_fom = '/output/Evaluation/FoM.txt'
    file_out_k_means = '/output/Cluster/k_means.txt'
    file_out_som = '/output/Cluster/som.txt'

    im_data_land2000, im_height, im_width = read_tiff(file_land2000)
    im_data_land2005, im_height, im_width = read_tiff(file_land2005)
    im_data_land2010, _, _ = read_tiff(file_land2010)
    im_data_land2015, _, _ = read_tiff(file_land2015)
    number_of_pixels = np.count_nonzero(im_data_land2005)
    data_train, data_test, data_train_label, data_test_label,label_2005, label_2010,sample_2010_norm = \
        Data_preparation_and_split(file_land2005, file_land2010, file_land2015, file_AllFactor_2005, file_AllFactor_2010, file_neighbor_2005, file_neighbor_2010 )

    ann_model = ANN_model_train(data_train, data_test, data_train_label, data_test_label)
    ann_whole = ANN_model_predict(ann_model, sample_2010_norm,im_data_land2010,im_data_land2015,im_height,im_width,number_of_pixels,file_out_ann)

    lr_model = LR_model_train(data_train, data_test, data_train_label, data_test_label)
    lr_whole = LR_model_predict(lr_model, sample_2010_norm,im_data_land2010,im_data_land2015,im_height,im_width,number_of_pixels,file_out_lr)

    rf_model = RF_model_train(data_train, data_test, data_train_label, data_test_label)
    rf_whole = RF_model_predict(rf_model, sample_2010_norm,im_data_land2010,im_data_land2015,im_height,im_width,number_of_pixels,file_out_rf)

    svm_model = SVM_model_train(data_train, data_test, data_train_label, data_test_label)
    svm_whole = SVM_model_predict(svm_model, sample_2010_norm,im_data_land2010,im_data_land2015,im_height,im_width,number_of_pixels,file_out_svm)

    cnn_model = CNN_model_train(data_train, data_test, data_train_label, data_test_label)
    cnn_whole = CNN_model_predict(cnn_model, sample_2010_norm,im_data_land2010,im_data_land2015,im_height,im_width,number_of_pixels,file_out_cnn)

    k_means_res = k_means_model(im_data_land2000, file_AllFactor_2010, im_height, im_width, file_out_k_means)
    som_res = SOM_model(im_data_land2000, file_AllFactor_2010, im_height, im_width, file_out_som)

    # Calculate the confusion matrix for ANN model 2015 WholeArea
    confusion_matrix_ann_2015 = Confusion_Cal(im_data_land2015,ann_whole)
    PRF_ann,Accuracy_ann,Kappa_ann,macro_score_ann = Multilabel_Precision_Recall(confusion_matrix_ann_2015,"ann")

    data_error = FoM_Cal(ann_whole,im_data_land2010, im_data_land2015, im_height, im_width, file_out_fom, "ann")

if __name__ == "__main__":
    run()