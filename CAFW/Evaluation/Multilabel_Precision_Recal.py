import numpy as np
from joblib import Parallel, delayed

# Main function to compute confusion matrix values. 
def Confusion_Cal(im_data_refer, data_new):
    im_height = im_data_refer.shape[0]
    im_width = im_data_refer.shape[1]
    coordinates = [
        (row, col)
        for row in range(12, im_height - 12)
        for col in range(12, im_width - 12)
    ]

    # Parallel computation.
    results = Parallel(n_jobs=-1)(delayed(_calculate_cell)(im_data_refer, data_new, row, col) for row, col in coordinates)
    Confusion_matrix = np.sum(results, axis=0)
    return Confusion_matrix

# Helper function to compute confusion matrix values.
def _calculate_cell(im_data_refer, data_new, row, col):
    Confusion_matrix = np.zeros((6, 6))    
    ref_value = im_data_refer[row][col]
    new_value = data_new[row][col]
    if ref_value != 0:
        Confusion_matrix[ref_value - 1][new_value - 1] += 1    
    return Confusion_matrix


def Multilabel_Precision_Recall(confusion_matrix,name):

    num_classes = confusion_matrix.shape[0]
    PrecisionRecallF1 = np.zeros((num_classes, 3))

    for i in range(num_classes):
        # Precision: True Positives / (True Positives + False Positives)
        tp = confusion_matrix[i][i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        
        # Recall: True Positives / (True Positives + False Negatives)
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        
        # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0
        PrecisionRecallF1[i] = [precision, recall, f1_score]

    # Overall accuracy
    total_true = np.trace(confusion_matrix)
    total_samples = np.sum(confusion_matrix)
    accuracy = total_true / total_samples

    # Cohen's kappa
    expected_agreement = np.sum(np.sum(confusion_matrix, axis=0) * np.sum(confusion_matrix, axis=1)) / (total_samples ** 2)
    kappa = (accuracy - expected_agreement) / (1 - expected_agreement) if (1 - expected_agreement) != 0 else 0.0
    
    # Macro F1 Score (average of all class F1 scores)
    macro_F1_score = np.mean(PrecisionRecallF1[:, 2])
    
    print(f"{name} : PRF {PrecisionRecallF1}")
    print(f"{name} : Accuracy {accuracy}")
    print(f"{name} : Kappa {kappa}")
    print(f"{name} : macro_F1_score {macro_F1_score}")

    return PrecisionRecallF1, accuracy, kappa, macro_F1_score