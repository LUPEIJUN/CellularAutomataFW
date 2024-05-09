import numpy as np
import random
from osgeo import gdal

def FoM_Cal(im_data,im_data_land2010, im_data_land2015, im_height, im_width, output_path, name):
    error_1, error_2, error_3, correct_4, total_errors = 0, 0, 0, 0, 0
    valid_pixels = [
        (row, col)
        for row in range(12, im_height - 12)
        for col in range(12, im_width - 12)
        if im_data_land2010[row][col] != 0
    ]
    data_error = np.zeros((im_height, im_width))
    for row, col in valid_pixels:
        label = im_data[row][col]
        land2010_label = im_data_land2010[row][col]
        land2015_label = im_data_land2015[row][col]

        # Stay unchanged but simulated change
        if land2015_label == land2010_label and label != land2010_label:
            data_error[row][col] = 50
            error_1 += 1

        # Actual change but simulated unchanged
        elif land2015_label != land2010_label and label == land2010_label:
            data_error[row][col] = 100
            error_2 += 1

        # Actual and simulated change, but incorrectly predicted
        elif land2015_label != land2010_label and label not in [land2010_label, land2015_label]:
            data_error[row][col] = 150
            error_3 += 1

        # Actual and simulated change, and correctly predicted
        elif land2015_label != land2010_label and label == land2015_label:
            data_error[row][col] = 200
            correct_4 += 1

        # Total errors: Predicted results inconsistent with actual
        if land2015_label != label:
            total_errors += 1

    # Calculate Figure of Merit (FoM)
    fom = correct_4 / (error_1 + error_2 + error_3 + correct_4)

    # Print the error analysis
    print(f"{name}: Number of unchanged to changed errors (error_1): {error_1}")
    print(f"{name}: Number of changed to unchanged errors (error_2): {error_2}")
    print(f"{name}: Number of incorrectly predicted changes (error_3): {error_3}")
    print(f"{name}: Number of correctly predicted changes (correct_4): {correct_4}")
    print(f"{name}: Total errors: {total_errors}")
    print(f"{name}: Correct FoM: {fom}")

    # Save the error analysis to a text file
    np.savetxt(output_path, data_error, fmt='%s', newline='\n')
    return data_error