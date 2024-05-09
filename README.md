# Land-use-change-simulation-framework

This is a novel cellular automata (CA) framework featuring both spatial and temporal techniques for Land Use Change (LUC) simulation.

## Requirements

Ensure that you have the following dependencies installed:

- Python 3
- TensorFlow > 1.2
- Keras > 2.1.1
- NumPy

## Data Requirements

To effectively run the simulation, the following data is required:

1.**Time-Series Land Use Maps**: A sequence of maps that reflect land use over time.
2.**Driving Factors**: Data on the factors that influence land use changes.

## Simulation Options

### 1. Factors Spatialization

Spatial factors affecting land use can be spatialized based on:

- Euclidean Distance
- Future possibilities: Network-based time consumption and others.

### 2. Partition

Methods for data partition include:

- Self-Organizing Maps (SOM)
- K-means Clustering
- Future possibilities

### 3. Feature Extraction

Feature extraction models can act as encoders:

- Convolutional Neural Networks (CNN)
- Logistic Regression (LR)
- Random Forest (RF)
- Support Vector Machines (SVM)

### 4. Cellular Automata Models

Available CA model types:

- Raster-Based
- Future possibilities: Vector-based, Multi-label

### 5. Evaluation

Evaluation indicators:

- FoM
- Accuracy
- Kappa
- F1

## Usage

1. Install all required dependencies.
2. Prepare the data according to the Data Requirements section.
3. Configure the simulation parameters based on your specific use case.
4. Run the simulation using the appropriate CA model.

## Contributing

Contributions and improvements are welcome! Feel free to fork the repository and submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).
