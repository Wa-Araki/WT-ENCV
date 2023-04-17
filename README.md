# WT-ENCV
This is a machine-learning model based on a chemometrics approach to accurately predict the concentration of components from spectra. The model is interpretable by displaying the regression coefficients on Gaussian distributions with various widths.

We provide a data sets (data_sets.csv) containing monomer concentrations in copolymerization solutions and FTIR spectra to demonstrate the capabilities of this model. The model class is designed to resemble scikit-learn's basic structure. Please try out the example.ipynb.

## Getting Started
### Prerequisites
Ensure that you have the following libraries installed. The model relies on basic functionality, so there is minimal dependency on specific library versions.

```
numpy
pandas
scikit-learn
scipy
matplotlib
```

### Installation
Clone the GitHub repository:

```
git clone https://github.com/Wa-Araki/WT-ENCV.git
```

Change to the repository's directory:
```
cd WT-ENCV
```
Install the required libraries, if not already installed:

```
pip install numpy pandas scikit-learn scipy matplotlib
```

### Process
This model first decomposes a spectrum into peaks with various widths using wavelet transform.

Subsequently, a sparse linear regression model is built using the wavelet coefficients.

The interpretation of this model is expected to reveal the relationship between broad regions in spectra and the model prediction. We are planning to publish our research findings, comparing the results of our model with other models.

### Usage
To use the model, follow these steps:

1. Run the provided Jupyter Notebook (example.ipynb) to explore how the model works with the provided data sets (data_sets.csv).

2. Modify the Notebook or create your own script to utilize the model with your own data, following the same structure demonstrated in example.ipynb.

3. Evaluate the performance of the model and make any necessary adjustments to improve its accuracy and interpretability.

## Contributing
We welcome contributions to the WT-ENCV repository. To contribute, please:

1. Fork the repository.

2. Create a new branch for your feature or bugfix.

3. Commit your changes and push to your fork.

4. Create a pull request to merge your changes into the main repository.

Please ensure your code follows the style guidelines and passes any provided tests.

## License
This project is licensed under the GNU Lesser General Public License v3 (LGPL v3.0).
