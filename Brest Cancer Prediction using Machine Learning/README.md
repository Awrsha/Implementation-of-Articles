# Breast Cancer Prediction using Machine Learning and Deep Learning

This project aims to predict breast cancer using machine learning and deep learning techniques. The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset available in the `sklearn.datasets` module. The dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, and the target variable indicates whether the mass is malignant or benign.

## Project Structure

- **Notebook**: `breast_cancer_pred_ml.ipynb`
  - Jupyter notebook containing the entire pipeline of data preprocessing, model selection, evaluation, and deep learning implementation using TensorFlow/Keras.
  
- **README.md**: `README.md`
  - This file, providing an overview of the project, its goals, and the structure of the repository.

## Requirements

- Python 3
- Jupyter Notebook
- Required libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `imbalanced-learn`, `tensorflow`, `keras`, `xgboost`, `catboost`
  
## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/amir-rs/breast-cancer-prediction.git
   ```

2. Navigate to the project directory:

   ```bash
   cd breast-cancer-prediction
   ```

3. Open and run the Jupyter notebook `breast_cancer_pred_ml.ipynb`:

   ```bash
   jupyter notebook breast_cancer_pred_ml.ipynb
   ```

4. Follow the instructions in the notebook to execute each cell and observe the results.

## Results

- Various machine learning models such as Random Forest, AdaBoost, XGBoost, Logistic Regression, and SVM are trained and evaluated on the dataset.
- Deep learning model using TensorFlow/Keras is implemented and evaluated for breast cancer prediction.
- Performance metrics such as accuracy, precision, recall, and F1 score are calculated for each model.
- Receiver Operating Characteristic (ROC) curves are plotted to visualize model performance.

## Contributions

Contributions to improve the project are welcome! If you find any issues or have suggestions for enhancements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.