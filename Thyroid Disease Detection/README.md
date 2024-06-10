# Thyroid Disease Detection

This project aims to predict thyroid disease based on various patient features using machine learning techniques.

## Files

- `Thyroid_Prediction.ipynb`: Jupyter Notebook containing the data exploration, model training, and evaluation code.
- `app.py`: Streamlit web application for interactive prediction of thyroid disease.
- `thyroidDF.csv`: Dataset containing patient features and target labels.
- `thyroid_prediction.py`: Python script containing the model training and prediction code.

## Usage

### Jupyter Notebook (Thyroid_Prediction.ipynb)

1. Open the notebook in Jupyter or any compatible environment.
2. Run the cells sequentially to load the data, train the model, and evaluate its performance.
3. Explore the analysis and results presented in the notebook.

### Streamlit Web Application (app.py)

1. Install the necessary Python packages: `pip install -r requirements.txt`.
2. Run the Streamlit app: `streamlit run app.py`.
3. Use the interactive interface to input patient data and predict thyroid disease.

### Model Training and Prediction (thyroid_prediction.py)

1. Execute the script: `python thyroid_prediction.py`.
2. The script loads the dataset, preprocesses the data, trains the model, and saves the trained model as `model.pkl`.
3. You can then use the saved model for making predictions in other applications.

## Dependencies

- Python 3.x
- TensorFlow
- Streamlit
- Pandas
- Scikit-learn
- Plotly
- Joblib
- Pickle

Install the required packages using `pip install -r requirements.txt`.

## License

This project is licensed under the [MIT License](LICENSE).