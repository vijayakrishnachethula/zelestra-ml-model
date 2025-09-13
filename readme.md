# Solar Panel Efficiency Prediction

This project predicts the efficiency of solar panels using advanced machine learning techniques and feature engineering. The model leverages LightGBM, XGBoost, and CatBoost, and uses stacking and pseudo-labeling for improved accuracy.

## Project Structure

- `model_final.py` : Main script for data processing, feature engineering, model training, stacking, and prediction.
- `train.csv` : Training dataset .
- `test.csv` : Test dataset .

## Features

- Extensive feature engineering for solar panel data.
- Aggressive feature selection using LightGBM.
- Model stacking with Ridge regression meta-model.
- Pseudo-labeling for refining predictions.
- Handles categorical and numerical features robustly.

## Requirements

- Python 3.7+
- pandas
- numpy
- lightgbm
- xgboost
- catboost
- scikit-learn

Install dependencies:
```
pip install pandas numpy lightgbm xgboost catboost scikit-learn
```

## Usage

1. Place `train.csv` and `test.csv` in the project directory.
2. Run the main script:
```
python model_final.py
```
3. The predictions will be saved to `submission.csv`.

## Model Pipeline

1. **Feature Engineering:** Creates new features from raw data.
2. **Feature Selection:** Selects important features using LightGBM.
3. **Model Training:** Trains LightGBM, XGBoost, and CatBoost models with cross-validation.
4. **Stacking:** Combines model outputs using RidgeCV.
5. **Pseudo-Labeling:** Augments training data with high-confidence test predictions.
6. **Submission:** Outputs final predictions clipped between 0 and 1.

## Notes

- Data files (`train.csv`, `test.csv`) are included in this repository for reproducibility.
- For best results, ensure data columns match those expected in the script.

## License

This project is for educational and research purposes.