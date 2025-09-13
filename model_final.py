

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
import warnings

warnings.filterwarnings('ignore')


def calculate_custom_score(y_true, y_pred):
    """Calculates the competition's specific score based on RMSE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    score = 100 * (1 - rmse)
    return score


print("\n[Stage 1/5] Loading and engineering features...")
try:
    train_df = pd.read_csv('train.csv'); test_df = pd.read_csv('test.csv')
except FileNotFoundError as e:
    print(f"ERROR: Raw data file not found: {e}."); exit()

full_df = pd.concat([train_df.drop('efficiency', axis=1), test_df], axis=0, ignore_index=True)
y_true = train_df['efficiency']; test_ids = test_df['id']
numeric_cols = ['temperature', 'irradiance', 'humidity', 'panel_age', 'maintenance_count', 'soiling_ratio', 'voltage', 'current', 'module_temperature', 'cloud_coverage', 'wind_speed', 'pressure']
categorical_cols = ['string_id', 'error_code', 'installation_type']
for col in numeric_cols: full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
for col in categorical_cols: full_df[col] = full_df[col].astype(str).fillna('missing')
train_medians = full_df.iloc[:len(train_df)][numeric_cols].median(); full_df[numeric_cols] = full_df[numeric_cols].fillna(train_medians)
full_df['temp_difference'] = full_df['module_temperature'] - full_df['temperature']; full_df['power_approx'] = full_df['voltage'] * full_df['current']; full_df['power_irradiance_ratio'] = full_df['power_approx'] / (full_df['irradiance'] + 1e-6)



full_df['maintenance_per_age'] = full_df['maintenance_count'] / (full_df['panel_age'] + 1)
full_df['power_per_maintenance'] = full_df['power_approx'] / (full_df['maintenance_count'] + 1)

full_df['voltage_efficiency'] = full_df['voltage'] / (full_df['module_temperature'] + 273.15)  
full_df['current_per_irradiance'] = full_df['current'] / (full_df['irradiance'] + 1)

string_power_mean = full_df.groupby('string_id')['power_approx'].transform('mean')
full_df['string_deviation'] = abs(full_df['power_approx'] - string_power_mean) / (string_power_mean + 1)



full_df['error_frequency'] = full_df.groupby('error_code')['error_code'].transform('count')
full_df['error_rarity_score'] = 1 / (full_df['error_frequency'] + 1)

string_power_mean = full_df.groupby('string_id')['power_approx'].transform('mean')
full_df['string_deviation'] = abs(full_df['power_approx'] - string_power_mean) / (string_power_mean + 1)

full_df['thermal_zone'] = pd.cut(full_df['temp_difference'], bins=5, labels=[1,2,3,4,5]).astype(float)
full_df['optimal_thermal_zone'] = (full_df['thermal_zone'] == 3).astype(int)

install_voltage_median = full_df.groupby('installation_type')['voltage'].transform('median')
full_df['install_voltage_advantage'] = full_df['voltage'] / (install_voltage_median + 1)

full_df['thermal_maintenance_efficiency'] = full_df['maintenance_per_age'] * full_df['optimal_thermal_zone']


full_df['env_stability'] = 1 / (1 + full_df['cloud_coverage']/100 + full_df['humidity']/100 + abs(full_df['wind_speed'] - 3)/10)




for group_col in ['string_id', 'installation_type']:
    for num_col in ['temperature', 'irradiance', 'voltage', 'power_approx']:
        agg_stats = full_df.groupby(group_col)[num_col].agg(['mean', 'std']).add_prefix(f'{num_col}_by_{group_col}_'); full_df = full_df.merge(agg_stats, on=group_col, how='left')
full_df.fillna(0, inplace=True)
X_fe = full_df[:len(train_df)].drop('id', axis=1); X_test_fe = full_df[len(train_df):].drop('id', axis=1)
X_proc = pd.get_dummies(X_fe, columns=categorical_cols, dummy_na=False); X_test_proc = pd.get_dummies(X_test_fe, columns=categorical_cols, dummy_na=False)
X_proc, X_test_proc = X_proc.align(X_test_proc, join='left', axis=1, fill_value=0)
print("Feature engineering complete.")

print("\n[Stage 2/5] Performing aggressive feature selection...")
fs_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
fs_model.fit(X_proc, y_true)
importances = pd.DataFrame({'feature': X_proc.columns, 'importance': fs_model.feature_importances_})

important_features = importances[importances['importance'] > 1].feature.tolist()

X_proc_selected = X_proc[important_features]
X_test_proc_selected = X_test_proc[important_features]
print(f"Feature selection complete. Reduced from {len(X_proc.columns)} to {len(important_features)} features.")

N_SPLITS = 10

y_bins = pd.cut(y_true, bins=10, labels=False)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

lgbm_params = {'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 2500, 'learning_rate': 0.01, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'lambda_l1': 0.1, 'lambda_l2': 0.1, 'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42}
xgb_params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'n_estimators': 2500, 'learning_rate': 0.01, 'max_depth': 7, 'subsample': 0.7, 'colsample_bytree': 0.7, 'gamma': 0.2, 'lambda': 1, 'alpha': 0.5, 'random_state': 42, 'n_jobs': -1, 'tree_method': 'hist'}
catboost_params = {'iterations': 3000, 'learning_rate': 0.015, 'depth': 8, 'l2_leaf_reg': 5, 'loss_function': 'RMSE', 'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0}

oof_preds = np.zeros((len(X_proc), 3)); test_preds = np.zeros((len(X_test_proc), 3))
print(f"\n[Stage 3/5] Starting {N_SPLITS}-Fold Stratified Cross-Validation...")

for fold, (train_index, val_index) in enumerate(skf.split(X_proc_selected, y_bins)):
    print(f"--- FOLD {fold+1}/{N_SPLITS} ---")
    y_train, y_val = y_true.iloc[train_index], y_true.iloc[val_index]
    
    
    X_proc_train, X_proc_val = X_proc_selected.iloc[train_index], X_proc_selected.iloc[val_index]
  
    X_fe_train, X_fe_val = X_fe.iloc[train_index], X_fe.iloc[val_index]
    cat_features_names = [col for col in X_fe_train.columns if X_fe_train[col].dtype == 'object']

    print("  - Training LGBM..."); lgbm_model = lgb.LGBMRegressor(**lgbm_params); lgbm_model.fit(X_proc_train, y_train, eval_set=[(X_proc_val, y_val)], callbacks=[lgb.early_stopping(150, verbose=False)]); oof_preds[val_index, 0] = lgbm_model.predict(X_proc_val); test_preds[:, 0] += lgbm_model.predict(X_test_proc_selected) / N_SPLITS
    print("  - Training XGBoost..."); xgb_model = xgb.XGBRegressor(**xgb_params); xgb_model.fit(X_proc_train, y_train, eval_set=[(X_proc_val, y_val)], verbose=False); oof_preds[val_index, 1] = xgb_model.predict(X_proc_val); test_preds[:, 1] += xgb_model.predict(X_test_proc_selected) / N_SPLITS
    print("  - Training CatBoost..."); cat_model = catboost.CatBoostRegressor(**catboost_params, cat_features=cat_features_names); cat_model.fit(X_fe_train, y_train, eval_set=(X_fe_val, y_val), early_stopping_rounds=150, use_best_model=True); oof_preds[val_index, 2] = cat_model.predict(X_fe_val); test_preds[:, 2] += cat_model.predict(X_test_fe) / N_SPLITS



print("\n[Stage 4/5] Evaluating base models and training meta-model...")
model_names = ['LGBM', 'XGBoost', 'CatBoost']
oof_scores = []
for i, name in enumerate(model_names):
    score = calculate_custom_score(y_true, oof_preds[:, i]); oof_scores.append(score); print(f"{name} OOF Score:     {score:.6f}")

level1_features_train = pd.DataFrame(oof_preds, columns=model_names)
level1_features_test = pd.DataFrame(test_preds, columns=model_names)

meta_model = RidgeCV(alphas=np.logspace(-2, 2, 50))
meta_model.fit(level1_features_train, y_true)
oof_stack_preds = meta_model.predict(level1_features_train)
stack_score = calculate_custom_score(y_true, oof_stack_preds)
print(f"\nLevel 1 Stack OOF Score: {stack_score:.6f}")

print("\n[Stage 5/5] Performing Pseudo-Labeling to refine predictions...")
test_predictions_initial = meta_model.predict(level1_features_test)

CONF_THRESH_UPPER = 0.95
CONF_THRESH_LOWER = 0.05
pseudo_indices = np.where((test_predictions_initial > CONF_THRESH_UPPER) | (test_predictions_initial < CONF_THRESH_LOWER))[0]

if len(pseudo_indices) > 0:
    print(f"  - Found {len(pseudo_indices)} high-confidence test samples to use as pseudo-labels.")
    
    X_pseudo = X_proc_selected.iloc[pseudo_indices]
    y_pseudo = test_predictions_initial[pseudo_indices]

    
    X_augmented = pd.concat([X_proc_selected, X_pseudo], axis=0)
    y_augmented = np.concatenate([y_true, y_pseudo])
    
    print("  - Retraining best model on augmented data...")
    final_model = lgb.LGBMRegressor(**lgbm_params)
    final_model.fit(X_augmented, y_augmented)
    
    final_predictions = final_model.predict(X_test_proc_selected)
else:
    print("  - No high-confidence samples found for pseudo-labeling. Using initial stacked predictions.")
    final_predictions = test_predictions_initial

LOWER_BOUND, UPPER_BOUND = 0.0, 1.0
final_predictions_clipped = np.clip(final_predictions, LOWER_BOUND, UPPER_BOUND)
submission_df = pd.DataFrame({'id': test_ids, 'efficiency': final_predictions_clipped})
submission_df.to_csv('submission.csv', index=False)

