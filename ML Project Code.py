import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb  # XGBoost Library Import

# Plot Setting
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False


# ---  Random Forest Analysis Function (Rutting, Roughness) ---
def run_random_forest_analysis(X, y, target_name, model_config):
    print(f"--- '{target_name}'configuring Random forest model ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(
        n_estimators=model_config['B'], max_features=model_config['m'],
        random_state=42, oob_score=True, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    max_importance = feature_importance_df['Importance'].max()
    feature_importance_df['Normalized Importance'] = feature_importance_df[
                                                         'Importance'] / max_importance if max_importance > 0 else 0.0
    feature_importance_df = feature_importance_df.sort_values(by='Normalized Importance', ascending=False)
    print("상위 5개 중요 변수(RF):\n", feature_importance_df.head())
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Normalized Importance'], color='mediumpurple')
    plt.xlabel('Relative Importance')
    plt.title(f"Random Forest Feature Importance for '{target_name}'")
    plt.gca().invert_yaxis();
    plt.tight_layout();
    plt.show()
    print("-" * 40 + "\n")


# --- XGBoost Analysis Function with a monotonic constraint of Age ---
def run_xgboost_with_monotonicity(X, y, target_name):
    print(f"--- '{target_name}'configuring XGBoost model ---")

    # Locate the column of Age
    try:
        age_column_index = X.columns.get_loc("age")
    except KeyError:
        print("'AGE' column was not located, will be proceeded with Random Forest model")
        # In case Age is not located, Proceed with Random forest model
        return

    # Monotonic Constraint Setting: AGE Column: '1', Rest: '0'
    constraints = [0] * len(X.columns)
    constraints[age_column_index] = 1
    print(f">>> AGE({age_column_index} column), Monotonic constraint will be applied\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.1,
        monotonic_constraints=tuple(constraints),
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)

    importances = xgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    max_importance = feature_importance_df['Importance'].max()
    feature_importance_df['Normalized Importance'] = feature_importance_df[
                                                         'Importance'] / max_importance if max_importance > 0 else 0.0
    feature_importance_df = feature_importance_df.sort_values(by='Normalized Importance', ascending=False)

    print("Top 5 factors (XGBoost):\n", feature_importance_df.head())

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Normalized Importance'], color='darkgreen')
    plt.xlabel('Relative Importance')
    plt.title(f"XGBoost Feature Importance for '{target_name}'\n(with Monotonic Constraint on AGE)")
    plt.gca().invert_yaxis();
    plt.tight_layout();
    plt.show()
    print("-" * 40 + "\n")


# --- Main analysis block ---
try:
    excel_file_name = '00 LTPP Data Final v4.xlsx'
    sheet_names = ['Gator', 'Longwp', 'Trans', 'Rutting', 'Roughness']
    sheet_to_target_map = {
        'Gator': 'gator', 'Longwp': 'longwp', 'Trans': 'trans',
        'Rutting': 'rut', 'Roughness': 'iri'
    }
    model_configurations = {
        'gator': {'B': 128, 'm': 6}, 'longwp': {'B': 112, 'm': 5},
        'trans': {'B': 114, 'm': 6}, 'rut': {'B': 106, 'm': 5},
        'iri': {'B': 122, 'm': 6},
    }

    for sheet in sheet_names:
        print(f"*** Processing sheet: {sheet} ***\n")
        df = pd.read_excel(excel_file_name, sheet_name=sheet, header=0)
        X_data = df.iloc[:, 2:18]
        target_col_name = sheet_to_target_map[sheet]

        if target_col_name in df.columns:
            y_data_series = df[target_col_name]
            if sheet in ['Gator', 'Longwp', 'Trans']:
                run_xgboost_with_monotonicity(X_data, y_data_series, target_col_name)
            else:
                model_config = model_configurations[target_col_name]
                run_random_forest_analysis(X_data, y_data_series, target_col_name, model_config)
            # <<<-------------------- >>>
        else:
            print(f" In '{sheet}' sheet, target variable'{target_col_name}'could not be found.")

except FileNotFoundError:
    print(f"오류: '{excel_file_name}' File is not found.")
except Exception as e:
    print(f"Unknown Error: {e}")