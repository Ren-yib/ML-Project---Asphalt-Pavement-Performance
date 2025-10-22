import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb  # XGBoost Library Import
from sklearn.model_selection import train_test_split

# Plot Setting
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False


# --- XGBoost Analysis Function ---
def run_xgboost_analysis(X, y, target_name, apply_age_constraint=False):
    # Determine settings based on whether the constraint should be applied
    monotonic_constraints = None
    title_suffix = ""
    plot_color = 'darkslateblue'  # Default color

    if apply_age_constraint:
        print(f"--- Configuring XGBoost model for '{target_name}' with Age constraint ---")
        try:
            age_column_index = X.columns.get_loc("age")
            constraints = [0] * len(X.columns)
            constraints[age_column_index] = 1
            monotonic_constraints = tuple(constraints)
            title_suffix = "\n(with Monotonic Constraint on AGE)"
            plot_color = 'darkgreen'
            print(f">>> Applying monotonic constraint on AGE (column {age_column_index})\n")
        except KeyError:
            print(">>> 'age' column not found. Proceeding without constraint.\n")
    else:
        print(f"--- Configuring XGBoost model for '{target_name}' without constraints ---")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate and train the XGBoost model
    xgb_model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.1,
        monotonic_constraints=monotonic_constraints,
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)

    # Process and plot feature importances
    importances = xgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    max_importance = feature_importance_df['Importance'].max()
    feature_importance_df['Normalized Importance'] = feature_importance_df[
                                                         'Importance'] / max_importance if max_importance > 0 else 0.0
    feature_importance_df = feature_importance_df.sort_values(by='Normalized Importance', ascending=False)

    print("Top 5 Variables (XGBoost):\n", feature_importance_df.head())

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Normalized Importance'], color=plot_color)
    plt.xlabel('Relative Importance')
    plt.title(f"XGBoost Feature Importance for '{target_name}'" + title_suffix)
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

    for sheet in sheet_names:
        print(f"*** Processing sheet: {sheet} ***\n")
        df = pd.read_excel(excel_file_name, sheet_name=sheet, header=0)
        X_data = df.iloc[:, 2:18]
        target_col_name = sheet_to_target_map[sheet]

        if target_col_name in df.columns:
            y_data_series = df[target_col_name]

            # Determine whether to apply the constraint based on the sheet name
            apply_constraint = sheet in ['Gator', 'Longwp', 'Trans']

            # Call the XGBoost function for all models
            run_xgboost_analysis(X_data, y_data_series, target_col_name, apply_age_constraint=apply_constraint)

        else:
            print(f" In '{sheet}' sheet, target variable '{target_col_name}' could not be found.")

except FileNotFoundError:
    print(f"Error: '{excel_file_name}' File is not found.")
except Exception as e:
    print(f"Unknown Error: {e}")