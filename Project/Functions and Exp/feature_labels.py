import pandas as pd
import numpy as np

def create_features_and_labels(df_train, window_size=12):
    """
    Takes a raw, imputed dataframe (train or test) and applies
    all feature engineering (X) and label creation (y).
    """
    
    print(f"--- Starting feature engineering for dataframe with shape {df_train.shape} ---")
    
    # --- 1. Create Target Label (y) ---
    print("Creating 6-hour lookahead target (y)...")
    grouped_y = df_train.groupby('patient_id')['SepsisLabel']
    y_label = grouped_y.shift(-1).rolling(6, min_periods=1).max()
    y_label = y_label.reset_index(level=0, drop=True)
    df_train['SepsisLabel_6hr_lookahead'] = y_label
    df_train['SepsisLabel_6hr_lookahead'].fillna(0, inplace=True)

    # --- 2. Define Feature Columns ---
    admin_cols = ['patient_id', 'ICULOS', 'SepsisLabel', 'SepsisLabel_6hr_lookahead']
    static_features_cols = ['Age', 'Gender', 'HospAdmTime', 'Unit1', 'Unit2']
    
    vitals_and_labs_cols = [
        col for col in df_train.columns 
        if col not in admin_cols and col not in static_features_cols
    ]
    print(f"Found {len(vitals_and_labs_cols)} vitals/labs columns to engineer.")

    # --- 3. Create Rolling Features (X) ---
    print(f"Calculating {window_size}-hour rolling features (X)...")
    
    grouped = df_train.groupby('patient_id')
 
    shifted_features = grouped[vitals_and_labs_cols].shift(1)
    
    # Group the SHIFTED data by the original patient_id index
    rolling_groups = shifted_features.groupby(df_train['patient_id'])

    print("Calculating mean...")
    df_mean = rolling_groups.rolling(window_size, min_periods=1).mean()
    print("Calculating std dev...")
    df_std = rolling_groups.rolling(window_size, min_periods=1).std()
    print("Calculating max...")
    df_max = rolling_groups.rolling(window_size, min_periods=1).max()

    # --- 4. Clean up Indexes ---
    df_mean = df_mean.reset_index(level=0, drop=True)
    df_std = df_std.reset_index(level=0, drop=True)
    df_max = df_max.reset_index(level=0, drop=True)

    df_mean.columns = [f'{col}_mean_{window_size}hr' for col in df_mean.columns]
    df_std.columns = [f'{col}_std_{window_size}hr' for col in df_std.columns]
    df_max.columns = [f'{col}_max_{window_size}hr' for col in df_max.columns]

    # --- 5. Combine All Features ---
    print("Combining all features...")
    
    
    features_df = df_train[static_features_cols + ['patient_id', 'ICULOS']]
    features_df = pd.concat([features_df, df_mean, df_std, df_max], axis=1)

    labels_df = df_train['SepsisLabel_6hr_lookahead']

    # --- 6. Final Cleanup ---
    print("Cleaning up NaNs...")
    features_df.fillna(0, inplace=True)
    
    
    final_features = features_df[labels_df.notna()].copy()
    final_labels = labels_df[labels_df.notna()].copy()
    
    print("--- Feature engineering complete! ---")
    
    return final_features, final_labels