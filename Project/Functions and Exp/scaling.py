import pandas as pd
from sklearn.preprocessing import StandardScaler
# import joblib 
import os

def scale_features(X_train_path, X_test_path): #(scaler_save_path= 'sepsis_scaler.joblib')(for future deployment)
    """
    Loads feature data, fit a standard scalar on a training data, 
    transform both train and test data, and save the results.

    args:
          X_train_path (str): Path to the X_train parquet file.
          X_test_path (str): Path to the X_test parquet file.
          scaler_save_path (str): path to save fitted scaled features.(for future deployment)  

    returns:
          pd.DataFrame, pd.DataFame: The scaled_X_train and X_test Dataframes
    """
    if not os.path.exists("D:/Sepsis_Capstone/Project/X_train.parquet") or not os.path.exists("D:/Sepsis_Capstone/Project/X_test.parquet"):
        raise FileNotFoundError("Could not find the required files")
    
    X_train = pd.read_parquet("D:/Sepsis_Capstone/Project/X_train.parquet")
    X_test = pd.read_parquet("D:/Sepsis_Capstone/Project/X_test.parquet")
    print(f"loaded X_train{X_train.shape} and X_test{X_test.shape}")

    meta_data_cols = ['patient_id', 'ICULOS']
    present_metadata = [col for col in meta_data_cols if col in X_train.columns]

    features_to_scale= [col for col in X_train if col not in present_metadata]
    print(f"Found {len(features_to_scale)} features to scale.")

    # fitting the scalar (Train data only)
    scalar= StandardScaler()
    scalar.fit(X_train[features_to_scale])

    # save the scaler for later use
    # joblib.dump(scalar, scalar_save_path)(for future Deployment)

    # Transform Data
    X_train_scaled_np = scalar.transform(X_train[features_to_scale])
    X_test_scaled_np = scalar.transform(X_test[features_to_scale])

    # Rebuilding the DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled_np, columns= features_to_scale, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_np, columns= features_to_scale, index=X_test.index)

    # Adding the Meta Data Columns back to avoid losing the patient's Details
    for col in present_metadata:
        X_train_scaled[col] = X_train[col]
        X_test_scaled[col] = X_test[col]
    
    print("Scaling Complete")
    return X_train_scaled, X_test_scaled
