import pandas as pd
import numpy as np
import os
from tqdm import tqdm 

path_a = 'D:/Sepsis_Capstone/Project/training_setA' #  copied all the other file in single folder set_A to prevent mid processing crashes or error. 
#path_b = 'D:/Sepsis_Capstone/Project/training_setB' # no need to make a seperate path 

files_A = [os.path.join(path_a, f) for f in os.listdir(path_a) if f.endswith('.psv')]  
#files_B = [os.path.join(path_b, f) for f in os.listdir(path_a) if f.endswith('.psv')] # this step is not required any more

#all_patient_files = files_A+files_B  # this step is not required any more


print(f"Total patient files found: {len(files_A)}")

# -- Define columns types --
# columns that are static (demographics)
static_cols = ['Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS']

# All other columns (except our target) are vitals or labs 
all_cols = pd.read_csv(files_A[0], sep='|').columns
vital_cols = [col for col in all_cols if col not in static_cols and col!= 'SepsisLabel']

processed_data_list= []


for file_path in tqdm(files_A):
    try:
        df_patient = pd.read_csv(file_path, sep='|')

    #----Imputation steps
    #----Fill up the empty rows of the specified columns
        df_patient[static_cols]= df_patient[static_cols].ffill().bfill()
        df_patient[vital_cols] = df_patient[vital_cols].ffill().bfill()
    #----Add the Patient ID
    #----Easier to determine who's vitals belongs to 
        patient_id = file_path.split(os.sep)[-1].replace('.psv','')
        df_patient['patient_id']=patient_id
    
    #----Append to our master data list
        processed_data_list.append(df_patient)
    
    except Exception as e:
        print(f"\nError in Processing file{file_path}:{e}")

# --- Combine and Save (This happens AFTER the loop) ---
if not processed_data_list:
    print("No data was processed. Please check your files and paths.")
else:
    print("\nCombining all processed dataframes...")
    # This step might take a lot of RAM
    df_master = pd.concat(processed_data_list, ignore_index=True)

    # Fill any remaining NaNs (e.g., labs never taken) with 0
    df_master.fillna(0, inplace=True)

    print("Saving the final processed dataset...")
    # Save as Parquet for much faster loading next time!
    df_master.to_parquet('processed_sepsis_data.parquet')

    print(f"--- Process Complete ---")
    print(f"Total rows (patient-hours): {len(df_master)}")
    print(f"Final combined shape: {df_master.shape}")
    print(df_master.head())