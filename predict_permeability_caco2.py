import argparse
import pandas as pd
import joblib
import os
import sys
import numpy as np
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator, rdMolDescriptors
from padelpy import padeldescriptor
from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.stats import pearsonr, spearmanr
import lightgbm as lgb
import xgboost as xgb
import shutil


from utils import get_atomic_features, get_embeddings, get_fingerprints, get_descriptors

def parse_args():
    parser = argparse.ArgumentParser(description="Predict permeability from SMILES input.")
    parser.add_argument("--input", type=str, required=True, help="Path to input file containing SMILES.")
    parser.add_argument("--output", type=str, help="Output file path for predictions (CSV format).")
    return parser.parse_args()

def load_smiles(input_path):
    try:
        with open(input_path, "r") as f:
            smiles = [line.strip() for line in f if line.strip()]
        return smiles
    except Exception as e:
        sys.exit(f"Error reading input file: {e}")

def create_smiles_csv(smiles_list):
    ids = [f"{i+1}" for i in range(len(smiles_list))]
    df = pd.DataFrame({
        'ID': ids,
        'SMILES': smiles_list
    })
    return df

def main():
    args = parse_args()
    smiles = load_smiles(args.input)

    if not smiles:
        sys.exit("No SMILES found in input file.")

    test_df = create_smiles_csv(smiles)


    try:
        cwd = os.getcwd()
    
        smi_path = os.path.join(cwd, 'temp', 'Test_Caco2.smi')
        mol_dir = os.path.join(cwd, 'temp', 'Test_mol_Caco2')
        padel_2d_dir = os.path.join(cwd, "temp", "Test_2d_padel_Caco2.csv")
        padel_3d_dir = os.path.join(cwd, "temp", "Test_3d_padel_Caco2.csv")
        MODEL_DIR = os.path.join(cwd, 'models', 'Caco2')
        model_name = "MoLFormer-XL-both-10pct_model_1_caco2'"
        model_save_path = os.path.join(MODEL_DIR, model_name)


        # smi_path = 'temp/Test_Caco2.smi'
        # mol_dir = 'temp/Test_mol_Caco2'
        # padel_2d_dir = "temp/Test_2d_padel_Caco2.csv"
        # padel_3d_dir = "temp/Test_3d_padel_Caco2.csv"

        
        test_df_atomic = get_atomic_features(test_df)
        test_df_emb = get_embeddings(test_df,model_save_path)
        test_df_fp = get_fingerprints(test_df,smi_path)
        test_df_desc = get_descriptors(test_df,smi_path,mol_dir,padel_2d_dir, padel_3d_dir )
        # data_type = 'descriptors'
        # df = test_df_desc
        # df = df.sort_values(by='ID')
        # selected_features = joblib.load(f'{MODEL_DIR}selected_features_{data_type}.joblib')
        # columns = ['ID', 'SMILES'] + selected_features
        # print(test_df_desc[columns])

        DATA_TYPES = ['descriptors', 'fingerprints', 'embeddings', 'atomic']
        test_files = {'descriptors': test_df_desc, 'fingerprints': test_df_fp, 'embeddings': test_df_emb, 'atomic': test_df_atomic}
        test_dfs = {}
        for data_type in DATA_TYPES:
            df = test_files[data_type]
            df = df.sort_values(by='ID')
            df = df.dropna()
            # Load selected features
            selected_features = joblib.load(f'{MODEL_DIR}/selected_features_{data_type}.joblib')
            columns = ['ID', 'SMILES'] + selected_features
            df = df[columns]
            test_dfs[data_type] = df
            # print(test_dfs)
        
        scaled_dfs = {}
        for data_type in DATA_TYPES:
            if data_type == 'descriptors':
                name = 'Descriptor'
            elif data_type == 'fingerprints':
                name = 'Fingerprints'
            elif data_type == 'embeddings':
                name = 'Embeddings'
            else:
                name = 'Atomic'
            df = test_dfs[data_type]
            
            scaler = joblib.load(f'{MODEL_DIR}/scaler_{name}.joblib')  
            features = df.drop(columns=['ID', 'SMILES'] )
            scaled_features = pd.DataFrame(scaler.transform(features), columns=features.columns, index=df.index)
            scaled_dfs[data_type] = pd.concat([df[['ID', 'SMILES'] ], scaled_features], axis=1)
        
        # print(scaled_dfs)
            models_weak = [
            lgb.LGBMRegressor(),
            RandomForestRegressor(),
            GradientBoostingRegressor(),
            AdaBoostRegressor(),
            xgb.XGBRegressor(),
            ExtraTreesRegressor(),
            KNeighborsRegressor(),
            SVR(),
            MLPRegressor(),
            DecisionTreeRegressor(),
        ]

        models_meta = [
            SVR()
        ]
        n_folds=5
        meta_features_test = []
        for data_type, df_test in test_dfs.items():
            X_eval = df_test.drop(columns=['ID', 'SMILES'])
            fold_meta_features_test = np.zeros((X_eval.shape[0], len(models_weak)))
            
            for i, model_class in tqdm(enumerate(models_weak), desc=f"Predicting with weak models for {data_type}"):
                test_predictions_folds = []
                model_name = model_class.__class__.__name__
                for fold_idx in range(n_folds):
                    # Load the model
                    model = joblib.load(f'{MODEL_DIR}/weak_{data_type}_{model_name}_fold_{fold_idx}.joblib')
                    # Predict
                    test_predictions_fold = np.clip(model.predict(X_eval), -10, -4.0)
                    test_predictions_folds.append(test_predictions_fold)
                fold_meta_features_test[:, i] = np.mean(test_predictions_folds, axis=0)
            
            meta_features_test.append(fold_meta_features_test)
        meta_features_test = np.hstack(meta_features_test)
        
        for model_class in models_meta:
            model_name = model_class.__class__.__name__
            test_predictions_folds = []
            for fold_idx in range(n_folds):
                
                model = joblib.load(f'{MODEL_DIR}/meta_{model_name}_fold_{fold_idx}.joblib')
               
                test_predictions_fold = np.clip(model.predict(meta_features_test), -10, -4.0)
                test_predictions_folds.append(test_predictions_fold)
          
            final_predictions = np.mean(test_predictions_folds, axis=0)
            
        output_df = pd.DataFrame({
        'SMILES': df_test['SMILES'],
        'Permeability': final_predictions
        })
        output_file = args.output if args.output else os.path.join(cwd, 'results', 'output_caco2.csv')
        
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        output_df.to_csv(output_file, index=False)
    
        print("SMILES,Permeability")  
        for idx, row in output_df.iterrows():
            print(f"{row['SMILES']},{row['Permeability']:.2f}")
        
        print(f"Results saved to {output_file}")

    except Exception as e:
        sys.exit(f"Error during prediction: {str(e)}")
    
    finally:
        
        temp_dir = 'temp'
        if os.path.exists(temp_dir):
            try:
                for item in os.listdir(temp_dir):
                    item_path = os.path.join(temp_dir, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                
            except Exception as e:
                print(f"Warning: Could not clean up contents of {temp_dir}: {str(e)}")   


if __name__ == "__main__":
    main()
#     # Example test file paths (user should provide these)
#     test_files = {
#         'descriptors': 'features/Descriptors/Test_2d_3d_all_descriptors_RRCK.csv',
#         'fingerprints': 'features/Fingerprints/Test_All_fingerprints_test_RRCK.csv',
#         'embeddings': 'features/Embeddings/Test_MoLFormer-XL-both-10pct_model_1_fine_tuned_embeddings_RRCK.csv',
#         'atomic': 'features/Atomic/Test_all_atomic_desc_RRCK.csv'
#     }
#     predictions = main(test_files)
#     print(predictions)