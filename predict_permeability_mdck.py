import argparse
import pandas as pd
import joblib
import os
import sys
import numpy as np
import glob
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator, rdMolDescriptors
from padelpy import padeldescriptor
import shutil

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
cwd = os.getcwd() 

directory = os.path.join(cwd,'fingerprints_xml')
pattern = '*.xml'  
print(directory)
xml_files = glob.glob(os.path.join(directory, pattern))

FP_list = ['AtomPairs2DCount',
 'AtomPairs2D',
 'EState',
 'Extended',
 'Fingerprinter',
 'Graphonly',
 'KlekotaRothCount',
 'KlekotaRoth',
 'MACCS',
 'PubChem',
 'SubstructureCount',
 'Substructure']

fp = dict(zip(FP_list, xml_files))


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

    try:
        test_df = create_smiles_csv(smiles)
  
        
        with open(os.path.join(cwd, 'temp','Test_MDCK.smi'), 'w') as f:
            for _, row in test_df.iterrows():
                smiles = row['SMILES']
                id = row['ID']
                f.write(f"{smiles} {id}\n")

        # KlekotaRoth fingerprint
        fingerprint = 'KlekotaRoth'
        fingerprint_output_file = ''.join([cwd,'/temp/', fingerprint, '_test_mdck.csv']) 
        
        fingerprint_descriptortypes = os.path.join(cwd, 'fingerprints_xml','KlekotaRothFingerprinter.xml')
        
        padeldescriptor(mol_dir= os.path.join(cwd, 'temp','Test_MDCK.smi'), 
                        d_file=fingerprint_output_file, 
                        descriptortypes=fingerprint_descriptortypes,
                        detectaromaticity=True,
                        standardizenitro=True,
                        standardizetautomers=True,
                        threads=2,
                        removesalt=True,
                        log=True,
                        fingerprints=True)

        df = pd.read_csv(fingerprint_output_file)
        df.drop(['Name'], axis=1, inplace=True)
    

        models_dir = os.path.join(cwd, 'models', 'MDCK')

        
        scaler_path = os.path.join(cwd, 'models', 'MDCK','scaler_KR_fp_mdck.joblib')                          
        model_base_name = 'AdaBoostRegressor'                   
        n_folds = 5   

        X_test = df.drop(columns=['ID', 'SMILES'], errors='ignore')
   
        scaler = joblib.load(scaler_path)
        X_new_scaled = scaler.transform(X_test)
        X_new_scaled = pd.DataFrame(X_new_scaled, columns=X_test.columns, index=df.index)

        all_fold_preds = []

        for fold in range(1, n_folds + 1):
            fold_model_path = os.path.join(models_dir, f"{model_base_name}_fold{fold}_mdck.joblib")
            fold_model = joblib.load(fold_model_path)
            preds = fold_model.predict(X_new_scaled)
            preds = np.clip(preds, -10, -4.0)  
            all_fold_preds.append(preds)

        all_fold_preds = np.array(all_fold_preds)
        final_predictions = np.mean(all_fold_preds, axis=0)

        output_df = pd.DataFrame({
            'SMILES': test_df['SMILES'],
            'Permeability': final_predictions
        })

        output_file = args.output if args.output else os.path.join(cwd, 'results', 'output_mdck.csv')
        # print(output_df)
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
