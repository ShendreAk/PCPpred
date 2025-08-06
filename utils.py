import os
import subprocess
import glob
import joblib
from tqdm import tqdm 
import pandas as pd
import numpy as np
import ast
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator, rdMolDescriptors, Descriptors, Descriptors3D
from rdkit.ML.Descriptors import MoleculeDescriptors
from padelpy import padeldescriptor

def get_atomic_composition_frequency(smiles_list):
    atomic_frequencies = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        composition = {'Br': 0, 'C': 0, 'Cl': 0, 'F': 0, 'N': 0, 'O': 0, 'S': 0}  
        
        if mol is not None:  
            total_atoms = mol.GetNumAtoms() 
            
            for atom in mol.GetAtoms():
                composition[atom.GetSymbol()] += 1 
            # Calculate frequency
            frequency = {atom: count / total_atoms for atom, count in composition.items()}
            atomic_frequencies.append(frequency)  # Append the frequency for this SMILES
    
    return atomic_frequencies

#Degree of atoms
target_atoms = {'Br', 'C', 'Cl', 'F', 'N', 'O', 'S'}
def compute_target_atom_degrees(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  

    degrees_dict = {atom: 0 for atom in target_atoms}
    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        if atom_symbol in target_atoms:
            current_degree = atom.GetDegree()
            
            if current_degree > degrees_dict[atom_symbol]:
                degrees_dict[atom_symbol] = current_degree

    return degrees_dict

#Bond type
def compute_bond_types_for_cyclic_peptides(df):
    
    def compute_bond_types(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None  # Handles invalid SMILES

        bond_types = {
            'Single': 0,
            'Double': 0,
            'Triple': 0,
            'Aromatic': 0,
            'Conjugated': 0,
            'No-bond': 0
        }

     
        for bond in mol.GetBonds():
            bond_order = bond.GetBondTypeAsDouble()
            if bond_order == 1.0:
                bond_types['Single'] += 1
            elif bond_order == 2.0:
                bond_types['Double'] += 1
            elif bond_order == 3.0:
                bond_types['Triple'] += 1
            elif bond_order == 1.5:
                bond_types['Aromatic'] += 1
            elif bond_order == 1.4:
                bond_types['Conjugated'] += 1
            else:
                bond_types['No-bond'] += 1

        return bond_types

    df['Bond_Types'] = df['SMILES'].apply(compute_bond_types)

    bond_types_df = df['Bond_Types'].apply(pd.Series)

    df = pd.concat([df, bond_types_df], axis=1)

    df.drop(columns=['Bond_Types'], inplace=True)

    return df

#Formal charges
def calculate_overall_formal_charge(smiles):
   
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  

    overall_charge = 0
    
    valence_electrons = {
        'C': 4,
        'N': 5,
        'O': 6,
        'S': 6,
        'P': 5,
        'F': 7,
        'Cl': 7,
        'Br': 7,
        'I': 7,
    }

    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        valence = valence_electrons.get(atom_symbol, 0)
        non_bonding = atom.GetNumImplicitHs() 
        bonding = atom.GetDegree() * 2  # 2 electrons for each bond

        # Calculate formal charge
        formal_charge = valence - (non_bonding + bonding // 2)
        overall_charge += formal_charge

    return overall_charge

def check_aromatic_and_ring(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (None, None)  # Handle invalid SMILES

    # Check if the molecule is aromatic
    is_aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())

    # Check if the molecule contains a ring
    ring_info = mol.GetRingInfo()
    is_in_ring = ring_info.NumRings() > 0

    return (int(is_aromatic), int(is_in_ring))



def get_atomic_features(input_data):
    
    df_test = input_data
    atomic_frequencies = get_atomic_composition_frequency(df_test['SMILES'])
    frequency_df = pd.DataFrame(atomic_frequencies)
    frequency_df.fillna(0, inplace=True)
    df_test_atomic_comp = pd.concat([df_test[['ID', 'SMILES']], frequency_df], axis=1)

    df = df_test[['ID', 'SMILES']]
    df['Atom_Degrees'] = df['SMILES'].apply(compute_target_atom_degrees)
    degrees_df = df['Atom_Degrees'].apply(pd.Series)
    degrees_df.columns = [f'Degree_{atom}' for atom in target_atoms] if target_atoms else degrees_df.columns
    df_degree_atoms = pd.concat([df, degrees_df], axis=1).drop('Atom_Degrees', axis=1)

    df_bond_type = compute_bond_types_for_cyclic_peptides(df_test[['ID', 'SMILES']])

    df_formal_charge = df_test[['ID', 'SMILES']].copy()
    df_formal_charge['Overall_Formal_Charge'] = df_formal_charge['SMILES'].apply(calculate_overall_formal_charge)

    df_other = df_test[['ID', 'SMILES']].copy()
    df_other[['Is_Aromatic', 'Is_In_Ring']] = df_other['SMILES'].apply(check_aromatic_and_ring).apply(pd.Series)

    merged_df = pd.merge(df_test_atomic_comp, df_degree_atoms, on=['ID', 'SMILES'], how='inner')
    merged_df = pd.merge(merged_df, df_bond_type, on=['ID', 'SMILES'], how='inner')
    merged_df = pd.merge(merged_df, df_formal_charge, on=['ID', 'SMILES'], how='inner')
    df_atomic = pd.merge(merged_df, df_other, on=['ID', 'SMILES'], how='inner')

    return df_atomic



def get_embeddings(input_data,model_save_path):
    test_df = input_data


    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"The model directory {model_save_path} does not exist.")

    tokenizer = AutoTokenizer.from_pretrained(model_save_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_save_path, trust_remote_code=True)

    test_encodings = tokenizer(list(test_df['SMILES']), truncation=True, padding=True, max_length=325, return_tensors="pt")

    batch_size = 16 

    def generate_embeddings(encodings, batch_size):
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(encodings['input_ids']), batch_size), desc="Processing batches"):
                batch = {key: val[i:i + batch_size] for key, val in encodings.items()}
                outputs = model(**batch)
                embeddings.append(outputs.last_hidden_state)
        return torch.cat(embeddings, dim=0)
    test_embeddings = generate_embeddings(test_encodings, batch_size)
    # print('Embeddings shape: ',test_embeddings.shape)
    test_embeddings = torch.mean(test_embeddings, dim=1)
    # print(test_embeddings.shape)
    column_names = [f'x_fine_emb_MFXL{i}' for i in range(test_embeddings.shape[1])]
    embeddings_df = pd.DataFrame(data=test_embeddings, columns=column_names)
    test_emb = pd.concat([test_df, embeddings_df], axis=1)
    # print(test_emb)
    # print(test_emb.shape)
    return test_emb

def generate_count_morgan_fp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # Count Morgan fingerprint generator
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
        count_fp = mfpgen.GetCountFingerprint(mol)
        
        # Convert the count fingerprint to a dense array
        dense_fp = np.zeros((nBits,), dtype=int)
        for bit, count in count_fp.GetNonzeroElements().items():
            dense_fp[bit] = count
            
        return dense_fp
    else:
        return None

def generate_morgan_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # Create a Morgan fingerprint generator
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        
        # Generate the Morgan fingerprint as a bit vector
        morgan_fp = mfpgen.GetFingerprint(mol)
        
        # Convert the bit vector to a dense array
        dense_fp = np.zeros((2048,), dtype=int)
        for bit in range(2048):
            dense_fp[bit] = morgan_fp[bit]
        
        return dense_fp
    else:
        print("Invalid SMILES string.")
        return None


def get_fingerprints(input_data, smi_path):
    cwd = os.getcwd()
    test_df = input_data
    with open(smi_path, 'w') as f:
      for _, row in test_df.iterrows():
          smiles = row['SMILES']
          id = row['ID']
          f.write(f"{smiles} {id}\n")

    directory = 'fingerprints_xml'
    pattern = '*.xml'  

    xml_files = glob.glob(os.path.join(directory, pattern))
    xml_files.sort()
    FP_list = ['AtomPairs2DCount','AtomPairs2D','EState','Extended','Fingerprinter','Graphonly','KlekotaRothCount','KlekotaRoth','MACCS','PubChem','SubstructureCount','Substructure']
    fp = dict(zip(FP_list, xml_files))

    fingerprint = 'Substructure'
    fingerprint_output_file = ''.join(['temp/',fingerprint,'_test.csv']) 
    fingerprint_descriptortypes = fp[fingerprint]
    padeldescriptor(mol_dir=smi_path, 
                    d_file=fingerprint_output_file, 
                    descriptortypes= fingerprint_descriptortypes,
                    detectaromaticity=True,
                    standardizenitro=True,
                    standardizetautomers=True,
                    threads=2,
                    removesalt=True,
                    log=True,
                    fingerprints=True)

    df1 = pd.read_csv(fingerprint_output_file)
    df1.drop(['Name'],axis=1,inplace=True)
    df1 = pd.concat([test_df[['ID','SMILES']], df1], axis=1)
    df1.to_csv(os.path.join(cwd, 'temp','substr.csv'), index=False)
    # print(df1)

    fingerprint = 'SubstructureCount'
    fingerprint_output_file = ''.join(['temp/',fingerprint,'_test.csv']) 
    fingerprint_descriptortypes = fp[fingerprint]
    padeldescriptor(mol_dir=smi_path, 
                    d_file=fingerprint_output_file, 
                    descriptortypes= fingerprint_descriptortypes,
                    detectaromaticity=True,
                    standardizenitro=True,
                    standardizetautomers=True,
                    threads=2,
                    removesalt=True,
                    log=True,
                    fingerprints=True)

    df2 = pd.read_csv(fingerprint_output_file)
    df2.drop(['Name'],axis=1,inplace=True)
    df2 = pd.concat([test_df[['ID','SMILES']], df2], axis=1)
    df2.to_csv(os.path.join(cwd, 'temp','substrcount.csv'), index=False)
    # print(df2)

    fingerprint = 'AtomPairs2DCount'
    fingerprint_output_file = ''.join(['temp/',fingerprint,'_test.csv']) 
    fingerprint_descriptortypes = fp[fingerprint]
    padeldescriptor(mol_dir=smi_path, 
                    d_file=fingerprint_output_file, 
                    descriptortypes= fingerprint_descriptortypes,
                    detectaromaticity=True,
                    standardizenitro=True,
                    standardizetautomers=True,
                    threads=2,
                    removesalt=True,
                    log=True,
                    fingerprints=True)

    df3 = pd.read_csv(fingerprint_output_file)
    df3.drop(['Name'],axis=1,inplace=True)
    df3 = pd.concat([test_df[['ID','SMILES']], df3], axis=1)
    df3.to_csv(os.path.join(cwd, 'temp','AP2dCount.csv'), index=False)
    # print(df3)

    fingerprint = 'AtomPairs2D'
    fingerprint_output_file = ''.join(['temp/',fingerprint,'_test.csv']) 
    fingerprint_descriptortypes = fp[fingerprint]
    padeldescriptor(mol_dir=smi_path, 
                    d_file=fingerprint_output_file, 
                    descriptortypes= fingerprint_descriptortypes,
                    detectaromaticity=True,
                    standardizenitro=True,
                    standardizetautomers=True,
                    threads=2,
                    removesalt=True,
                    log=True,
                    fingerprints=True)

    df4 = pd.read_csv(fingerprint_output_file)
    df4.drop(['Name'],axis=1,inplace=True)
    df4 = pd.concat([test_df[['ID','SMILES']], df4], axis=1)
    df4.to_csv(os.path.join(cwd, 'temp','AP2d.csv'), index=False)
    # print(df4)

    fingerprint = 'EState'
    fingerprint_output_file = ''.join(['temp/',fingerprint,'_test.csv']) 
    fingerprint_descriptortypes = fp[fingerprint]
    padeldescriptor(mol_dir=smi_path, 
                    d_file=fingerprint_output_file, 
                    descriptortypes= fingerprint_descriptortypes,
                    detectaromaticity=True,
                    standardizenitro=True,
                    standardizetautomers=True,
                    threads=2,
                    removesalt=True,
                    log=True,
                    fingerprints=True)

    df5 = pd.read_csv(fingerprint_output_file)
    df5.drop(['Name'],axis=1,inplace=True)
    df5 = pd.concat([test_df[['ID','SMILES']], df5], axis=1)
    df5.to_csv(os.path.join(cwd, 'temp','Estate.csv'), index=False)
    # print(df5)

    fingerprint = 'Extended'
    fingerprint_output_file = ''.join(['temp/',fingerprint,'_test.csv']) 
    fingerprint_descriptortypes = fp[fingerprint]
    padeldescriptor(mol_dir=smi_path, 
                    d_file=fingerprint_output_file, 
                    descriptortypes= fingerprint_descriptortypes,
                    detectaromaticity=True,
                    standardizenitro=True,
                    standardizetautomers=True,
                    threads=2,
                    removesalt=True,
                    log=True,
                    fingerprints=True)

    df6 = pd.read_csv(fingerprint_output_file)
    df6.drop(['Name'],axis=1,inplace=True)
    df6 = pd.concat([test_df[['ID','SMILES']], df6], axis=1)
    df6.to_csv(os.path.join(cwd, 'temp','Extended.csv'), index=False)
    # print(df6)

    fingerprint = 'Fingerprinter'
    fingerprint_output_file = ''.join(['temp/',fingerprint,'_test.csv']) 
    fingerprint_descriptortypes = fp[fingerprint]
    padeldescriptor(mol_dir=smi_path, 
                    d_file=fingerprint_output_file, 
                    descriptortypes= fingerprint_descriptortypes,
                    detectaromaticity=True,
                    standardizenitro=True,
                    standardizetautomers=True,
                    threads=2,
                    removesalt=True,
                    log=True,
                    fingerprints=True)

    df7 = pd.read_csv(fingerprint_output_file)
    df7.drop(['Name'],axis=1,inplace=True)
    df7 = pd.concat([test_df[['ID','SMILES']], df7], axis=1)
    df7.to_csv(os.path.join(cwd, 'temp','fingerprinter.csv'), index=False)
    # print(df7)

    fingerprint = 'Graphonly'
    fingerprint_output_file = ''.join(['temp/',fingerprint,'_test.csv']) 
    fingerprint_descriptortypes = fp[fingerprint]
    padeldescriptor(mol_dir=smi_path, 
                    d_file=fingerprint_output_file, 
                    descriptortypes= fingerprint_descriptortypes,
                    detectaromaticity=True,
                    standardizenitro=True,
                    standardizetautomers=True,
                    threads=2,
                    removesalt=True,
                    log=True,
                    fingerprints=True)

    df8 = pd.read_csv(fingerprint_output_file)
    df8.drop(['Name'],axis=1,inplace=True)
    df8 = pd.concat([test_df[['ID','SMILES']], df8], axis=1)
    df8.to_csv(os.path.join(cwd, 'temp','graphonly.csv'), index=False)
    # print(df8)

    fingerprint = 'KlekotaRothCount'
    fingerprint_output_file = ''.join(['temp/',fingerprint,'_test.csv']) 
    fingerprint_descriptortypes = fp[fingerprint]
    padeldescriptor(mol_dir=smi_path, 
                    d_file=fingerprint_output_file, 
                    descriptortypes= fingerprint_descriptortypes,
                    detectaromaticity=True,
                    standardizenitro=True,
                    standardizetautomers=True,
                    threads=2,
                    removesalt=True,
                    log=True,
                    fingerprints=True)

    df9 = pd.read_csv(fingerprint_output_file)
    df9.drop(['Name'],axis=1,inplace=True)
    df9 = pd.concat([test_df[['ID','SMILES']], df9], axis=1)
    df9.to_csv(os.path.join(cwd, 'temp','KRCount.csv'), index=False)
    # print(df9)

    fingerprint = 'KlekotaRoth'
    fingerprint_output_file = ''.join(['temp/',fingerprint,'_test.csv']) 
    fingerprint_descriptortypes = fp[fingerprint]
    padeldescriptor(mol_dir=smi_path, 
                    d_file=fingerprint_output_file, 
                    descriptortypes= fingerprint_descriptortypes,
                    detectaromaticity=True,
                    standardizenitro=True,
                    standardizetautomers=True,
                    threads=2,
                    removesalt=True,
                    log=True,
                    fingerprints=True)

    df10 = pd.read_csv(fingerprint_output_file)
    df10.drop(['Name'],axis=1,inplace=True)
    df10 = pd.concat([test_df[['ID','SMILES']], df10], axis=1)
    df10.to_csv(os.path.join(cwd, 'temp','KR.csv'), index=False)
    # print(df10)

    fingerprint = 'MACCS'
    fingerprint_output_file = ''.join(['temp/',fingerprint,'_test.csv']) 
    fingerprint_descriptortypes = fp[fingerprint]
    padeldescriptor(mol_dir=smi_path, 
                    d_file=fingerprint_output_file, 
                    descriptortypes= fingerprint_descriptortypes,
                    detectaromaticity=True,
                    standardizenitro=True,
                    standardizetautomers=True,
                    threads=2,
                    removesalt=True,
                    log=True,
                    fingerprints=True)

    df11 = pd.read_csv(fingerprint_output_file)
    df11.drop(['Name'],axis=1,inplace=True)
    df11 = pd.concat([test_df[['ID','SMILES']], df11], axis=1)
    df11.to_csv(os.path.join(cwd, 'temp','MACCS.csv'), index=False)
    # print(df11)

    fingerprint = 'PubChem'
    fingerprint_output_file = ''.join(['temp/',fingerprint,'_test.csv']) 
    fingerprint_descriptortypes = fp[fingerprint]
    padeldescriptor(mol_dir=smi_path, 
                    d_file=fingerprint_output_file, 
                    descriptortypes= fingerprint_descriptortypes,
                    detectaromaticity=True,
                    standardizenitro=True,
                    standardizetautomers=True,
                    threads=2,
                    removesalt=True,
                    log=True,
                    fingerprints=True)

    df12 = pd.read_csv(fingerprint_output_file)
    df12.drop(['Name'],axis=1,inplace=True)
    df12 = pd.concat([test_df[['ID','SMILES']], df12], axis=1)
    df12.to_csv(os.path.join(cwd, 'temp','pubchem.csv'), index=False)
    # print(df12)

    dfcount_morgan = test_df.copy()
    dfcount_morgan.loc[:,'count_morgan_fp'] = dfcount_morgan['SMILES'].apply(generate_count_morgan_fp)
    # print(dfcount_morgan)
    df_count = pd.DataFrame(dfcount_morgan['count_morgan_fp'].tolist(), index=dfcount_morgan.index)
    # print(df_count)
    df13 = pd.concat([test_df, df_count.add_prefix('count_fp_')], axis=1)
    df13.to_csv(os.path.join(cwd, 'temp','morgan.csv'), index=False)
    # print( df13)
   

    dfmorgan = test_df.copy()
    dfmorgan.loc[:,'Morganfingerprints'] = dfmorgan['SMILES'].apply(generate_morgan_fingerprints)
    # print(dfmorgan)
    df_morgan = pd.DataFrame(dfmorgan['Morganfingerprints'].tolist(), index=dfmorgan.index)
    df14 = pd.concat([test_df, df_morgan.add_prefix('Morgan_fp_')], axis=1)
    df14.to_csv(os.path.join(cwd, 'temp','morgancount.csv'), index=False)
    # print(df14)


    df_first = df1.merge(df2, on=['ID', 'SMILES'], how='inner').merge(df3, on=['ID', 'SMILES'], how='inner').merge(df4, on=['ID', 'SMILES'], how='inner').merge(df5, on=['ID', 'SMILES'], how='inner').merge(df6, on=['ID', 'SMILES'], how='inner')
    df_second = df_first.merge(df7, on=['ID', 'SMILES'], how='inner').merge(df8, on=['ID', 'SMILES'], how='inner').merge(df9, on=['ID', 'SMILES'], how='inner').merge(df10, on=['ID', 'SMILES'], how='inner').merge(df11, on=['ID', 'SMILES'], how='inner')
    df_final = df_second.merge(df12, on=['ID', 'SMILES'], how='inner').merge(df13, on=['ID', 'SMILES'], how='inner').merge(df14, on=['ID', 'SMILES'], how='inner')
    # print(df_final)

    return df_final

def generate_3d_conformations(input_file, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Processing molecules"):
        parts = line.strip().split()
        if len(parts) != 2:
            continue  
        smiles, ID = parts

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles} for id: {ID}")
            continue

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        
        output_file = os.path.join(output_dir, f"{ID}.mdl")
        with open(output_file, 'w') as out:
            out.write(Chem.MolToMolBlock(mol))

    # print("3D conformations generated and saved.")

def run_3dpadel_descriptor(input_dir , output_file):
    
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    padel_dir = os.path.join(utils_dir, "padel-descriptor")
    
    padel_jar = os.path.join(padel_dir, "PaDEL-Descriptor.jar")
    
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    command = [
        "java",
        "-jar",
        padel_jar,
        "-3d",
        "-dir",
        input_dir,
        "-file",
        output_file
    ]

    try:
        result = subprocess.run(
            command,
            cwd=padel_dir,
            check=True,
            capture_output=True,
            text=True
        )
        # print("PaDEL-Descriptor executed successfully.")
        # print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running PaDEL-Descriptor: {e}")
        print(e.stderr)
    except FileNotFoundError:
        print(f"PaDEL-Descriptor.jar not found in {padel_dir}")

def run_2dpadel_descriptor(input_dir , output_file):
    
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    padel_dir = os.path.join(utils_dir, "padel-descriptor")
    
    padel_jar = os.path.join(padel_dir, "PaDEL-Descriptor.jar")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    command = [
        "java",
        "-jar",
        padel_jar,
        "-2d",
        "-dir",
        input_dir,
        "-file",
        output_file
    ]

    try:
        result = subprocess.run(
            command,
            cwd=padel_dir,
            check=True,
            capture_output=True,
            text=True
        )
        # print("PaDEL-Descriptor executed successfully.")
        # print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running PaDEL-Descriptor: {e}")
        print(e.stderr)
    except FileNotFoundError:
        print(f"PaDEL-Descriptor.jar not found in {padel_dir}")


def get_descriptors(input_data, smi_path, mol_dir,padel_2d_dir, padel_3d_dir):
    test_df = input_data
    test_df['ID'] = test_df['ID'].astype(int)
    #RDkit
    calc_2d = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Chem.Descriptors._descList])

    def calculate_2d_rdkit_descriptors(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None 
        return calc_2d.CalcDescriptors(mol)
    descriptor_data = []
    for smiles in test_df['SMILES']:
        descriptors = calculate_2d_rdkit_descriptors(smiles)
        if descriptors is not None:
            descriptor_data.append(descriptors)
        else:
            descriptor_data.append([np.nan] * len(calc_2d.GetDescriptorNames()))

    descriptor_df = pd.DataFrame(descriptor_data, columns=calc_2d.GetDescriptorNames())
    test_2d_rdkit = pd.concat([test_df[['ID','SMILES']], descriptor_df], axis=1)
    test_2d_rdkit.to_csv('temp/test_2d_rdkit.csv', index=False)

    def generate_3d_descriptors(smiles):
      
      mol = Chem.MolFromSmiles(smiles)
      if mol is None:
          print(f"Invalid SMILES: {smiles}")
          return None
      
      # Add hydrogens
      mol = Chem.AddHs(mol)
      
      # Generate 3D coordinates for the molecule
      AllChem.EmbedMolecule(mol)
      
      try:
          descriptors = Descriptors3D.CalcMolDescriptors3D(mol)
          return descriptors
      except Exception as e:
          print(f"Error calculating descriptors for SMILES '{smiles}': {e}")
          return None
      
    descriptor_data = []
    for smiles in tqdm(test_df['SMILES'],desc='3d_descriptors', unit='smiles'):
        descriptors =generate_3d_descriptors(smiles) 
        if descriptors is not None:
            descriptor_data.append(descriptors)
        else:
            descriptor_data.append({'PMI1': np.nan,
      'PMI2': np.nan,
      'PMI3': np.nan,
      'NPR1': np.nan,
      'NPR2': np.nan,
      'RadiusOfGyration': np.nan,
      'InertialShapeFactor': np.nan,
      'Eccentricity': np.nan,
      'Asphericity': np.nan,
      'SpherocityIndex': np.nan,
      'PBF': np.nan})
    descriptor_df = pd.DataFrame(descriptor_data)
    num_columns = descriptor_df.shape[1]
    descriptor_df.columns = [f'3d_rdkit_{i+1}' for i in range(num_columns)]
    test_3d_rdkit = pd.concat([test_df[['ID','SMILES']], descriptor_df], axis=1)
    # print(test_3d_rdkit)
    test_3d_rdkit.to_csv('temp/test_3d_rdkit.csv', index=False)

    # Mordred Descriptors
    from mordred import Calculator, descriptors
    calc_mor = Calculator(descriptors, ignore_3D=True)  # Use all available Mordred 2D descriptors

    def calculate_mordred_descriptors(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return calc_mor(mol)
        except Exception as e:
            print(f"Error processing SMILES {smiles} for Mordred descriptors: {e}")
            return None

    descriptor_data = []
    for smiles in test_df['SMILES']:
        descriptors = calculate_mordred_descriptors(smiles)
        if descriptors is not None:
            descriptor_data.append(descriptors)
        else:
            descriptor_data.append([np.nan] * len(calc_mor.descriptors))

    # Convert Mordred results to DataFrame
    descriptor_df = pd.DataFrame(descriptor_data, columns=[desc.__class__.__name__ for desc in calc_mor.descriptors])
    test_mordred_2d = pd.concat([test_df[['ID', 'SMILES']], descriptor_df], axis=1)
    test_mordred_2d.to_csv('temp/test_2d_mordred.csv', index=False)

    
    generate_3d_conformations(smi_path, mol_dir)

    utils_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(utils_dir, mol_dir)
    output_file2d = os.path.join(utils_dir, padel_2d_dir)
    output_file3d = os.path.join(utils_dir, padel_3d_dir)
    
    # output_file = r"temp/Test_3d_padel_RRCK.csv"
    run_3dpadel_descriptor(input_dir, output_file3d)
    run_2dpadel_descriptor(input_dir, output_file2d)

    df_2d_padel = pd.read_csv(output_file2d)
    df_2d_padel['ID'] = df_2d_padel['Name'].str.extract(r'_(\d+)$')
    df_2d_padel['ID'] = df_2d_padel['ID'].astype(int)
    df_2d_padel = df_2d_padel.drop('Name',axis=1)
    df_2d_padel = df_2d_padel.fillna(0)
    merged_df = df_2d_padel.merge(test_mordred_2d[['ID', 'SMILES']], on='ID', how='left')
    merged_df = merged_df[['ID', 'SMILES'] + [col for col in merged_df.columns if col not in ['ID', 'SMILES']]]
    df_ordered = merged_df.merge(test_mordred_2d[['ID']], on='ID', how='right')
    test_2d_padel = df_ordered.reindex(test_mordred_2d.index)
    # print(test_2d_padel)
    test_2d_padel.to_csv('temp/test_2d_padel_curated.csv', index=False)

    df_3d_padel = pd.read_csv(output_file3d)
    df_3d_padel['ID'] = df_3d_padel['Name'].str.extract(r'_(\d+)$')
    df_3d_padel['ID'] = df_3d_padel['ID'].astype(int)
    df_3d_padel = df_3d_padel.drop('Name',axis=1)
    df_3d_padel = df_3d_padel.fillna(0)
    merged_df = df_3d_padel.merge(test_mordred_2d[['ID', 'SMILES']], on='ID', how='left')
    merged_df = merged_df[['ID', 'SMILES'] + [col for col in merged_df.columns if col not in ['ID', 'SMILES']]]
    df_ordered = merged_df.merge(test_mordred_2d[['ID']], on='ID', how='right')
    test_3d_padel = df_ordered.reindex(test_mordred_2d.index)
    # print(test_3d_padel)
    test_3d_padel.to_csv('temp/test_3d_padel_curated.csv', index=False)

    test_2d_rdkit = pd.read_csv('temp/test_2d_rdkit.csv')
    test_3d_rdkit = pd.read_csv('temp/test_3d_rdkit.csv')
    test_2d_padel = pd.read_csv('temp/test_2d_padel_curated.csv')
    test_3d_padel = pd.read_csv('temp/test_3d_padel_curated.csv')
    test_mordred_2d = pd.read_csv('temp/test_2d_mordred.csv')


    df_2d_test = test_2d_rdkit.merge(test_mordred_2d, on=['ID', 'SMILES'], how='inner').merge(test_2d_padel, on=['ID', 'SMILES'], how='inner')
    df_2d_test.to_csv('temp/test_2d_all_desc.csv', index=False)
    df_3d_test = test_3d_rdkit.merge(test_3d_padel, on=['ID', 'SMILES'], how='inner')
    df_3d_test.to_csv('temp/test_3d_all_desc.csv', index=False)

    df_2d_test = pd.read_csv('temp/test_2d_all_desc.csv')
    df_3d_test = pd.read_csv('temp/test_3d_all_desc.csv')

    test_desc = df_2d_test.merge(df_3d_test, on=['ID', 'SMILES'], how='inner')
    # print(test_desc)
    # print(test_desc.shape)
    test_desc.to_csv('temp/all_2d_3d_desc.csv', index=False)

    return test_desc

if __name__ == "__main__":
  
  # get_descriptors('temp/Test_RRCK.smi')
  pass




    
