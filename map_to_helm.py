import sys
import argparse
import pandas as pd
import re
import os
import warnings

warnings.filterwarnings('ignore')


# Load the MAP monomer library
df1 = pd.read_csv('data/MAP_momomers_library_new.csv')
map_to_helm_dict = df1.set_index('MAP_denotion')['Symbol'].sort_index(ascending=False).to_dict()


##MAP to HELM sequence
def process_HELM_seq(helm_seq, ID):
    if '{' in helm_seq:
        start = helm_seq.index('{')
        end = helm_seq.index('}')
        cyc_seq = helm_seq[start:end]
        seq_len = len(helm_seq[end+1:].split('.'))
        cyc_list = cyc_seq.split('-')
        start_pos = cyc_list[0][-1]
        end_pos = cyc_list[1]
       
        if start_pos == 'N' and end_pos == 'C':
            return f'PEPTIDE{ID}{{{helm_seq[end+1:]}}}$PEPTIDE{ID},PEPTIDE{ID},1:R1-{seq_len}:R2$$$'
        elif start_pos != '1' and end_pos == str(seq_len):
            return f'PEPTIDE{ID}{{{helm_seq[end+1:]}}}$PEPTIDE{ID},PEPTIDE{ID},{start_pos}:R3-{seq_len}:R2$$$'
        elif start_pos == '1' and end_pos != str(seq_len):
            return f'PEPTIDE{ID}{{{helm_seq[end+1:]}}}$PEPTIDE{ID},PEPTIDE{ID},1:R1-{end_pos}:R3$$$'
        else:
            return f'PEPTIDE{ID}{{{helm_seq[end+1:]}}}$PEPTIDE{ID},PEPTIDE{ID},{start_pos}:R3-{end_pos}:R3$$$'
    else:
        return f'PEPTIDE{ID}{{{helm_seq}}}$$$$'
def convert_map_to_helm_sequence(map_str, ID):
    nterm_pattern = r'\{nt:[^}]+\}'
    cyc_pattern = r'\{cyc:\s*([N]|\d+)-([C]|\d+)\}'
    string = ''
    nterm_modifications = re.findall(nterm_pattern, map_str)
    map_str = re.sub(nterm_pattern, '', map_str)
    cyc_string = re.search(cyc_pattern, map_str)
    # print("cyc_string",cyc_string[0])
    map_str = re.sub(cyc_pattern, '', map_str)
    if cyc_string:
        string += ''.join(cyc_string[0]) + ''.join(nterm_modifications) + map_str
    else:
        string += ''.join(nterm_modifications) + map_str

    tokens = []
    i = 0
    while i < len(string):
        matched = False
        for key in map_to_helm_dict.keys():
            if string[i:].startswith(key):
                if string[i-4:i] == 'cyc:':
                    val = map_to_helm_dict[key]
                    token = f'[{val}]' if len(val) > 1 else f'{val}'
                    tokens.append(token)
                    i += len(key)
                    matched = True
                    break
                else:
                    val = map_to_helm_dict[key]
                    token = f'[{val}].' if len(val) > 1 else f'{val}.'
                    tokens.append(token)
                    i += len(key)
                    matched = True
                    break

        if not matched:
            tokens.append(string[i])
            i += 1
    helm_seq = ''.join(tokens).rstrip('.')
    # print(helm_seq)
    return helm_seq

def main():
    parser = argparse.ArgumentParser(description='Convert MAP sequence to HELM sequence')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-s', '--sequence', type=str, help='Single MAP sequence')
    parser.add_argument('-i', '--id', type=str, help='Peptide ID (required with -s)', required=False)
    group.add_argument('-f', '--file', type=str, help='Input file with MAP sequences and IDs')
    parser.add_argument('-o', '--output', type=str, help='Output file path for HELM sequences (used with -f)')

    args = parser.parse_args()

    if args.sequence:
        if not args.id:
            parser.error("--id is required when using --sequence")
        helm_seq = convert_map_to_helm_sequence(args.sequence, args.id)
        result = process_HELM_seq(helm_seq, args.id)
        print(result)
    elif args.file:
        # Use provided output path or default to temp/helm_output_<input_filename>
        output_file = args.output if args.output else os.path.join('temp', f'helm_output_{os.path.basename(args.file)}')
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(args.file, 'r') as f, open(output_file, 'w') as out:
            for line in f:
                line = line.strip()
                if line:
                    # Expect each line to contain MAP sequence and ID separated by a comma
                    try:
                        map_str, peptide_id = line.split(',')
                        helm_seq = convert_map_to_helm_sequence(map_str.strip(), peptide_id.strip())
                        result = process_HELM_seq(helm_seq, peptide_id.strip())
                        out.write(f'{result}\n')
                    except ValueError:
                        out.write(f'Error: Invalid format in line "{line}". Expected "MAP_sequence,peptide_id"\n')
        print(f"HELM sequences written to {output_file}")

if __name__ == "__main__":
    main()
