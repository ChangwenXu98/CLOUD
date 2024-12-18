import json
import os
import numpy as np
import pandas as pd
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.core.structure import Structure, Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm
import argparse
from multiprocessing import Pool
import glob
from time import perf_counter

def generate_seq(struct, gen_str, wyckoff_multiplicity_dict):
    analyzer = SpacegroupAnalyzer(struct)
    symm_dataset = analyzer.get_symmetry_dataset()
    wyckoff_positions = symm_dataset['wyckoffs']

    spg_num = str(analyzer.get_space_group_number())
    seq = " ".join(gen_str[spg_num])

    wyckoff_ls = []
    for i in range(len(wyckoff_positions)):
        multiplicity = wyckoff_multiplicity_dict[spg_num][wyckoff_positions[i]]
        wyckoff_symbol = multiplicity + wyckoff_positions[i]
        if wyckoff_symbol not in wyckoff_ls:
            wyckoff_ls.append(wyckoff_symbol)
    seq = seq + ' | ' + ' '.join(wyckoff_ls)

    comp_ls = []
    for element, ratio in struct.composition.fractional_composition.get_el_amt_dict().items():
        ratio = str(np.round(ratio, 2))
        comp_ls.append(element)
        comp_ls.append(ratio)
   
    seq = seq + ' | ' + ' '.join(comp_ls)

    return seq

def process_cif(file_path):
    try:
        struct = Structure.from_file(file_path)
        # Replace with your custom conversion function
        string_representation = generate_seq(struct, gen_str, wyckoff_multiplicity_dict)
        return string_representation
    except:
        return None

def save_results(results, output_file):
    df = pd.DataFrame(results, columns=["gen_str"])
    df.to_csv(output_file, mode='a', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get cloud from cif")
    parser.add_argument("dir", help="Path to the cif file")
    parser.add_argument("out", help="Path to save cloud csv")
    parser.add_argument("numproc", help="number of processes")
    parser.add_argument("batchsize", help="batch size")
    args = parser.parse_args()

    with open("data/wyckoff-position-multiplicities.json") as file:
        # dictionary mapping Wyckoff letters in a given space group to their multiplicity
        wyckoff_multiplicity_dict = json.load(file)

    with open('data/generator.json', 'r') as fp:
        gen_str = json.load(fp)

    cif_files = glob.glob(os.path.join(args.dir, '*.cif'))
    num_batches = len(cif_files) // int(args.batchsize) + 1
    # t1_start = perf_counter() 
    # count = 0
    with Pool(int(args.numproc)) as pool:
        for batch_num in range(num_batches):
            batch_files = cif_files[batch_num * int(args.batchsize):(batch_num + 1) * int(args.batchsize)]
            results = pool.map(process_cif, batch_files)
            # count += 1
            save_results(results, args.out)
            print(f"Processed batch {batch_num + 1}/{num_batches}")

    # t1_stop = perf_counter()

    # print("Throughput:", count * int(args.batchsize) / (t1_stop - t1_start))

