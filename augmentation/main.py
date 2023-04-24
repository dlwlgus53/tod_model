import os, sys, json
import random, argparse
from eda import run_eda
import pdb
from tqdm import tqdm

def init_experiment(seed):
    random.seed(seed)
def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration
    parser.add_argument('--method', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--save_data_path', type=str)
    parser.add_argument('--train_data_path', type=str)
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_config()
    init_experiment(args.seed)
    if args.method == 'eda':
        augmentor = run_eda
    org = json.load(open(args.train_data_path, "r"))
    for dial in tqdm(org):
        for turn in dial:
            user = turn['user'].replace("<sos_u> ","").replace(" <eos_u>","").strip()
            new_user = '<sos_u> ' + run_eda(user) + ' <eos_u>'
            turn['user'] = new_user
            
    with open(args.save_data_path, 'w') as f: json.dump(org, f, ensure_ascii=False, indent=4)
    