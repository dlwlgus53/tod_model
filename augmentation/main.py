import os, sys, json
import random, argparse
import pdb
from tqdm import tqdm
import copy 
import random

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
    random.seed(1)
    args = parse_config()
    init_experiment(args.seed)
    augmentor = None
    if args.method == 'eda':
        from methods.eda import run_eda
        augmentor = run_eda    
    elif args.method == 'aeda':
        from methods.aeda import run_aeda
        augmentor = run_aeda    
    elif args.method == 'bt':
        from methods.bt import run_bt
        augmentor = run_bt
    else:
        raise Exception('Wrong method option')
    
    org = json.load(open(args.train_data_path, "r"))
    aug = copy.deepcopy(org)

    for dial in tqdm(aug):
        for turn in dial:
            turn['dial_id'] = turn['dial_id'] + "_aug1"
            user = turn['user'].replace("<sos_u> ","").replace(" <eos_u>","").strip()
            new_user = '<sos_u> ' + augmentor(user) + ' <eos_u>'
            turn['user'] = new_user
            
    # concat org and aug        
    org_aug = org+aug
    
    # shuffle
    random.shuffle(org_aug)
    with open(args.save_data_path, 'w') as f: json.dump(org_aug, f, ensure_ascii=False, indent=4)
    