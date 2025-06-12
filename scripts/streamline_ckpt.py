import torch
import argparse
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='+', help='input checkpoint')
    # parser.add_argument('--output', type=str, help='output checkpoint', default=None)

    args = parser.parse_args()
    
    for inp in tqdm(args.input):
        try:
            sd = torch.load(inp, map_location='cpu')
            sd2 = sd['model_state_dict']
            output = f's_{inp}'
            torch.save(sd2, output)
        except Exception as e:
            print(e)