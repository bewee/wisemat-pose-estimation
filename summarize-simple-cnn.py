import argparse
import sys
import os
import importlib
import pandas as pd
import torch
from tensorflow.python.summary.summary_iterator import summary_iterator
from glob import glob
from torch.utils.data import DataLoader
from lib.dataset import PressurePoseDataset
from lib.transforms import *
from lib.F import mpjpe, per_joint_position_errors, pcp, pck, mpjpe_arms_omitted

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def evaluate(net, val_set):
    val_data = next(iter(DataLoader(val_set, batch_size=len(val_set))))
    X = val_data[0]
    y = val_data[1]
    y_hat = net(X).detach()
    
    return {
        "MPJPE": mpjpe(y_hat, y).item(),
        "MPJPEAO": mpjpe_arms_omitted(y_hat, y).item(),
        "SDPJPE": torch.std(per_joint_position_errors(y_hat, y)).item(),
        "PCP": pcp(y_hat, y).item(),
        "PCK": pck(y_hat, y).item(),
    }

if __name__ == '__main__':
    parser = MyParser(description='Summarize simple-cnn results')
    parser.add_argument('v_num', type=str, help='verson number of simple-cnn to evaluate')
    parser.add_argument('description', type=str, help='description of the experiment')

    args = parser.parse_args()

    LOG_FOLDER = "tb_logs"
    MODEL = "simple-cnn"
    VERSION = f"version_{args.v_num}"
    CHECKPOINT = "best.ckpt"
    SET = "test"
    input_transforms = [MinMaxNormalize(), Boolean()]

    res = dict()
    res['description'] = args.description

    for file in glob(f"{LOG_FOLDER}/{MODEL}/{VERSION}/events.out.tfevents.*"):
        for summary in summary_iterator(file):
            try:
                tag = summary.summary.value[0].tag
                value = summary.summary.value[0].simple_value
                if tag == 'epoch':
                    res['epochs'] = value
                if tag == 'train_time':
                    res['total time'] = value
                if tag == 'train_loss':
                    res['train loss'] = value
            except:
                pass

    simple_cnn = importlib.import_module("simple-cnn")
    net = simple_cnn.Net.load_from_checkpoint(f"{LOG_FOLDER}/{MODEL}/{VERSION}/checkpoints/{CHECKPOINT}")

    res_bodies_at_rest = evaluate(net, PressurePoseDataset(f"data/bodies-at-rest/data-{SET}_inplace.pkl", 'data/bodies-at-rest', target_transform=SkeletonCleverToCommon(), input_transform=Stack(input_transforms)))
    for key in res_bodies_at_rest.keys():
        res[f"{key} (Bodies at Rest)"] = res_bodies_at_rest[key]
    res_SLP = evaluate(net, PressurePoseDataset(f"data/SLP/data-{SET}_inplace.pkl", 'data/SLP', target_transform=SkeletonSLPToCommon(), input_transform=Stack(input_transforms)))
    for key in res_SLP.keys():
        res[f"{key} (SLP)"] = res_SLP[key]
    res_Softline = evaluate(net, PressurePoseDataset(f"data/Softline/data-{SET}_inplace.pkl", 'data/Softline', input_transform=Stack(input_transforms)))
    for key in res_Softline.keys():
        res[f"{key} (Softline)"] = res_Softline[key]

    df = None
    if os.path.exists('experiments-simple-cnn.csv'):
        df = pd.read_csv('experiments-simple-cnn.csv')
    else:
        df = pd.DataFrame(columns = [])
    df = df.append(res, ignore_index=True)
    df.to_csv('experiments-simple-cnn.csv', index=False)
