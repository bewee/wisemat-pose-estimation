import argparse
import sys
import os
import importlib
import pandas as pd
import torch
from torch.utils.data import DataLoader
from lib.dataset import PressurePoseDataset
from lib.transforms import *
from lib.F import mpjpe, per_joint_position_errors, pcp, pck, mpjpe_arms_omitted

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
    net = importlib.import_module("baseline").Baseline()

    SET = "test"
    res = dict()

    res_bodies_at_rest = evaluate(net, PressurePoseDataset(f"data/bodies-at-rest/data-{SET}_inplace.pkl", 'data/bodies-at-rest', target_transform=SkeletonCleverToCommon()))
    for key in res_bodies_at_rest.keys():
        res[f"{key} (Bodies at Rest)"] = res_bodies_at_rest[key]
    res_SLP = evaluate(net, PressurePoseDataset(f"data/SLP/data-{SET}_inplace.pkl", 'data/SLP', target_transform=SkeletonSLPToCommon()))
    for key in res_SLP.keys():
        res[f"{key} (SLP)"] = res_SLP[key]
    res_Softline = evaluate(net, PressurePoseDataset(f"data/Softline/data-{SET}_inplace.pkl", 'data/Softline'))
    for key in res_Softline.keys():
        res[f"{key} (Softline)"] = res_Softline[key]

    df = pd.DataFrame(columns = [])
    df = df.append(res, ignore_index=True)
    df.to_csv(f"experiments-baseline.csv", index=False)
