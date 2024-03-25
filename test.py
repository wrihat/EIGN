

import os
import time
import math
import argparse
import random
import numpy as np
import paddle
import paddle.nn.functional as F
from pgl.utils.data import Dataloader
from dataset import *
from model import *
from utils import rmse, mae, sd, pearson
from tqdm import tqdm

def test(model_path, model_name, net, tst_loader, other_test_loader):
    print("......................................................start testing ............................................................")
    a = paddle.load(os.path.join(model_path, model_name))
    net.set_state_dict(a['model'])
    print('epoch:')
    print(a['epoch'])

    RMSE_test, MAE_test, SD_test, R_test = evaluate(net, tst_loader, args) 
    log = 'Test: RMSE: %.4f, MAE: %.4f, SD: %.4f, R: %.4f.\n' % (RMSE_test, MAE_test, SD_test, R_test)
    print('The Performance on PDBbind:')
    print(log)

    RMSE_test, MAE_test, SD_test, R_test = evaluate(net, other_test_loader, args)
    log = 'Test: RMSE: %.4f, MAE: %.4f, SD: %.4f, R: %.4f.\n' % (RMSE_test, MAE_test, SD_test, R_test)
    print('The Performance on CSAR_NRC_HiQ:')
    print(log)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pdbbind2016_general')
    parser.add_argument('--another_test_dataset', type=str, default='csar_nrc_hiq')
    parser.add_argument("--input_dim", type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--model_name', type=str, default='EIGN')
    parser.add_argument('--trained_model', type=str, default='EIGN_trained_model')
    parser.add_argument('--model_saved_dir', type=str, default='./trained_model')
    parser.add_argument('--dataset_dir', type=str, default='./dataset/')
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_egcl", type=int, default=2)
    parser.add_argument("--num_deal", type=int, default=2)
    args = parser.parse_args()
    paddle.set_device('gpu:%s' % args.cuda) if int(args.cuda) != -1 else paddle.set_device('cpu')
    if args.model_name == 'EIGN':
        model = EIGN(args)
    test_loader = ''
    valid_loader = ''
    test_complex = InteractGraphDataset(args.dataset_dir, "%s_test" % args.dataset)
    test_loader = Dataloader(test_complex, args.batch_size, shuffle=False, collate_fn=collate_fn)
    another_test_complex = InteractGraphDataset(args.dataset_dir, "%s_test" % args.another_test_dataset)
    another_test_loader = Dataloader(another_test_complex, args.batch_size, shuffle=False, collate_fn=collate_fn)
    test(args.model_saved_dir, args.trained_model, model, test_loader, another_test_loader)  

