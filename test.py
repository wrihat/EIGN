
import os
import time
import math
import argparse
import random
import numpy as np

import paddle
import paddle.nn.functional as F
from pgl.utils.data import Dataloader
from dataset_egcl import GraphDataset, collate_fn
from models import *
from models_sign import *
from utils import rmse, mae, sd, pearson
from tqdm import tqdm


def test(model_path, model_name, net, tst_loader, other_test_loader):

    print('......................................................start testing ............................................................')
    a = paddle.load(os.path.join(model_path, model_name))
    net.set_state_dict(a['model'])
    print('epoch:')
    print(a['epoch'])

    RMSE_test, MAE_test, SD_test, R_test = evaluate(net, tst_loader, args)  # 在测试集上评估模型的rmse、mae、ad、r
    log = 'Test: RMSE: %.4f, MAE: %.4f, SD: %.4f, R: %.4f.\n' % (RMSE_test, MAE_test, SD_test, R_test)
    print('测试集上模型的性能：')
    print(log)
    RMSE_test, MAE_test, SD_test, R_test = evaluate(net, other_test_loader, args)  # 在测试集上评估模型的rmse、mae、ad、r
    log = 'Test: RMSE: %.4f, MAE: %.4f, SD: %.4f, R: %.4f.\n' % (RMSE_test, MAE_test, SD_test, R_test)
    print('CSAR_NRC_HiQ测试集上模型的性能：')
    print(log)

def setup_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

@paddle.no_grad()
def evaluate(model, loader, args):
    model.eval()
    y_pred_list = []
    y_list = []
    if args.train_model_name == 'IAGN' or args.train_model_name == 'GCN_DI_LI'or args.train_model_name == 'EIGN' or args.train_model_name == 'SIGN'or args.train_model_name == 'EGCL_LI':

        for batch_data in loader:
            # atom_g, bond2atom_g, bond_angle_g_list, global_feat, edge_type_l, inter_type_counts, y = batch_data
            # y_pred, _ = model(atom_g, bond2atom_g, bond_angle_g_list, global_feat, edge_type_l, inter_type_counts)
            atom_g, bond2atom_g, global_feat, edge_type_l, inter_type_counts, y = batch_data
            y_pred, _ = model(atom_g, bond2atom_g, global_feat, edge_type_l, inter_type_counts)
            y_pred_list += y_pred.tolist()
            y_list += y.tolist()
    else:
        for batch_data in loader:
            atom_g, bond2atom_g, global_feat, edge_type_l, inter_type_counts, y = batch_data
            y_pred = model(atom_g, bond2atom_g, global_feat, edge_type_l, inter_type_counts)
            y_pred_list += y_pred.tolist()
            y_list += y.tolist()
    y_hat = np.array(y_pred_list).reshape(-1,)
    y = np.array(y_list).reshape(-1,)
    return rmse(y, y_hat), mae(y, y_hat), sd(y, y_hat), pearson(y, y_hat)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=0)  # 训练模式为1, 测试模式为0
    parser.add_argument('--dataset', type=str, default='pdbbind2016_general')
    parser.add_argument('--another_test_dataset', type=str, default='csar_nrc_hiq_type_save')
    parser.add_argument('--train_model_name', type=str, default='EIGN')
    parser.add_argument("--input_dim", type=int, default=38)
    parser.add_argument('--logs_dir', type=str, default='./running_logs/results_logs/')
    parser.add_argument('--batch_size', type=int, default=45)
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument("--lamda", type=float, default=1.75)
    parser.add_argument('--test_model_name', type=str, default='pdbbind2016_general_EIGN_v1_3_model')
    parser.add_argument('--num_epoch', type=int, default=450)
    parser.add_argument('--model_saved_dir', type=str, default='./saved_model')
    parser.add_argument('--dataset_dir', type=str, default='./dataset/')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr_decay_rate", type=float, default=0.5)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_egcl", type=int, default=2)
    parser.add_argument("--num_deal", type=int, default=2)


    # 模型的参数
    args = parser.parse_args()
    if args.random_seed:
        setup_seed(args.random_seed)
    if not os.path.isdir(args.model_saved_dir):
        os.mkdir(args.model_saved_dir)

    paddle.set_device('gpu:%s' % args.cuda) if int(args.cuda) != -1 else paddle.set_device('cpu')
    if args.train_model_name == 'EIGN':
        model = EIGN(args)
    elif args.train_model_name == 'EIGN':
        model = EIGN(args)
    elif args.train_model_name == 'EGCL_DEAL':
        model = EGCL_DEAL(args)
    elif args.train_model_name == 'GAT':
        model = GAT(args)
    elif args.train_model_name == 'GIN':
        model = GIN(args)
    elif args.train_model_name == 'SIGN':
        model = SIGN(args)
    elif args.train_model_name == 'EGNN':
        model = EGNN(args)
    else:
        model = GCN(args)

    train_loader = ''
    test_loader = ''
    valid_loader = ''

    if args.mode == 1:  # 如果是训练模式的话
        # 训练模型
        # a = paddle.load(os.path.join(args.model_saved_dir, args.test_model_name))
        # model.set_state_dict(a['model'])
        test_complex = GraphDataset(args.dataset_dir, "%s_test" % args.dataset)
        test_loader = Dataloader(test_complex, args.batch_size, shuffle=False, collate_fn=collate_fn)
        valid_complex = GraphDataset(args.dataset_dir, "%s_val" % args.dataset)
        train_complex = GraphDataset(args.dataset_dir, "%s_train" % args.dataset)
        valid_loader = Dataloader(valid_complex, args.batch_size, shuffle=False, collate_fn=collate_fn)
        train_loader = Dataloader(train_complex, args.batch_size, shuffle=True, collate_fn=collate_fn)
        another_test_complex = GraphDataset(args.dataset_dir, "%s_test" % args.another_test_dataset)
        another_test_loader = Dataloader(another_test_complex, args.batch_size, shuffle=False, collate_fn=collate_fn)
        train(args, model, train_loader, test_loader, valid_loader, another_test_loader)

    else:
        print(".............................测试模型.............................")
        test_complex = GraphDataset(args.dataset_dir, "%s_test" % args.dataset)
        test_loader = Dataloader(test_complex, args.batch_size, shuffle=False, collate_fn=collate_fn)
        another_test_complex = GraphDataset(args.dataset_dir, "%s_test" % args.another_test_dataset)
        another_test_loader = Dataloader(another_test_complex, args.batch_size, shuffle=False, collate_fn=collate_fn)
        test(args.model_saved_dir, args.test_model_name, model, test_loader, another_test_loader)  # 测试查看模型的参数



