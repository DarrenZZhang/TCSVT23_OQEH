import argparse
from models import *
import tqdm
from utils import *
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from data import *
import scipy.io as sio

# parameter setting
parser = argparse.ArgumentParser()
# common
parser.add_argument('--dataset', type=str, default='nus21', choices=['coco', 'nus21'])
parser.add_argument('--nbit', type=int, default=128, choices=[16, 32, 64, 128])
parser.add_argument('--batchsize', type=int, default=128) 
parser.add_argument('--num_epoch', type=int, default=1)
parser.add_argument('--inter', type=int, default=1)
parser.add_argument('--lr', type=float, default=5e-5)

# model
parser.add_argument('--image_dim', type=int, default=4096)
parser.add_argument('--common_dim', type=int, default=128)
parser.add_argument('--nhead', type=int, default=1)
parser.add_argument('--trans_act', type=str, default='gelu')
parser.add_argument('--dropout', type=float, default=0.2)

# loss
parser.add_argument('--lamda1', type=float, default=10)
parser.add_argument('--lamda2', type=float, default=1)
parser.add_argument('--lamda_kl', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=0.3)

# justify 
parser.add_argument('--num_layer', type=int, default=4)
parser.add_argument('--topk', type=int, default=1)
parser.add_argument('--botk', type=float, default=1.0)
parser.add_argument('--gama', type=float, default=0.7)

args = parser.parse_args()
# seed_setting(seed=1)
# seed_setting(seed=2)

if args.dataset == 'coco':
    args.nclass = 80
elif args.dataset == 'nus21':
    args.nclass = 21
else:
    raise Exception('No this dataset!')

def train(args, train_loader):
    ################################
    # load pretext model
    ################################
    pretext_net = PretextNet(args)
    pretext_net.cuda()

    path_save_pretext = '../T100_pretext/checkpoints/' + args.dataset + '_pretext_model.pth.tar'
    pretext_net.load_state_dict(torch.load(path_save_pretext, map_location='cuda:0'))

    ################################
    # load nearest neighbor & indices
    ################################
    path_save_idx = '../T100_pretext/checkpoints/' + args.dataset + '_topk-train-indices.npy'
    all_knn_idx = np.load(path_save_idx)
    all_knn_idx = torch.Tensor(all_knn_idx).to(torch.long)[:, 1: (args.topk+1)]
    print(all_knn_idx.size())

    path_save_nbr = '../T100_pretext/checkpoints/' + args.dataset + '_train-all.pth'
    train_feat = torch.load(path_save_nbr)
    img_all = torch.Tensor(train_feat['I_tr'])
    print(img_all.size())
    label_all = torch.Tensor(train_feat['L_tr'])

    ################################
    # begin train
    ################################
    # 1. define modal
    fuseNet = FuseTransEncoder(args)
    fuseNet.cuda()

    # 2. define loss
    criterion_l2 = nn.MSELoss().cuda()

    # 3. define optimizer
    optimizer_fuseNet = torch.optim.Adam(fuseNet.parameters(),
                                        #  lr=1e-4,
                                        lr=args.lr,
                                         betas=(0.5, 0.999))

    # 5. train model
    pretext_net.eval()
    fuseNet.train()

    for epoch in range(args.num_epoch):
        for step, (img_, label, index) in tqdm.tqdm(enumerate(train_loader)):
        # for step, (img_, label, index) in enumerate(train_loader):
            nsample = len(index)
            img_ = img_.cuda()

            with torch.no_grad():
                img, _ = pretext_net(img_)
                # _, img = pretext_net(img_)

            # determine the neighbors
            img_knn_idx = all_knn_idx[index, :]  # n*k -> batchsize*k
            nbrs = torch.zeros((nsample, args.topk, args.image_dim))

            for i in range(nsample):
                i_knn_idx = img_knn_idx[i, :]
                nbrs[i, :, :] = img_all[i_knn_idx, :]
            nbrs = nbrs.cuda()

            # forward
            optimizer_fuseNet.zero_grad()

            # (1)
            h, img_recons = fuseNet(img, nbrs)
            b = torch.sign(h)

            # loss
            # (1) KL散度
            img_combine = args.beta * img + (1 - args.beta) * torch.mean(nbrs, dim=1)
            loss_kl = func.kl_div(img_recons.log_softmax(dim=-1),
                                img_combine.softmax(dim=-1),
                                reduction='batchmean')  * args.lamda_kl

            # (2) 关系重构loss
            img_norm = func.normalize(img, p=2, dim=1)  # default: dim=1
            S = img_norm.mm(img_norm.t())  # [0, 1]
            S_ = S - torch.eye(nsample).cuda()
            S_l1 = S_ * 2 - 1  # [-1, 1]
            S_l2 = args.gama * S_l1 + (1-args.gama) * S_l1.mm(S_l1) / S_l1.size(0)  # [-1, 1]
            simImg = S_l2 * args.botk  # [-1.2, 1.2]
            # simImg = S_l1 * args.botk  # [-1.2, 1.2]

            hash_norm = func.normalize(h)
            simHash = hash_norm.mm(hash_norm.t())  # [-1, 1]
            loss_relat = criterion_l2(simHash, simImg.cuda()) * args.lamda1

            # (2) quantization loss
            loss_sign = criterion_l2(h, b) * args.lamda2

            # total loss
            loss = loss_relat + loss_sign + loss_kl

            loss.backward()

            optimizer_fuseNet.step()

            # print log
            if (epoch + 1) % 2 == 0 and (step + 1) == len(train_loader):
                print('Epoch [%3d/%3d]: Total Loss: %.4f, loss1: %.4f, '
                      'loss2: %.4f, lamda_kl: %.4f' % (
                    epoch + 1, args.num_epoch,
                    loss.item(),
                    loss_relat.item(),
                    loss_sign.item(),
                    loss_kl.item(),
                ))

            # save
            if (epoch + 1) % args.inter == 0 and (step + 1) == len(train_loader):
                save_path = 'checkpoints/' + args.dataset + '_trained_model_' + str(args.nbit) + '_' + str(epoch + 1) +'.pth'
                torch.save(fuseNet, save_path)

    # save_path = args.dataset + '_anchor.pth'
    # torch.save(anchor, save_path)

    return fuseNet

def performance_eval(database_loader, query_loader):

    # load nearest neighbor
    path_save_nbr = '../T100_pretext/checkpoints/' + args.dataset + '_train-all.pth'
    train_feat = torch.load(path_save_nbr)
    img_all = torch.Tensor(train_feat['I_tr'])
    label_all = torch.Tensor(train_feat['L_tr'])

    for epoch in range(args.inter, args.num_epoch+1, args.inter):
        # set path
        save_path = 'checkpoints/' + args.dataset + '_trained_model_' + str(args.nbit) + '_' + str(epoch) + '.pth'
        model = torch.load(save_path)

        model.eval().cuda()
        re_BI, re_L, qu_BI, qu_L = compress(database_loader,
                                            query_loader,
                                            model,
                                            img_all,
                                            label_all,
                                            )
        ## save
        _dict = {
            'retrieval_B': re_BI,
            'L_db':re_L,
            'val_B': qu_BI,
            'L_te':qu_L,
        }
        sava_path = 'hashcode/HASH_' + args.dataset + '_' + str(args.nbit)+ 'bits_' + str(epoch)  + '.mat'
        sio.savemat(sava_path, _dict)

    return 0

def compress(database_loader, query_loader, model, img_trian, label_all):
    # load pretex model
    pretext_net = PretextNet(args)
    pretext_net.cuda()

    path_save_pretext = '../T100_pretext/checkpoints/' + args.dataset + '_pretext_model.pth.tar'
    pretext_net.load_state_dict(torch.load(path_save_pretext, map_location='cuda:0'))
    pretext_net.eval()

    # retrieval
    re_BI = list([])
    re_L = list([])

    for _, (data_I, data_L, _) in tqdm.tqdm(enumerate(database_loader)):
        with torch.no_grad():
            data_I = data_I.cuda()
            data_L = data_L.cuda()

            data_I, _ = pretext_net(data_I)
            # _, data_I = pretext_net(data_I)

            # determine the neighbors
            nsample = len(data_I)
            img_knn_idx = NeighborExtractor_asym(data_I, img_trian.cuda(), args.topk)
            nbrs = torch.zeros((nsample, args.topk, args.image_dim))
            for i in range(nsample):
                i_knn_idx = img_knn_idx[i, :]
                nbrs[i, :, :] = img_trian[i_knn_idx, :]
            nbrs = nbrs.cuda()

            # forward hashing model
            code_I, _ = model(data_I, nbrs)

        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())
        re_L.extend(data_L.cpu().data.numpy())

    # query
    qu_BI = list([])
    qu_L = list([])

    for _, (data_I, data_L, _) in enumerate(query_loader):
        with torch.no_grad():
            data_I = data_I.cuda()
            data_L = data_L.cuda()

            data_I, _ = pretext_net(data_I)
            # _, data_I = pretext_net(data_I)

            # determine the neighbors
            nsample = len(data_I)
            img_knn_idx = NeighborExtractor_asym(data_I, img_trian.cuda(), args.topk)
            nbrs = torch.zeros((nsample, args.topk, args.image_dim))
            for i in range(nsample):
                i_knn_idx = img_knn_idx[i, :]
                nbrs[i, :, :] = img_trian[i_knn_idx, :]
            nbrs = nbrs.cuda()

            # forward hashing model
            code_I, _ = model(data_I, nbrs)

        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())
        qu_L.extend(data_L.cpu().data.numpy())

    re_BI = np.array(re_BI)
    re_L = np.array(re_L)

    qu_BI = np.array(qu_BI)
    qu_L = np.array(qu_L)

    return re_BI, re_L, qu_BI, qu_L

def NeighborExtractor_asym(img, img_trian, topk):

    # cosine similarity
    img_norm = func.normalize(img, p=2, dim=1)
    img_trian_norm = func.normalize(img_trian, p=2, dim=1)
    S = img_norm.mm(img_trian_norm.t())  # [batch, 10000] [0, 1]

    # select neighbor
    topk_wgt, topk_idx = torch.topk(S, topk)

    return topk_idx

if __name__ == '__main__':

    ##################
    # load data
    ##################
    if args.dataset == 'coco':
        datahub = MSCOCO_step2_train(root="./data/coco/",
                                    img_root="/data/CuiHui/coco/raw/train2017/",
                                    batch_size=args.batchsize,
                                    num_workers=4)
        print('coco!')
    elif args.dataset == 'nus21':
        datahub = NUSWIDE_step2_train(root="./data/nus21/",
                                    img_root="/data/CuiHui/nus21/raw/",
                                    batch_size=args.batchsize,
                                    num_workers=4)
        print('nus21!')

    ##################
    # for train
    ##################
    print("begin train")
    model_trained = train(args, datahub.train_loader)

    ##################
    # load data
    ##################
    if args.dataset == 'coco':
        datahub = MSCOCO_step2_eval(root="./data/coco/",
                                    img_root="/data/CuiHui/coco/raw/train2017/",
                                    img_root2="/data/CuiHui/coco/raw/val2017/",
                                    batch_size=args.batchsize,
                                    num_workers=4)
    elif args.dataset == 'nus21':
        datahub = NUSWIDE_step2_eval(root="./data/nus21/",
                                img_root="/data/CuiHui/nus21/raw/",
                                batch_size=args.batchsize,
                                num_workers=4)       
    ##################
    # for test
    ##################
    print("begin test")
    performance_eval(datahub.database_loader, datahub.test_loader)


    ##################
    # for print
    ##################
    print('lamda1: %.8f, lamda2: %.8f, lamda_kl: %.8f'
          % (args.lamda1, args.lamda2, args.lamda_kl))
    print('alpha: %.8f, beta: %.8f'
        % (args.alpha, args.beta))
    print("****************train end **************************")