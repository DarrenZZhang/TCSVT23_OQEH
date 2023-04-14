import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import argparse
from model import PretextNet
from utils import *
import torch.nn.functional as func
import numpy as np
from termcolor import colored
from memory import MemoryBank, fill_memory_bank
from data import DATASET_step1
import time

# parameter setting
torch.cuda.set_device(1)
parser = argparse.ArgumentParser()
# common
parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'nus21'])
parser.add_argument('--batchsize', type=int, default=128)

# pretext
parser.add_argument('--image_dim', type=int, default=4096)
parser.add_argument('--lr', type=float, default=0.000001)
parser.add_argument('--num_epoch_pretext', type=int, default=1)
parser.add_argument('--topk_p', type=int, default=5)

args = parser.parse_args()
# seed_setting(seed=1)
seed_setting(seed=2022)

if args.dataset == 'coco':
    args.nclass = 80
    args.ntrain = 10000
elif args.dataset == 'nus21':
    args.nclass = 21
    args.ntrain = 10500
else:
    raise Exception('No this dataset!')

def train_pretext(args, train_loader, train_loader_eval):

    ################################
    # begin train
    ################################
    # 1. define modal
    pretext_net = PretextNet(args)
    pretext_net.cuda()

    # 2. define loss
    criterion_clr = SimCLRLoss(temperature=0.1).cuda()

    # 3. define optimizer
    print(colored('optimizer', 'blue'))
    params_pretext = [
        # {'params': pretext_net.vgg.features.parameters()},
        # {'params': pretext_net.vgg.classifier[0:3].parameters()},
        {'params': pretext_net.vgg.classifier[3:].parameters()},
        {'params': pretext_net.contrastive_head.parameters()},
    ]
    optimizer_pretext = torch.optim.Adam(params_pretext, lr=args.lr)
    print(optimizer_pretext)

    # Checkpoint
    checkpoint_save_path = './checkpoints/' + args.dataset + '_pretext_checkpoint.pth.tar'
    if os.path.exists(checkpoint_save_path):
        print(colored('Restart from checkpoint {}'.format(checkpoint_save_path), 'blue'))
        checkpoint = torch.load(checkpoint_save_path, map_location='cpu')
        optimizer_pretext.load_state_dict(checkpoint['optimizer'])
        pretext_net.load_state_dict(checkpoint['model'])
        pretext_net.cuda()
        start_epoch = checkpoint['epoch']
    else:
        print(colored('No checkpoint file at {}'.format(checkpoint_save_path), 'blue'))
        start_epoch = 0
        pretext_net = pretext_net.cuda()

    # Time
    start_time = time.time() * 1000

    # 4. train model
    pretext_net.train()

    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch, args.num_epoch_pretext):
        print(colored('Epoch %d/%d' %(epoch, args.num_epoch_pretext), 'yellow'))
        print(colored('-'*15, 'yellow'))

        for step, (img, label, _) in enumerate(train_loader):
            view1 = img[0].cuda()
            view2 = img[1].cuda()

            nsample, c, h, w = view1.size() # torch.Size([128, 3, 224, 224])

            # forward
            input = torch.cat([view1.unsqueeze(1), view2.unsqueeze(1)], dim=1) # [nsample, 2, c, h, w]
            input_ = input.view(-1, c, h, w) # [nsample*2, c, h, w]

            feat, output = pretext_net(input_) # [nsample*2, ndim]
            output_ = func.normalize(output, p=2, dim=1) # important!
            output_ = output_.view(nsample, 2, -1) # [nsample, 2, output_dim]

            # loss
            loss = criterion_clr(output_)

            optimizer_pretext.zero_grad()
            loss.backward()
            optimizer_pretext.step()

            # print log
            if (epoch + 1) % 2 == 0 and (step + 1) == len(train_loader):
                print('Epoch [%3d/%3d]: Loss: %.4f ' % (
                    epoch + 1, args.num_epoch_pretext, loss.item()))

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer_pretext.state_dict(), 'model': pretext_net.state_dict(),
                    'epoch': epoch + 1}, checkpoint_save_path)

    end_time = time.time() * 1000
    train_time = end_time - start_time
    print('[Train time] %.4f'  % (train_time / 1000))

    # Save final model
    pretext_save_path = './checkpoints/' + args.dataset + '_pretext_model.pth.tar'
    torch.save(pretext_net.state_dict(), pretext_save_path)

    ################################
    # begin eval
    ################################
    memory_bank_base = MemoryBank(args.ntrain, args.image_dim, args.nclass)
    memory_bank_base.cuda()

    # forward
    print('Fill Memory Bank!')
    I_tr, L_tr = fill_memory_bank(train_loader_eval, pretext_net, memory_bank_base)

    print('Training set is saved!')
    save_path = './checkpoints/' + args.dataset + '_train-all.pth'
    train_feat = {'I_tr': torch.Tensor(I_tr), 'L_tr': torch.Tensor(L_tr)}
    torch.save(train_feat, save_path)

    # mine_nearest_neighbors
    print('topk: ', args.topk_p)
    distances, indices, acc = memory_bank_base.mine_nearest_neighbors(args.topk_p)

    print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(args.topk_p, 100*acc))
    idx_save_path = './checkpoints/' + args.dataset + '_topk-train-indices.npy'
    np.save(idx_save_path, indices)

    dis_save_path = './checkpoints/' + args.dataset + '_topk-train-distances.npy'
    np.save(dis_save_path, distances)

    return pretext_net

if __name__ == '__main__':

    ##################
    # load datanclass
    ##################
    if args.dataset == 'coco':
        datahub = DATASET_step1(root="./data/coco/",
                                img_root="/data/CuiHui/coco/raw/train2017/",
                                batch_size=args.batchsize,
                                num_workers=10)
        print('coco!')
    elif args.dataset == 'nus21':
        datahub = DATASET_step1(root="./data/nus21/",
                                img_root="/data/CuiHui/nus21/raw/",
                                batch_size=args.batchsize,
                                num_workers=10) 
        print('nus21!')

    ####################
    # for train pretext
    ####################
    print("begin train pretext!")
    pretext_net = train_pretext(args, datahub.train_loader, datahub.train_loader_eval)
