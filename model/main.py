import argparse
from torch.utils.data import DataLoader
import glob
from torchvision import transforms
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
import socket
from datetime import datetime
import os
import torch
from tensorboardX import SummaryWriter
import tqdm
import torch.nn as nn
from torch.nn import functional as F
import math
import torch.backends.cudnn as cudnn
from torch.nn import init
from torch.autograd import Variable

ROOT_path = os.path.abspath('D:\PyCharm\PyCode\Hemangioma\Data')

class DefaultConfig(object):
    num_epochs = 10
    epoch_start_i = 0
    checkpoint_step = 5
    validation_step = 1
    crop_height = 256
    crop_width = 448
    batch_size = 2
    # 训练集所在位置
    data = r'D:/PyCharm/PyCode/Hemangioma/Data/train_dataset/'

    log_dirs = os.path.join(ROOT_path, 'Log/OCT')

    lr = 0.0001
    lr_mode = 'poly'
    net_work = 'UNet'
    # net_work= 'MSSeg'  #net_work= 'BaseNet'

    momentum = 0.9  #
    weight_decay = 1e-08  #

    mode = 'train'
    num_classes = 2

    k_fold = 10
    test_fold = 4
    num_workers = 0

    cuda = '0'
    use_gpu = True
    save_model_path = './checkpoints'


from torch.autograd import Variable


class DiceLoss(nn.Module):
    def __init__(self, smooth=0.01):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        input = torch.sigmoid(input)
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        intersect = (input * target).sum()
        union = torch.sum(input) + torch.sum(target)
        Dice = (2 * intersect + self.smooth) / (union + self.smooth)
        dice_loss = 1 - Dice
        return dice_loss

class Data(torch.utils.data.Dataset):
    Unlabelled = [0, 0, 0]
    sick = [255, 255, 255]
    COLOR_DICT = np.array([Unlabelled, sick])

    def __init__(self, dataset_path, scale=(320, 320), mode='train'):
        super().__init__()
        self.mode = mode
        self.img_path = dataset_path + '/img'
        self.mask_path = dataset_path + '/mask'
        self.image_lists, self.label_lists = self.read_list(self.img_path)
        self.resize = scale
        self.flip = iaa.SomeOf((2, 5), [
            iaa.PiecewiseAffine(scale=(0, 0.1), nb_rows=4, nb_cols=4, cval=0),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.1),
            iaa.Affine(rotate=(-20, 20),
                       scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
            iaa.OneOf([
                iaa.GaussianBlur((0, 1.0)),  # blur images with a sigma between 0 and 3.0
                iaa.AverageBlur(k=(3, 5)),  # blur image using local means with kernel sizes between 2 and 7
                iaa.MedianBlur(k=(3, 5)),  # blur image using local medians with kernel sizes between 2 and 7
            ]),
            iaa.contrast.LinearContrast((0.5, 1.5))],
                               random_order=True)

        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        # load image and crop
        img = Image.open(self.image_lists[index]).convert('RGB')
        img = img.resize(self.resize)
        img = np.array(img)
        labels = self.label_lists[index]
        # load label
        if self.mode != 'test':
            label_ori = Image.open(self.label_lists[index]).convert('RGB')
            label_ori = label_ori.resize(self.resize)
            label_ori = np.array(label_ori)
            label = np.ones(shape=(label_ori.shape[0], label_ori.shape[1]), dtype=np.uint8)

            # convert RGB  to one hot

            for i in range(len(self.COLOR_DICT)):
                equality = np.equal(label_ori, self.COLOR_DICT[i])
                class_map = np.all(equality, axis=-1)
                label[class_map] = i

            # augment image and label
            if self.mode == 'train':
                seq_det = self.flip.to_deterministic()  # 固定变换
                segmap = ia.SegmentationMapsOnImage(label, shape=label.shape)
                img = seq_det.augment_image(img)
                label = seq_det.augment_segmentation_maps([segmap])[0].get_arr().astype(np.uint8)

            label_img = torch.from_numpy(label.copy()).float()
            if self.mode == 'val':
                img_num = len(os.listdir(os.path.dirname(labels)))
                labels = label_img, img_num
            else:
                labels = label_img
        imgs = img.transpose(2, 0, 1) / 255.0
        img = torch.from_numpy(imgs.copy()).float()  # self.to_tensor(img.copy()).float()
        return img, labels

    def __len__(self):
        return len(self.image_lists)

    def read_list(self, image_path):
        fold = os.listdir(image_path)
        # fold = sorted(os.listdir(image_path), key=lambda x: int(x[-2:]))
        # print(fold)

        img_list = []
        label_list = []
        if self.mode == 'train':
            for item in fold:
                name = item.split('.')[0]
                img_list.append(os.path.join(image_path, item))
                label_list.append(os.path.join(image_path.replace('img', 'mask'), '{}.png'.format(name)))


        elif self.mode == 'val':
            for item in fold:
                name = item.split('.')[0]
                img_list.append(os.path.join(image_path, item))
                label_list.append(os.path.join(image_path.replace('img', 'mask'), '{}.png'.format(name)))


        elif self.mode == 'test':
            for item in fold:
                name = item.split('.')[0]
                img_list.append(os.path.join(image_path, item))
                label_list.append(os.path.join(image_path.replace('img', 'mask'), '{}.png'.format(name)))

        return img_list, label_list


import shutil
import os.path as osp


def save_checkpoint(state, best_pred, epoch, is_best, checkpoint_path, filename='./checkpoint/checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, osp.join(checkpoint_path, 'best.pth'))
    # shutil.copyfile(filename, osp.join(checkpoint_path, 'model_ep{}.pth'.format(epoch+1)))


def adjust_learning_rate(opt, optimizer, epoch):
    if opt.lr_mode == 'step':
        lr = opt.lr * (0.1 ** (epoch // opt.step))
    elif opt.lr_mode == 'poly':
        lr = opt.lr * (1 - epoch / opt.num_epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(opt.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def one_hot_it(label, label_info):
    # return semantic_map -> [H, W, num_classes]
    semantic_map = []
    for info in label_info:
        color = label_info[info]
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def compute_score(predict, target, forground=1, smooth=1):
    score = 0
    count = 0
    target[target != forground] = 0
    predict[predict != forground] = 0
    assert (predict.shape == target.shape)
    overlap = ((predict == forground) * (target == forground)).sum()  # TP
    union = (predict == forground).sum() + (target == forground).sum() - overlap  # FP+FN+TP
    FP = (predict == forground).sum() - overlap  # FP
    FN = (target == forground).sum() - overlap  # FN
    TN = target.shape[0] * target.shape[1] - union  # TN

    # print('overlap:',overlap)
    dice = (2 * overlap + smooth) / (union + overlap + smooth)

    precsion = ((predict == target).sum() + smooth) / (target.shape[0] * target.shape[1] + smooth)

    jaccard = (overlap + smooth) / (union + smooth)

    Sensitivity = (overlap + smooth) / ((target == forground).sum() + smooth)

    Specificity = (TN + smooth) / (FP + TN + smooth)

    return dice, precsion, jaccard, Sensitivity, Specificity

def eval_seg(predict, target, forground=1):
    pred_seg = torch.round(torch.sigmoid(predict)).int()
    pred_seg = pred_seg.data.cpu().numpy()
    label_seg = target.data.cpu().numpy().astype(dtype=int)
    assert (pred_seg.shape == label_seg.shape)

    Dice = []
    True_label = []
    TP = FPN = acc = 0

    # 计算前景类别的 Dice 系数和其他指标
    foreground_overlap = ((pred_seg == forground) * (label_seg == forground)).sum()
    foreground_union = (pred_seg == forground).sum() + (label_seg == forground).sum()
    foreground_dice = (2 * foreground_overlap + 0.1) / (foreground_union + 0.1)
    Dice.append(foreground_dice)
    True_label.append((label_seg == forground).sum())
    TP += foreground_overlap
    FPN += foreground_union

    # 计算背景类别的 Dice 系数和其他指标
    background_overlap = ((pred_seg == 0) * (label_seg == 0)).sum()
    background_union = (pred_seg == 0).sum() + (label_seg == 0).sum()
    background_dice = (2 * background_overlap + 0.1) / (background_union + 0.1)
    Dice.append(background_dice)
    True_label.append((label_seg == 0).sum())
    TP += background_overlap
    FPN += background_union

    # 计算准确率
    acc = (pred_seg == label_seg).sum() / (pred_seg.shape[0] * pred_seg.shape[1] * pred_seg.shape[2])

    # 计算全局 Dice 系数
    global_dice = 2 * TP / (FPN + 0.1)

    return Dice, True_label, acc, global_dice

# 请注意，compute_score 函数需要实现 Dice、Precision 和 Jaccard 的计算逻辑。
# 如果您需要这些指标，请确保 compute_score 函数完整并正确实现。

#模型训练
def val(args, model, dataloader):
    print('\n')
    print('Start Validation!')
    with torch.no_grad():
        model.eval()
        tbar = tqdm.tqdm(dataloader, desc='\r')

        total_Dice = [[] for _ in range(args.num_classes)]  # 初始化为正确的类别数量
        total_Dice1 = []
        total_Dice2 = []
        total_Dice.append(total_Dice1)
        total_Dice.append(total_Dice2)
        Acc = []

        cur_cube = []
        cur_label_cube = []
        next_cube = []
        counter = 0
        end_flag = False

        for i, (data, labels) in enumerate(tbar):
            # tbar.update()
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = labels[0].cuda()
            slice_num = labels[1][0].long().item()

            # get RGB predict image

            predicts = model(data)

            predict = torch.argmax(predicts, dim=1)
            batch_size = predict.size()[0]

            counter += batch_size
            if counter <= slice_num:
                cur_cube.append(predict)
                cur_label_cube.append(label)
                if counter == slice_num:
                    end_flag = True
                    counter = 0
            else:
                last = batch_size - (counter - slice_num)

                last_p = predict[0:last]
                last_l = label[0:last]

                first_p = predict[last:]
                first_l = label[last:]

                cur_cube.append(last_p)
                cur_label_cube.append(last_l)
                end_flag = True
                counter = counter - slice_num

            if end_flag:
                end_flag = False
                predict_cube = torch.stack(cur_cube, dim=0).squeeze()
                label_cube = torch.stack(cur_label_cube, dim=0).squeeze()
                cur_cube = []
                cur_label_cube = []
                if counter != 0:
                    cur_cube.append(first_p)
                    cur_label_cube.append(first_l)

                assert predict_cube.size()[0] == slice_num
                Dice, true_label, acc, mean_dice = eval_seg(predict_cube, label_cube, args.num_classes)

                for class_id in range(args.num_classes):
                    if true_label[class_id] != 0:
                        total_Dice[class_id].append(Dice[class_id])
                Acc.append(acc)
                # 计算 Dice 系数时修正
                dice1 = sum(total_Dice[0]) / len(total_Dice[0]) if len(total_Dice[0]) != 0 else 0
                dice2 = sum(total_Dice[1]) / len(total_Dice[1]) if len(total_Dice[1]) != 0 else 0
                ACC = sum(Acc) / len(Acc)
                tbar.set_description('Mean_D: %3f, Dice1: %.3f, Dice2: %.3f, ACC: %.3f' % (
                mean_dice, dice1, dice2, ACC))
        print('Mean_Dice:', mean_dice)
        print('Dice1:', dice1)
        print('Dice2:', dice2)
        print('Acc:', ACC)
        return mean_dice, dice1, dice2, ACC


def train(args, model, optimizer, criterion, dataloader_train, dataloader_val):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(args.log_dirs, current_time + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)
    model_tensor = torch.rand([1,3,512,512])
    writer.add_graph(model,model_tensor)
    step = 0
    best_pred = 0.0
    for epoch in range(args.num_epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        train_loss = 0.0
        #        is_best=False
        for i, (data, label) in enumerate(dataloader_train):
            # if i>9:
            #     break
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda().long()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            tq.update(args.batch_size)
            train_loss += loss.item()
            tq.set_postfix(loss='%.6f' % (train_loss / (i + 1)))
            step += 1
            if step % 10 == 0:
                writer.add_scalar('Train/loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('Train/loss_epoch', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))

        if epoch % args.validation_step == 0:
            mean_Dice, Dice1, Dice2, acc = val(args, model, dataloader_val)
            writer.add_scalar('Valid/Mean_val', mean_Dice, epoch)
            writer.add_scalar('Valid/Dice1_val', Dice1, epoch)
            writer.add_scalar('Valid/Dice2_val', Dice2, epoch)
            writer.add_scalar('Valid/Acc_val', acc, epoch)
            is_best = mean_Dice > best_pred
            best_pred = max(best_pred, mean_Dice)
            checkpoint_dir = args.save_model_path
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_latest = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_dice': best_pred,
            }, best_pred, epoch, is_best, checkpoint_dir, filename=checkpoint_latest)


def test(model, dataloader, args, save_path):
    print('start test!')
    with torch.no_grad():
        model.eval()
        # precision_record = []
        tq = tqdm.tqdm(dataloader, desc='\r')
        tq.set_description('test')
        comments = os.getcwd().split('\\')[-1]
        for i, (data, label_path) in enumerate(tq):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                # label = label.cuda()
            output = model(data)
            predict = torch.argmax(output, dim=1, keepdim=True)
            pred = predict.data.cpu().numpy()
            pred_RGB = Data.COLOR_DICT[pred.astype(np.uint8)]

            for index, item in enumerate(label_path):
                img = Image.fromarray(pred_RGB[index].squeeze().astype(np.uint8))
                _, name = os.path.split(item)

                img.save(os.path.join(save_path, name))
                # tq.set_postfix(str=str(save_img_path))
        tq.close()

from model.unet_model import *

def main(mode='train', args=None):
    # create dataset and dataloader
    dataset_path = args.data
    dataset_train = Data(os.path.join(dataset_path, 'train'), scale=(args.crop_width, args.crop_height), mode='train')
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    dataset_val = Data(os.path.join(dataset_path, 'val'), scale=(args.crop_height, args.crop_width), mode='val')
    dataloader_val = DataLoader(
        dataset_val,
        # this has to be 1
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    dataset_test = Data(os.path.join(dataset_path, 'test'), scale=(args.crop_height, args.crop_width), mode='test')
    dataloader_test = DataLoader(
        dataset_test,
        # this has to be 1
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda


    # load model
    model = UNet(n_channels=3, n_classes=2)
    print(args.net_work)
    cudnn.benchmark = True
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,betas=(0.9, 0.999),eps=1e-08, weight_decay=args.weight_decay,amsgrad=False)
    criterion = DiceLoss()
    #criterion = nn.CrossEntropyLoss()
    if mode == 'train':
        train(args, model, optimizer, criterion, dataloader_train, dataloader_val)