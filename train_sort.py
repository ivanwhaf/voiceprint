import argparse
import os
import random
import time

import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn import metrics
from torch.utils.data import DataLoader, Subset

from dataset import AISHELL1, AISHELL1Sort
from loss import *
from model import ECAPA_TDNN

parser = argparse.ArgumentParser()
parser.add_argument('-project_name', type=str, help='project name',
                    default='voiceprint_aishell1_ecapatdnn_150utt_noise_0.1_sort')
parser.add_argument('-noise_type', type=str, help='noise type', default='asymmetric')
parser.add_argument('-noise_rate', type=float, help='noise rate', default=0.8)
parser.add_argument('-dataset', type=str, help='dataset', default='aishell1')
# parser.add_argument('-dataset_path', type=str, help='path of dataset',
#                     default='F:/Data/dataset/Voiceprint/myairbridge-AISHELL-1/data_aishell/wav')
parser.add_argument('-dataset_path', type=str, help='path of dataset',
                    default='F:/Data/dataset/Voiceprint/myairbridge-AISHELL-1/data_aishell/npy')
parser.add_argument('-data_type', type=str, help='data type', default='npy')
parser.add_argument('-num_classes', type=int, help='number of classes', default=380)
parser.add_argument('-epochs', type=int, help='training epochs', default=100)
parser.add_argument('-batch_size', type=int, help='batch size', default=64)
parser.add_argument('-lr', type=float, help='learning rate', default=5e-4)
parser.add_argument('-l2_reg', type=float, help='l2 regularization', default=1e-4)
parser.add_argument('-threshold', type=float, help='score threshold', default=0.6)
parser.add_argument('-seed', type=int, help='numpy and pytorch seed', default=0)
parser.add_argument('-log_dir', type=str, help='log dir', default='output')
args = parser.parse_args()


def create_dataloader(root):
    train_set = AISHELL1Sort()
    val_set = AISHELL1(root, dataset_type='val', data_type=args.data_type)
    test_set = AISHELL1(root, dataset_type='test', data_type=args.data_type)

    # generate DataLoader
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    return train_set, val_loader, test_loader


def train(model, train_loader, optimizer, criterion, epoch, device, train_loss_lst, train_acc_lst):
    model.train()
    criterion.train()
    correct = 0
    train_loss = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        # print(inputs.shape, labels.shape) # bs*80*401
        # inputs = torch.unsqueeze(inputs, 1)
        outputs = model(inputs)

        # criterion = nn.CrossEntropyLoss()
        loss, pred = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pred = pred.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()

        # print train loss and accuracy
        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]  Loss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))

    # record loss and accuracy
    train_loss /= len(train_loader)  # must divide iter num
    train_loss_lst.append(train_loss)
    train_acc_lst.append(correct / len(train_loader.dataset))

    print('Train set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)'
          .format(train_loss, correct, len(train_loader.dataset),
                  100. * correct / len(train_loader.dataset)))
    return train_loss_lst, train_acc_lst


def validate(model, criterion, val_loader, device, val_loss_lst, val_acc_lst):
    model.eval()
    criterion.eval()
    val_loss = 0
    correct = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            # data = torch.unsqueeze(data, 1)
            output = model(data)

            # criterion1 = nn.CrossEntropyLoss()
            # val_loss += criterion1(output, target).item()
            loss, pred = criterion(output, target)
            val_loss += loss.item()

            # val_loss += F.nll_loss(output, target, reduction='sum').item()

            # find index of max prob
            pred = pred.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    # print val loss and accuracy
    val_loss /= len(val_loader)
    print('\nVal set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'
          .format(val_loss, correct, len(val_loader.dataset),
                  100. * correct / len(val_loader.dataset)))

    # record loss and accuracy
    val_loss_lst.append(val_loss)
    val_acc_lst.append(correct / len(val_loader.dataset))
    return val_loss_lst, val_acc_lst


def test(model, test_loader, device, output_path):
    model.eval()
    with torch.no_grad():
        TP, TN, FP, FN = np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(100)
        for d1, d2 in test_loader:
            data1, target1 = d1
            data2, target2 = d2

            data1, target1 = data1.to(device), target1.to(device)
            data2, target2 = data2.to(device), target2.to(device)

            # data1 = torch.unsqueeze(data1, 1)
            # data2 = torch.unsqueeze(data2, 1)

            outputs1 = model(data1)
            outputs2 = model(data2)

            cos = F.cosine_similarity(outputs1, outputs2)
            # feats1 = F.normalize(outputs1, dim=-1)
            # feats2 = F.normalize(outputs2, dim=-1)
            # cos = torch.mm(feats1, feats2.T)

            for i in range(len(cos)):
                t1 = target1[i]
                t2 = target2[i]
                c = cos[i]

                for j in range(0, 100, 1):
                    t = j * 0.01
                    if c >= t:
                        if t1 == t2:
                            TP[j] += 1
                        else:
                            FP[j] += 1
                    else:
                        if t1 == t2:
                            FN[j] += 1
                        else:
                            TN[j] += 1
        # print('TP:', TP, 'TN:', TN, 'FP:', FP, 'FN:', FN)
        FAR = FP / (FP + TN)
        FRR = FN / (TP + FN)
        print('FAR:', FAR, 'FRR', FRR)

        # calculate EER
        min_diff = 1e5
        min_idx = 0
        for i in range(len(FAR)):
            far = FAR[i]
            frr = FRR[i]
            diff = abs(far - frr)
            if diff < min_diff:
                min_diff = diff
                min_idx = i
        EER = FAR[min_idx]
        print('EER:', EER)

        # plot
        plt.figure("FRR and FAR", dpi=150)
        plt.plot(FAR, FRR, label='EER = {:.3f}'.format(EER))
        plt.xlabel("FAR")
        plt.ylabel("FRR")
        plt.title("FAR and FRR")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.savefig(os.path.join(output_path, "frr_far.png"))
        plt.show()
        plt.close()


def test2(model, test_loader, device, output_path):
    model.eval()

    scores, labels = [], []

    with torch.no_grad():
        for d1, d2 in test_loader:
            data1, target1 = d1
            data2, target2 = d2

            data1, target1 = data1.to(device), target1.to(device)
            data2, target2 = data2.to(device), target2.to(device)

            # data1 = torch.unsqueeze(data1, 1)
            # data2 = torch.unsqueeze(data2, 1)

            outputs1 = model(data1)
            outputs2 = model(data2)

            cos = F.cosine_similarity(outputs1, outputs2).cpu().numpy()

            for i in range(len(cos)):
                t1 = target1[i]
                t2 = target2[i]
                c = cos[i]

                if t1 == t2:
                    labels.append(1)
                else:
                    labels.append(0)

                scores.append(c)

        _, eer, fpr, fnr = tuneThresholdfromScore(scores, labels, [1, 0.1])

        roc_auc = metrics.auc(fpr, 1 - fnr)

        plt.figure("ROC", dpi=150)
        plt.plot(fpr, fnr, 'k--', label='ROC(area = {:.2f}) EER = {:.3f}'.format(roc_auc, eer), lw=2, color='blue')

        plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
        plt.ylim([-0.05, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('FNR')
        plt.title('ROC Curve')
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.savefig(os.path.join(output_path, "roc_curve.png"))
        plt.show()
        plt.close()
        print('EER:', eer)


def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = np.nanargmin(np.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    for tfa in target_fa:
        idx = np.nanargmin(np.absolute((tfa - fpr)))  # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    idxE = np.nanargmin(np.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE])

    return tunedThreshold, eer, fpr, fnr


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    set_seed(args.seed)

    # create output folder
    now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    output_path = os.path.join(args.log_dir, args.project_name + ' ' + now)
    os.makedirs(output_path)

    train_set, val_loader, test_loader = create_dataloader(args.dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = ResNet18(args.num_classes).to(device)
    model = ECAPA_TDNN(C=1024).to(device)

    # criterion = AMSoftmax(192, args.num_classes).to(device)
    criterion = AAMsoftmax(args.num_classes).to(device)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.l2_reg)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=2e-5)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)

    train_loss_lst, val_loss_lst = [], []
    train_acc_lst, val_acc_lst = [], []

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

    for epoch in range(args.epochs):
        # 重采样，去除头尾数据
        if epoch == 50:
            indices = np.arange(int(len(train_set) * 0.1), int(len(train_set) * 0.9))
            train_set = Subset(train_set, indices)
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

        train_loss_lst, train_acc_lst = train(model, train_loader, optimizer, criterion, epoch, device, train_loss_lst,
                                              train_acc_lst)
        val_loss_lst, val_acc_lst = validate(model, criterion, val_loader, device, val_loss_lst, val_acc_lst)

        scheduler.step()

        if epoch in [40, 80]:
            # args.lr *= 0.1
            # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.l2_reg)
            # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)

            torch.save(model.state_dict(), os.path.join(output_path, "model" + "_epoch" + str(epoch) + ".pth"))
            torch.save(criterion.state_dict(), os.path.join(output_path, "criterion" + "_epoch" + str(epoch) + ".pth"))

    try:
        test(model, test_loader, device, output_path)
    except:
        pass
    try:
        test2(model, test_loader, device, output_path)
    except:
        pass

    fig = plt.figure('Loss and acc', dpi=150)
    plt.plot(range(args.epochs), train_loss_lst, 'g', label='train loss')
    plt.plot(range(args.epochs), val_loss_lst, 'k', label='val loss')
    plt.plot(range(args.epochs), train_acc_lst, 'r', label='train acc')
    plt.plot(range(args.epochs), val_acc_lst, 'b', label='val acc')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('acc-loss')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_path, 'loss_acc.png'))
    plt.show()
    plt.close(fig)

    # save model
    torch.save(model.state_dict(), os.path.join(output_path, "model.pth"))
    torch.save(criterion.state_dict(), os.path.join(output_path, "criterion.pth"))
