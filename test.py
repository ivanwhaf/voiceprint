import argparse
import os
import wave

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader

from dataset import AISHELL1
from model import ECAPA_TDNN

parser = argparse.ArgumentParser()
parser.add_argument('-project_name', type=str, help='project name', default='test')
parser.add_argument('-dataset', type=str, help='dataset type', default='aishell1')
parser.add_argument('-dataset_path', type=str, help='path of dataset',
                    default='F:/Data/dataset/Voiceprint/myairbridge-AISHELL-1/data_aishell/npy')
parser.add_argument('-data_type', type=str, help='data type', default='npy')
parser.add_argument('-num_classes', type=int, help='number of classes', default=380)
parser.add_argument('-batch_size', type=int, help='batch size', default=64)
parser.add_argument('-seed', type=int, help='numpy and pytorch seed', default=0)
parser.add_argument('-log_dir', type=str, help='log dir', default='output')
args = parser.parse_args()


def read_music():
    f = wave.open('Are_You_Ok.wav', 'rb')

    params = f.getparams()
    print(params)
    nchannels, sampwidth, framerate, nframes = params[0:4]

    str_data = f.readframes(nframes)

    f.close()

    wave_data = np.fromstring(str_data, dtype=np.int16)
    print(wave_data.shape)
    # wave_data = wave_data * 1.0/max(abs(wave_data))  # wave幅值归一化

    wave_data.shape = -1, 2
    print(wave_data.shape)

    wave_data = wave_data.T

    wav_time = np.arange(0, nframes) / framerate

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(wav_time, wave_data[0])
    plt.subplot(2, 1, 2)
    plt.plot(wav_time, wave_data[1], c="r")
    plt.xlabel("time")
    plt.show()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.specgram(wave_data[0], Fs=framerate,
                 scale_by_freq=True, sides='default')
    plt.subplot(2, 1, 2)
    plt.specgram(wave_data[1], Fs=framerate,
                 scale_by_freq=True, sides='default')
    plt.ylabel('Frequency(Hz)')
    plt.xlabel('Time(s)')
    plt.show()


def read_aishell():
    f = wave.open('018_1_recorded0_001_normal.wav', 'rb')
    params = f.getparams()
    print(params)
    nchannels, sampwidth, framerate, nframes = params[0:4]
    str_data = f.readframes(nframes)
    f.close()

    wave_data = np.fromstring(str_data, dtype=np.int16)
    print(wave_data.shape)
    # wave_data = wave_data * 1.0/max(abs(wave_data))  # wave幅值归一化

    print(wave_data.shape)

    wave_data = wave_data.T

    wav_time = np.arange(0, nframes) / framerate

    plt.figure()
    plt.plot(wav_time, wave_data)
    plt.xlabel("time")
    plt.show()

    plt.figure()
    plt.specgram(wave_data, Fs=framerate, scale_by_freq=True, sides='default')
    plt.ylabel('Frequency(Hz)')
    plt.xlabel('Time(s)')
    plt.show()


def main():
    # read_music()
    # read_aishell()

    # path = 'F:\Data\dataset\Voiceprint\myairbridge-AISHELL-1\data_aishell\wav'
    # ids = os.listdir(path)
    # print(ids)
    #
    # shape_lst = []
    #
    # for i, id in enumerate(ids):
    #     if i < 340:
    #         speaker_folder = os.path.join(path, id, 'train', id)
    #     elif i < 380:
    #         speaker_folder = os.path.join(path, id, 'dev', id)
    #     else:
    #         speaker_folder = os.path.join(path, id, 'test', id)
    #     print(speaker_folder)
    #     files = os.listdir(speaker_folder)
    #     # print(files)
    #
    #     for file in files:
    #         file_path = os.path.join(speaker_folder, file)
    #         audio, sr = librosa.load(file_path, sr=16000)
    #         shape_lst.append(audio.shape[0])
    #         # mel_spectrogram=librosa.feature.melspectrogram(audio,sr,n_mels=256,hop_length=128,fmax=8000)
    #         # print(audio.shape,mel_spectrogram.shape)
    # print(max(shape_lst), min(shape_lst))
    root = 'F:/Data/dataset/Voiceprint/myairbridge-AISHELL-1/data_aishell/wav'
    dataset = AISHELL1(root, True)
    print(len(dataset))


def create_dataloader(root):
    train_set = AISHELL1(root, dataset_type='train', data_type=args.data_type)
    val_set = AISHELL1(root, dataset_type='val', data_type=args.data_type)
    test_set = AISHELL1(root, dataset_type='test', data_type=args.data_type)

    # generate DataLoader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def test(model, test_loader, device, output_path):
    model.eval()
    with torch.no_grad():
        TP, TN, FP, FN = np.zeros(200), np.zeros(200), np.zeros(200), np.zeros(200)
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

                for j in range(0, 200, 1):
                    t = j * 0.005
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
        # plt.savefig(os.path.join(output_path, "frr_far.png"))
        plt.show()
        plt.close()

    return FAR, FRR


def test2(model, test_loader, device):
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
        # plt.savefig("roc_curve.png")
        plt.show()
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
    print(len(fpr))
    return tunedThreshold, eer, fpr, fnr


def compare_far_frr(model1, model2, test_loader, device, output_path):
    model1.eval()

    scores, labels = [], []
    with torch.no_grad():
        for d1, d2 in test_loader:
            data1, target1 = d1
            data2, target2 = d2

            data1, target1 = data1.to(device), target1.to(device)
            data2, target2 = data2.to(device), target2.to(device)

            outputs1 = model1(data1)
            outputs2 = model1(data2)

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

        _, eer, fpr1, fnr1 = tuneThresholdfromScore(scores, labels, [1, 0.1])

        roc_auc = metrics.auc(fpr1, 1 - fnr1)

        plt.figure("ROC", dpi=150)
        plt.plot(fpr1, fnr1, 'k--', label='ROC(area = {:.2f}) EER = {:.3f}'.format(roc_auc, eer), lw=2, color='blue')

        plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
        plt.ylim([-0.05, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('FNR')
        plt.title('ROC Curve')
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.savefig(os.path.join(output_path, "model1_roc_curve.png"))
        # plt.show()
        print('Model1 EER:', eer)
        plt.close()

    model2.eval()
    scores, labels = [], []
    test_set = AISHELL1(args.dataset_path, dataset_type='test', data_type=args.data_type)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        for d1, d2 in test_loader:
            data1, target1 = d1
            data2, target2 = d2

            data1, target1 = data1.to(device), target1.to(device)
            data2, target2 = data2.to(device), target2.to(device)

            outputs1 = model2(data1)
            outputs2 = model2(data2)

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

        _, eer, fpr2, fnr2 = tuneThresholdfromScore(scores, labels, [1, 0.1])

        roc_auc = metrics.auc(fpr2, 1 - fnr2)

        plt.figure("ROC", dpi=150)
        plt.plot(fpr2, fnr2, 'k--', label='ROC(area = {:.2f}) EER = {:.3f}'.format(roc_auc, eer), lw=2,
                 color='blue')

        plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
        plt.ylim([-0.05, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('FNR')
        plt.title('ROC Curve')
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.savefig(os.path.join(output_path, "model2_roc_curve.png"))
        # plt.show()
        print('Model2 EER:', eer)
        plt.close()

    percent = []
    r = min(len(fpr1), len(fpr2))
    for idx in range(r):
        fp1 = float(fpr1[idx])
        fp2 = float(fpr2[idx])
        diff = fp1 - fp2
        if fp2 != 0:
            p = diff / fp2
        else:
            p = 10000
        # p = diff
        percent.append(p)

    print('fpr1', fpr1)
    print('fpr2', fpr2)
    print('percent', percent)

    max_p = -1
    max_i = -1

    res = []
    res_i = []
    for i in range(len(percent)):
        if percent[i] < -0.2 and percent[i] != 10000:
            res.append(percent[i])
            res_i.append(i)
    print('res', res)
    print('rei', res_i)
    # print('max percent', max_p, 'max i', max_i)


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = ECAPA_TDNN(C=1024).to(device)
    # model.load_state_dict(
    #     torch.load('output/voiceprint_aishell1_ecapatdnn_150utt_noise_0.05_sort 2022-03-14-00-23-29/model.pth'))
    #
    # test_set = AISHELL1(args.dataset_path, dataset_type='test', data_type=args.data_type)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    #
    # FAR1, FRR1 = test(model, test_loader, device, './')
    #
    # model2 = ECAPA_TDNN(C=1024).to(device)
    # model2.load_state_dict(
    #     torch.load('output/voiceprint_aishell1_ecapatdnn_150utt_noise_0.05_sort 2022-03-13-12-34-11/model.pth'))
    #
    # FAR2, FRR2 = test(model2, test_loader, device, './')
    #
    # percent = []
    # for idx in range(len(FAR1)):
    #     fp1 = float(FAR1[idx])
    #     fp2 = float(FAR2[idx])
    #     diff = fp1 - fp2
    #     if fp2 != 0:
    #         p = diff / fp2
    #     else:
    #         p = 10000
    #     # p = diff
    #     percent.append(p)
    #
    # print('percent', percent)
    #
    # res = []
    # res_i = []
    #
    # ar1 = []
    # ar2 = []
    # for i in range(len(percent)):
    #     if percent[i] < -0.2 and percent[i] != 10000:
    #         res.append(percent[i])
    #         res_i.append(i)
    #         ar1.append(FAR1[i])
    #         ar2.append(FAR2[i])
    #
    # print('res', res)
    # print('rei', res_i)
    # print('AR1', ar1)
    # print('AR2', ar2)
    #
    # max_p = percent[0]
    # max_i = 0
    # for i in range(len(percent)):
    #     if percent[i] > max_p:
    #         max_p = percent[i]
    # print('max percent', max_p, 'max i', max_i)

    # root = 'H:/Data/datasets/Voiceprint/myairbridge-AISHELL-1/data_aishell/npy'
    # txt_root = "H:/Data/datasets/Voiceprint/myairbridge-AISHELL-1/data_aishell/data"
    # speakers = os.listdir(root)
    #
    # for idx, speaker in enumerate(speakers):
    #     speaker_folder = os.path.join(root, speaker)
    #     files = os.listdir(speaker_folder)
    #
    #     with open(os.path.join(txt_root, speaker + '.txt'), "w") as f:
    #         for file in files:
    #             file_path = os.path.join(speaker_folder, file)
    #             f.write(file_path + ' ' + str(idx) + '\n')
    #     print(speaker_folder)

    # sort inference loss
    sample_loss = np.load("sample_loss_150utt_noise_s0.1.npy", allow_pickle=True).item()

    sample_loss = sorted(sample_loss.items(), key=lambda x: x[1][0], reverse=False)

    with open('data_150utt_noise_s0.1_sort.txt', 'w') as f:
        for sample in sample_loss:
            path = sample[0]
            loss = sample[1][0]
            label = sample[1][1]

            f.write(path + ' ' + str(label) + ' ' + str(loss) + '\n')

    # train_set = AISHELL1(args.dataset_path, dataset_type='train', data_type=args.data_type)

    # test_set = AISHELL1(args.dataset_path, dataset_type='test', data_type=args.data_type)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # model1 = ECAPA_TDNN(C=1024).to(device)
    # model1.load_state_dict(
    #     torch.load('output/voiceprint_aishell1_ecapatdnn_150utt_noise_0.05_sort 2022-03-14-00-23-29/model.pth'))
    #
    # model2 = ECAPA_TDNN(C=1024).to(device)
    # model2.load_state_dict(
    #     torch.load('output/voiceprint_aishell1_ecapatdnn_150utt_noise_0.05_sort 2022-03-13-12-34-11/model.pth'))
    #
    # compare_far_frr(model1, model2, test_loader, device, './')
