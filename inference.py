import argparse
import random

import numpy as np
from torch.utils.data import DataLoader

from dataset import AISHELL1Inference
from loss import *
from model import ECAPA_TDNN

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, help='dataset', default='aishell1')
parser.add_argument('-dataset_path', type=str, help='path of dataset',
                    default='F:/Data/dataset/Voiceprint/myairbridge-AISHELL-1/data_aishell/npy')
parser.add_argument('-data_type', type=str, help='data type', default='npy')
parser.add_argument('-num_classes', type=int, help='number of classes', default=380)
parser.add_argument('-batch_size', type=int, help='batch size', default=256)
parser.add_argument('-seed', type=int, help='numpy and pytorch seed', default=0)
args = parser.parse_args()


def create_dataloader():
    train_set = AISHELL1Inference()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    return train_loader


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    set_seed(args.seed)
    # 集成模型
    model_lst = ["output/voiceprint_aishell1_ecapatdnn 2022-02-20-21-52-26/model.pth",
                 "output/voiceprint_aishell1_ecapatdnn 2022-02-20-10-25-33/model.pth",
                 "output/voiceprint_aishell1_ecapatdnn 2022-02-19-21-06-36/model.pth",
                 "output/voiceprint_aishell1_ecapatdnn 2022-02-19-10-57-39/model.pth",
                 "output/voiceprint_aishell1_ecapatdnn 2022-02-18-21-58-32/model.pth"]

    criterion_lst = ["output/voiceprint_aishell1_ecapatdnn 2022-02-20-21-52-26/criterion.pth",
                     "output/voiceprint_aishell1_ecapatdnn 2022-02-20-10-25-33/criterion.pth",
                     "output/voiceprint_aishell1_ecapatdnn 2022-02-19-21-06-36/criterion.pth",
                     "output/voiceprint_aishell1_ecapatdnn 2022-02-19-10-57-39/criterion.pth",
                     "output/voiceprint_aishell1_ecapatdnn 2022-02-18-21-58-32/criterion.pth"]

    train_loader = create_dataloader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_loss = {}

    # inference
    for i in range(len(model_lst)):
        model_path = model_lst[i]
        criterion_path = criterion_lst[i]

        model = ECAPA_TDNN(C=1024).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        criterion = AAMsoftmax(args.num_classes).to(device)
        criterion.load_state_dict(torch.load(criterion_path))
        criterion.eval()

        with torch.no_grad():
            for batch_idx, (inputs, labels, paths) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                # inputs = torch.unsqueeze(inputs, 1)
                outputs = model(inputs)

                criterion = AAMsoftmax(args.num_classes).to(device)
                losses, _ = criterion(outputs, labels)

                losses = losses.detach().cpu().numpy().tolist()
                labels = labels.detach().cpu().numpy().tolist()

                for j in range(len(paths)):
                    path = paths[j]
                    loss = losses[j]
                    label = labels[j]
                    if path not in sample_loss:
                        sample_loss[path] = [loss, label]
                    else:
                        x = sample_loss[path]
                        x[0] = x[0] + loss
                        sample_loss[path] = x
        print(model_path, "done")

    np.save("sample_loss_150utt_noise_s0.1.npy", sample_loss)
