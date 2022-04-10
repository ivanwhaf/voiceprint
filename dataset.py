import os

import librosa
import numpy as np
from SpecAugment import spec_augment_pytorch
from torch.utils.data import Dataset


def cut_pad(audio, length):
    if audio.shape[0] < length:
        diff = length - audio.shape[0]
        audio = np.pad(audio, (0, diff), 'wrap')
    else:
        audio = audio[:length]
    return audio


class AISHELL1Inference(Dataset):
    def __init__(self):
        self.data = []
        self.targets = []

        # root = "data_100utt_noise_s0.1.txt"
        # root = "data_100utt_noise_s0.05.txt"
        # root = "data_150utt_clean.txt"
        root = "data_150utt_noise_s0.1.txt"

        with open(root, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                line = line.split(' ')
                path = line[0]
                label = line[1]
                self.data.append(path)
                self.targets.append(int(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_path = self.data[index]
        target = self.targets[index]

        audio = np.load(file_path)

        return audio, target, file_path


class AISHELL1Sort(Dataset):
    def __init__(self):
        self.data = []
        self.targets = []

        # root = "data_100utt_clean.txt"
        # root = "data_100utt_clean_sort.txt"
        # root = "data_100utt_noise_s0.02_sort.txt"
        # root = "data_100utt_noise_s0.05_sort.txt"
        # root = "data_100utt_noise_s0.1_sort.txt"
        # root = "data_150utt_clean_sort.txt"
        # root = "data_150utt_noise_s0.05_sort.txt"
        root = "data_150utt_noise_s0.1_sort.txt"

        with open(root, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                line = line.split(' ')
                path = line[0]
                label = line[1]
                self.data.append(path)
                self.targets.append(int(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_path = self.data[index]
        target = self.targets[index]

        audio = np.load(file_path)

        return audio, target


class AISHELL1Online(Dataset):
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type

        self.data = []
        self.targets = []
        self.all_data = {'train': [], 'val': [], 'test': []}
        self.all_targets = {'train': [], 'val': [], 'test': []}

        root = "H:/Data/datasets/Voiceprint/myairbridge-AISHELL-1/data_aishell/data"
        files = os.listdir(root)

        for idx, file in enumerate(files):
            with open(os.path.join(root, file)) as f:
                if idx < 380:
                    for i, line in enumerate(f.readlines()):
                        line = line.strip('\n')
                        line = line.split(' ')
                        if i < 300 * 50:
                            self.all_data['train'].append(line[0])
                            self.all_targets['train'].append(int(line[1]))
                        elif i < 300 * 50 + 10:
                            self.all_data['val'].append(line[0])
                            self.all_targets['val'].append(int(line[1]))
                else:
                    for i, line in enumerate(f.readlines()):
                        line = line.strip('\n')
                        line = line.split(' ')
                        self.all_data['test'].append(line[0])
                        self.all_targets['test'].append(int(line[1]))

        if self.dataset_type == 'train':
            self.data = self.all_data['train']
            self.targets = self.all_targets['train']
        if self.dataset_type == 'val':
            self.data = self.all_data['val']
            self.targets = self.all_targets['val']
        if self.dataset_type == 'test':
            self.data = self.all_data['test']
            self.targets = self.all_targets['test']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # test
        if self.dataset_type == 'test':
            file_path = self.data[index]
            target = self.targets[index]

            idx = np.random.randint(0, len(self.data))
            file_path2 = self.data[idx]
            target2 = self.targets[idx]

            audio = np.load(file_path)
            audio2 = np.load(file_path2)

            return (audio, target), (audio2, target2)

        # train/val
        file_path = self.data[index]
        target = self.targets[index]

        audio = np.load(file_path)

        return audio, target


# 9040 235199
class AISHELL1(Dataset):
    def __init__(self, root, dataset_type, data_type='wav'):
        self.dataset_type = dataset_type
        self.data_type = data_type

        self.data = []
        self.targets = []
        self.all_data = {'train': [], 'val': [], 'test': []}
        self.all_targets = {'train': [], 'val': [], 'test': []}

        speakers = os.listdir(root)

        for idx, speaker in enumerate(speakers):
            if idx < 340:
                if self.data_type == 'wav':
                    speaker_folder = os.path.join(root, speaker, 'train', speaker)
                else:
                    speaker_folder = os.path.join(root, speaker)

                files = os.listdir(speaker_folder)[:150]
                for file in files:
                    file_path = os.path.join(speaker_folder, file)
                    self.all_data['train'].append(file_path)
                    self.all_targets['train'].append(idx)

                files = os.listdir(speaker_folder)[300:310]
                for file in files:
                    file_path = os.path.join(speaker_folder, file)
                    self.all_data['val'].append(file_path)
                    self.all_targets['val'].append(idx)

            elif idx < 380:
                if self.data_type == 'wav':
                    speaker_folder = os.path.join(root, speaker, 'dev', speaker)
                else:
                    speaker_folder = os.path.join(root, speaker)

                files = os.listdir(speaker_folder)[:150]
                for file in files:
                    file_path = os.path.join(speaker_folder, file)
                    self.all_data['train'].append(file_path)
                    self.all_targets['train'].append(idx)

                files = os.listdir(speaker_folder)[300:310]
                for file in files:
                    file_path = os.path.join(speaker_folder, file)
                    self.all_data['val'].append(file_path)
                    self.all_targets['val'].append(idx)
            else:
                if self.data_type == 'wav':
                    speaker_folder = os.path.join(root, speaker, 'test', speaker)
                else:
                    speaker_folder = os.path.join(root, speaker)

                files = os.listdir(speaker_folder)[:]
                for file in files:
                    file_path = os.path.join(speaker_folder, file)
                    self.all_data['test'].append(file_path)
                    self.all_targets['test'].append(idx)

            # print(speaker_folder, len(files))

        if dataset_type == 'train':
            self.data = self.all_data['train']
            self.targets = self.all_targets['train']
        if dataset_type == 'val':
            self.data = self.all_data['val']
            self.targets = self.all_targets['val']
        if dataset_type == 'test':
            self.data = self.all_data['test']
            self.targets = self.all_targets['test']

        # add noise
        # self.clean_sample_idx = []
        # self.noisy_sample_idx = []

        # noise_rate = 0.1
        # num_classes = 380
        # ntm = noise_rate * np.full((num_classes, num_classes), 1 / (num_classes - 1))
        # np.fill_diagonal(ntm, 1 - noise_rate)
        #
        # sample_indices = np.arange(len(self.data))
        # # np.random.shuffle(indices)
        #
        # # generate noisy label by noise transition matrix
        # for i in sample_indices:
        #     label = np.random.choice(num_classes, p=ntm[self.targets[i]])  # new label
        #     if label != self.targets[i]:
        #         self.noisy_sample_idx.append(i)
        #     self.targets[i] = label
        #
        # self.clean_sample_idx = np.setdiff1d(sample_indices, self.noisy_sample_idx)
        #
        # print('Noise type: Symmetric')
        # print('Noise rate:', noise_rate)
        # print('Noise transition matrix:\n', ntm)
        #
        # with open('data_150utt_noise_s0.1.txt', 'w') as f:
        #     for i in range(len(self.data)):
        #         p = self.data[i]
        #         l = self.targets[i]
        #         f.write(p + ' ' + str(l) + '\n')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # test
        if self.dataset_type == 'test':
            file_path = self.data[index]
            target = self.targets[index]

            idx = np.random.randint(0, len(self.data))
            file_path2 = self.data[idx]
            target2 = self.targets[idx]

            if self.data_type == 'wav':
                length = 16000 * 3  # 3 seconds

                audio, sr = librosa.load(file_path, sr=16000)
                audio = cut_pad(audio, length)
                audio = librosa.feature.melspectrogram(audio, sr, n_fft=512, n_mels=80, win_length=400, hop_length=160,
                                                       fmax=8000)

                audio2, sr2 = librosa.load(file_path2, sr=16000)
                audio2 = cut_pad(audio2, length)
                audio2 = librosa.feature.melspectrogram(audio2, sr, n_fft=512, n_mels=80, win_length=400,
                                                        hop_length=160,
                                                        fmax=8000)
            else:
                audio = np.load(file_path)
                audio2 = np.load(file_path2)

            return (audio, target), (audio2, target2)

        # train/val
        file_path = self.data[index]
        target = self.targets[index]

        if self.data_type == 'wav':
            audio, sr = librosa.load(file_path, sr=16000)

            length = 16000 * 3  # 3 seconds
            audio = cut_pad(audio, length)
            # audio = np.stack([audio], axis=0)
            audio = librosa.feature.melspectrogram(audio, sr, n_fft=512, n_mels=80, win_length=400, hop_length=160,
                                                   fmax=8000)

            if self.dataset_type == 'train':
                audio = spec_augment_pytorch.spec_augment(audio)
        else:
            audio = np.load(file_path)

        return audio, target
