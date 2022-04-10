import os

import librosa
import numpy as np
from SpecAugment import spec_augment_pytorch


def cut_pad(audio, length):
    if audio.shape[0] < length:
        diff = length - audio.shape[0]
        audio = np.pad(audio, (0, diff), 'wrap')
    else:
        audio = audio[:length]
    return audio


if __name__ == '__main__':
    root = 'F:/Data/dataset/Voiceprint/myairbridge-AISHELL-1/data_aishell/wav'
    # root = 'H:/Data/datasets/Voiceprint/myairbridge-AISHELL-1/data_aishell/wav'
    # save_path = 'F:/Data/dataset/Voiceprint/myairbridge-AISHELL-1/data_aishell/npy'
    save_path = 'H:/Data/datasets/Voiceprint/myairbridge-AISHELL-1/data_aishell/npy'

    speakers = os.listdir(root)
    for speaker in speakers:
        os.makedirs(os.path.join(save_path, speaker))

    for idx, speaker in enumerate(speakers):
        if idx < 340:
            speaker_folder = os.path.join(root, speaker, 'train', speaker)
            files = os.listdir(speaker_folder)
            for i, file in enumerate(files):
                file_path = os.path.join(speaker_folder, file)
                audio, sr = librosa.load(file_path, sr=16000)
                length = 16000 * 3  # 3 seconds
                audio = cut_pad(audio, length)
                audio = librosa.feature.melspectrogram(audio, sr, n_fft=512, n_mels=80, win_length=400, hop_length=160,
                                                       fmax=8000)
                if i < 300:
                    for j in range(50):
                        audio = spec_augment_pytorch.spec_augment(audio)
                        np.save(save_path + '/' + speaker + '/' + file.split('.')[0] + '_' + str(j + 1) + '.npy', audio)
                        # print(speaker, '/', file)
                else:
                    np.save(save_path + '/' + speaker + '/' + file.split('.')[0] + '.npy', audio)
                    # print(speaker, '/', file)
            print(idx, 'done')

        elif idx < 380:
            speaker_folder = os.path.join(root, speaker, 'dev', speaker)
            files = os.listdir(speaker_folder)
            for i, file in enumerate(files):
                file_path = os.path.join(speaker_folder, file)
                audio, sr = librosa.load(file_path, sr=16000)
                length = 16000 * 3  # 3 seconds
                audio = cut_pad(audio, length)
                audio = librosa.feature.melspectrogram(audio, sr, n_fft=512, n_mels=80, win_length=400, hop_length=160,
                                                       fmax=8000)

                if i < 300:
                    for j in range(50):
                        audio = spec_augment_pytorch.spec_augment(audio)
                        np.save(save_path + '/' + speaker + '/' + file.split('.')[0] + '_' + str(j + 1) + '.npy', audio)

                else:
                    np.save(save_path + '/' + speaker + '/' + file.split('.')[0] + '.npy', audio)
                    # print(speaker, '/', file)
            print(idx, 'done')

        else:
            speaker_folder = os.path.join(root, speaker, 'test', speaker)
            files = os.listdir(speaker_folder)
            for file in files:
                file_path = os.path.join(speaker_folder, file)
                audio, sr = librosa.load(file_path, sr=16000)
                length = 16000 * 3  # 3 seconds
                audio = cut_pad(audio, length)
                audio = librosa.feature.melspectrogram(audio, sr, n_fft=512, n_mels=80, win_length=400, hop_length=160,
                                                       fmax=8000)
                np.save(save_path + '/' + speaker + '/' + file.split('.')[0] + '.npy', audio)
                print(speaker, '/', file)
            print(idx, 'done')
