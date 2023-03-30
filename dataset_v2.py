import numpy as np
import librosa 
import torch
from torch.utils.data import Dataset
import fairseq


class IEMOCAPDataset(Dataset):
    def __init__(self, 
                 data_root: str,
                 train: bool = True,
                 sequence_length: int = 100,
                 features_name: str = "spec",
                 session_to_test: int = 5,
                 from_npy: str = None,
                 root_path:str = None,
                 wa2v_weights_path:str = None
                 ):
        
        super().__init__()
        if train:
          self.iemocap_table = data_root.query(f'session!={session_to_test}')
        else:
          self.iemocap_table = data_root.query(f'session=={session_to_test}')
        print(self.iemocap_table)
        self.table = self.iemocap_table
        self.train = train
        self.sequence_length = sequence_length
        self.features_name = features_name
        self.from_npy = from_npy

        if self.from_npy is not None:
          self.all_data = np.load(self.from_npy, allow_pickle=True) 
        else:
           self.root_path = root_path
        self.emo_to_int = dict(hap= 0, ang= 1, neu= 2, sad= 3, exc= 0)

        # WAV2VEC
        if features_name == "wav2vec" and from_npy is None:
            cp_path = wa2v_weights_path
            self.model_wav2vec, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
            self.model_wav2vec = self.model_wav2vec[0]
            self.model_wav2vec.eval()

    def __len__(self):
        return len(self.table)

    @staticmethod
    def load_wav(path: str):
        """ Load audio  """
        signal, sr = librosa.load(path)
        return signal, sr

    def melspec(self, signal, sample_rate):
        mel = librosa.feature.melspectrogram(y=signal, 
                                             sr=sample_rate, 
                                             n_mels = 80,
                                             hop_length=512, 
                                             win_length = 1024)
        return mel

    def mfcc(self, signal, sample_rate):
        mfcc = librosa.feature.mfcc(signal,
                                    sr=sample_rate, 
                                    n_mfcc=15, 
                                    n_fft=1024, 
                                    hop_length=256, 
                                    win_length = 1024)
        return mfcc      


    def spec(self, signal, sample_rate): 
        X = librosa.stft(signal, 
                         n_fft=1024,
                        center=False,
                         hop_length=256,
                        win_length = 1024)
        
        X= np.abs(X)**2
        return X

    def wav2vec(self, signal, sample_rate):
      wav2vec = self.model_wav2vec.feature_extractor(signal)
      return wav2vec


    @staticmethod
    def padding(data, seq_length=50):
        """
        :param seq_length:
        :param data:
        :return:
        """
        if len(data.shape) == 2:
            data = np.pad(data, ((0, seq_length - data.shape[0]), (0, 0)), 'wrap')
        return data


    def extract_features(self, signal, sr):
        if self.features_name.lower() == "mfcc": # 15/16 a la place de 80, ça va etre le pire parce que il su pprime le pitch et les emotions sont liées au pitch.
            features = self.mfcc(signal, sr)
        elif self.features_name.lower() == "melspec": # reprsentation condensee du spectogram
            features = self.melspec(signal, sr)
        elif self.features_name.lower() == "wav2vec": 
            features = self.wav2vec(torch.from_numpy(signal)[None],sr)[0].cpu().detach().numpy()
        elif self.features_name.lower() == "spec": 
            features = self.spec(signal, sr)  
        else:
          raise Exception("Sorry, choose only mfcc, melspec, wav2vec, spec")  

        return features   


    def __getitem__(self, item):
        while True:
            line = self.iemocap_table["wav_path"].iloc[item]
            emotion = self.iemocap_table["emotion"].iloc[item]
            emotion = self.emo_to_int[emotion]
            
            if self.from_npy is None:
              wav_path = self.root_path + "/" + line
              audio, sr = self.load_wav(wav_path) 
              features = self.extract_features(audio, sr=16000).transpose()
            else:
              features = self.all_data[item]
              
            self.number_frames = features.shape[0]
            if self.number_frames > self.sequence_length:
                break
            else:
                features = self.padding(features, seq_length=self.sequence_length+1)
                self.number_frames = features.shape[0]
                break

        self.current_frame = np.random.randint(0, self.number_frames - self.sequence_length)
        self.out = features[self.current_frame: self.current_frame + self.sequence_length]
        return torch.from_numpy(self.out), torch.tensor(emotion)  
