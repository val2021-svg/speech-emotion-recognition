import numpy as np
from tqdm import tqdm
from dataset import IEMOCAPDataset as IEMOCAPDataset_v1
from dataset_v2 import IEMOCAPDataset as IEMOCAPDataset_v2

def preprocess(version, preprocessed_path, df, features = None, session_to_test= None, train= None, root_path=None, wa2v_weights_path=None):
    if version == 1:
        x_features_train = IEMOCAPDataset_v1(data_root=df, features_name=features, session_to_test=session_to_test, train=train, root_path=root_path, wa2v_weights_path=wa2v_weights_path)
    elif version == 2:
        x_features_train = IEMOCAPDataset_v2(data_root=df, features_name=features, session_to_test=session_to_test, train=train, root_path=root_path, wa2v_weights_path=wa2v_weights_path)
    
    len = x_features_train.__len__()
    
    all_data = []
    for i in tqdm(range(len)):
        line = x_features_train.iemocap_table["wav_path"].iloc[i]
        wav_path = root_path + "/" + line
        audio, sr = x_features_train.load_wav(wav_path)
        data = x_features_train.extract_features(audio, sr).transpose()
        all_data.append(data)

    np.save(f"{preprocessed_path}/{features}-session_to_test_{session_to_test}-train_{train}.npy", np.array(all_data))
 
