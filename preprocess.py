import numpy as np
from tqdm import tqdm
from dataset import IEMOCAPDataset 

def preprocess(preprocessed_path, df, features = None, session_to_test= None, train= None, wa2v_weights_path=None):
    x_features_train = IEMOCAPDataset(data_root=df, features_name=features, session_to_test=session_to_test, train=train, wa2v_weights_path=wa2v_weights_path)
    len = x_features_train.__len__()
    
    all_data = []
    for i in tqdm(range(len)):
        line = x_features_train.iemocap_table["wav_path"].iloc[i]
        audio, sr = x_features_train.load_wav(line)
        data = x_features_train.extract_features(audio, sr).transpose()
        all_data.append(data)

    np.save(f"{preprocessed_path}/{features}-session_to_test_{session_to_test}-train_{train}.npy", np.array(all_data))
 