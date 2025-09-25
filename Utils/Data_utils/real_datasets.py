import os
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask


class CustomDataset(Dataset):
    def __init__(
        self, 
        name,
        data_root,
        window=64, 
        proportion=0.8, 
        save2npy=True, 
        neg_one_to_one=True,
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        style='separate', 
        distribution='geometric', 
        mean_mask_length=3,
        predict_step = None,
        ori_data_root=None,
        rawdata=None
    ):
        super(CustomDataset, self).__init__()
        assert period in ['train', 'test','longpredict'], 'period must be train or test.'
        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        if rawdata is not None:
            self.rawdata = rawdata
            self.scaler = self.read_predict_data(ori_data_root,self.name)
        elif period == 'longpredict':
            self.rawdata, self.scaler = self.read_initial_predict_data(data_root, ori_data_root, self.name)
        else:
            self.rawdata, self.scaler = self.read_data(data_root, self.name)
        # self.rawdata, self.scaler = self.read_data(data_root, self.name)
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        if period == 'longpredict' or rawdata is not None:
            self.window, self.period = window, period
            self.var_num = self.rawdata.shape[-1]
            self.save2npy = save2npy
            self.auto_norm = neg_one_to_one
            N,S,F = self.rawdata.shape
            data = self.scaler.transform(self.rawdata.reshape(-1, F)).reshape(N, S, F)
            if self.auto_norm:
                data = normalize_to_neg_one_to_one(data)
            self.samples = data
        else:
            self.window, self.period = window, period
            self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
            self.sample_num_total = max(self.len - self.window + 1, 0)
            self.save2npy = save2npy
            self.auto_norm = neg_one_to_one

            self.data = self.__normalize(self.rawdata)
            train, inference = self.__getsamples(self.data, proportion, seed)

            self.samples = train if period == 'train' else inference
        if period == 'test' or period == 'longpredict':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()
        self.sample_num = self.samples.shape[0]

    def __getsamples(self, data, proportion, seed):
        x = np.zeros((self.sample_num_total, self.window, self.var_num))
        for i in range(self.sample_num_total):
            start = i
            end = i + self.window
            x[i, :, :] = data[start:end, :]

        train_data, test_data = self.divide(x, proportion, seed)

        if self.save2npy:
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(test_data))
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), self.unnormalize(train_data))
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(test_data))
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), unnormalize_to_zero_to_one(train_data))
            else:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data)

        return train_data, test_data

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)
    
    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)
    
    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        # id_rdm = np.random.permutation(size)
        id_rdm = np.arange(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        df = pd.read_csv(filepath, header=0)
        data = df.values
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler

    @staticmethod
    def read_initial_predict_data(filepath,filepath_whole,name=''):
        df = pd.read_csv(filepath, header=0)
        data = df.values
        df1 = pd.read_csv(filepath_whole, header=0)
        data1 = df1.values
        scaler = MinMaxScaler()
        scaler = scaler.fit(data1)
        data = np.tile(data, (10, 1, 1))
        return data,scaler

    @staticmethod
    def read_predict_data(filepath_whole,name=''): # num * step * feature_n
        df1 = pd.read_csv(filepath_whole, header=0)
        data1 = df1.values
        scaler = MinMaxScaler()
        scaler = scaler.fit(data1)
        return scaler
    
    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period in ['test','longpredict'] :
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num
    

# class fMRIDataset(CustomDataset):
#     def __init__(
#         self,
#         proportion=1.,
#         **kwargs
#     ):
#         super().__init__(proportion=proportion, **kwargs)
#
#     @staticmethod
#     def read_data(filepath, name=''):
#         """Reads a single .csv
#         """
#         data = io.loadmat(filepath + '/sim4.mat')['ts']
#         scaler = MinMaxScaler()
#         scaler = scaler.fit(data)
#         return data, scaler
