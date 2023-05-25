import os
import argparse
import glob
import random
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
from train import train
from test import test
from segmentation import lines_segmentation, words_segmentation
from create_lmdb_dataset import createDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HTR:
    def __init__(self,
                 exp_name=None,
                 manual_seed=1,
                 workers=0,
                 batch_size=192,
                 num_iter=300000,
                 saved_model='',
                 adam=None,
                 lr=1.0,
                 select_data='/',
                 feature_extraction='ResNet',
                 prediction='Attn'
                 ):
        self.select_data = select_data
        self.lr = lr
        self.adam = adam
        self.saved_model = saved_model
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.workers = workers
        self.eval_data = None
        self.valid_data = None
        self.train_data = None
        self.valInterval = 2000
        self.Prediction = prediction
        self.FeatureExtraction = feature_extraction
        self.manualSeed = manual_seed
        self.FT = True
        self.beta1 = 0.9
        self.rho = 0.95
        self.eps = 1e-8
        self.grad_clip = 5.0
        self.baiduCTC = False
        self.batch_ratio = '1'
        self.total_data_usage_ratio = '1.0'
        self.batch_max_length = 25
        self.imgH = 32
        self.imgW = 100
        self.rgb = False
        self.character = ',-.0123456789:;ІАБВГДЕЖЗИКЛМНОПРСТУФХЧШЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяіѣ'
        self.sensitive = True
        self.PAD = False
        self.data_filtering_off = False
        self.Transformation = 'TPS'
        self.SequenceModeling = 'BiLSTM'
        self.num_fiducial = 20
        self.input_channel = 1
        self.output_channel = 512
        self.hidden_size = 256
        self.patience = 20
        self.num_gpu = torch.cuda.device_count()
        self.num_class = None
        self.benchmark_all_eval = False
        if exp_name is not None:
            self.exp_name = exp_name
        else:
            self.exp_name = f'{self.Transformation}-{self.FeatureExtraction}-{self.SequenceModeling}-{self.Prediction}'
            self.exp_name += f'-Seed{self.manualSeed}'
    
    def fit(self,
            train_data='./result/train_data',
            valid_data='./result/valid_data'
            ):
        self.train_data = train_data
        self.valid_data = valid_data

        os.makedirs(f'./saved_models/{self.exp_name}', exist_ok=True)
        random.seed(self.manualSeed)
        np.random.seed(self.manualSeed)
        torch.manual_seed(self.manualSeed)
        torch.cuda.manual_seed(self.manualSeed)
        cudnn.benchmark = True
        cudnn.deterministic = True

        if self.num_gpu > 1:
            self.workers = self.workers * self.num_gpu
            self.batch_size = self.batch_size * self.num_gpu
        
        train(self)
    
    def translate(self, input_path, save_path, saved_model):
        print('Segmentation stage...')
        sys.stdin.reconfigure(encoding='utf-8')
        sys.stdout.reconfigure(encoding='utf-8')
        lines_segmentation(input_path)
        for i in range(len(os.listdir('./lines_dataset/'))):
            path_for_image = f'./lines_dataset/{i}_line.jpg'
            path_for_save = './words_dataset/'
            words_segmentation(path_for_image, path_for_save)
            print('\r', f'Line {i} is ready', end='')

            with open('./words_gt.txt', 'w') as f:
                word_paths = sorted(os.listdir(path_for_save))
                for word_path in word_paths:
                    f.write(path_for_save[2:] + word_path + f'\t{i}\n')

            self.eval_data = './lmdb_dataset'
            if not os.path.isdir(self.eval_data):
                os.mkdir(self.eval_data)
            createDataset('./', './words_gt.txt', self.eval_data)
            
            self.saved_model = saved_model
            preds = test(self)
            
            with open(save_path, 'a') as f:
                for word in preds:
                    word = word.replace('ё', 'я')
                    f.write(word + ' ')
                f.write('\n')

            files = glob.glob('./words_dataset/*')
            for f in files:
                os.remove(f)
        files = glob.glob('./lines_dataset/*')
        for f in files:
            os.remove(f)
        files = glob.glob('./lmdb_dataset/*')
        for f in files:
            os.remove(f)
        os.remove('./words_gt.txt')
        os.rmdir('./words_dataset/')
        os.rmdir('./lines_dataset/')
        os.rmdir('./lmdb_dataset/')
        print('\nDone')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Path to image which you want to translate', default='../demo/demo1.jpg')
    parser.add_argument('--output', help='File name, where you want to save the translate', default='../translate.txt')
    opt = parser.parse_args()
    
    model = HTR(exp_name='TEST_OOP')
    model.translate(opt.input,
                    opt.output,
                    './saved_models/ResNet_both_final/best_accuracy.pth')
