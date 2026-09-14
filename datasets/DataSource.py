# five-fold split for ODV JVQD and 360SVQD
import pickle

import numpy as np
import scipy.io as sio

class ODVSource():
    def __init__(self):
        self.__files = self.__gen_file_name()
        self.__dmos = self.__get_dmos()
        self.fiveFolds=self.train_test_5fold()

    def train_test_5fold(self):

        num_videos=len(self.__files)
        index = np.array(range(num_videos), dtype=np.int32)
        index = index.astype(int).flatten()
        folds=[]
        fold_indexs=[]

        fold_indexs.append(index[0::5])
        fold_indexs.append(index[1::5])
        fold_indexs.append(index[2::5])
        fold_indexs.append(index[3::5])
        fold_indexs.append(index[4::5])
        del index

        # fold 1
        fold={}
        train_index = np.concatenate((fold_indexs[0],fold_indexs[1],fold_indexs[2],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[4]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 2
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[1],fold_indexs[2],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[0]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 3
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[0],fold_indexs[2],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[1]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 4
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[0],fold_indexs[1],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[2]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 5
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[0],fold_indexs[1],fold_indexs[2],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[3]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        return folds

    def __gen_file_name(self):
        nameset = []
        mat = sio.loadmat('./datasets/ODV_dmos_sorted.mat')
        name = mat['videos']
        for i in range(540):
            tmp=name[i]
            tmp = tmp.replace(' ', '')
            tmp=tmp.replace('.yuv', '.mp4')
            nameset.append(tmp)
        return nameset

    def __get_dmos(self):
        mat = sio.loadmat('./datasets/ODV_dmos_sorted.mat')
        dmos = mat['dmos']
        dmos = np.array(dmos).flatten()/100.0
        return dmos

    def __image_indexing(self, index):
        image_new = []
        for i in index:
            image_new.append(self.__files[i])
        return image_new

class JVQDSource():
    def __init__(self):
        self.files = self.__gen_file_name()
        self.__dmos = self.__get_dmos()
        self.fiveFolds = self.train_test_5fold()

    def train_test_5fold(self):

        num_videos=len(self.files)
        index = np.array(range(num_videos), dtype=np.int32)
        index = index.astype(int).flatten()
        folds=[]
        fold_indexs=[]

        fold_indexs.append(index[0::5])
        fold_indexs.append(index[1::5])
        fold_indexs.append(index[2::5])
        fold_indexs.append(index[3::5])
        fold_indexs.append(index[4::5])
        del index

        # fold 1
        fold = {}
        train_index = np.concatenate((fold_indexs[0], fold_indexs[1], fold_indexs[2], fold_indexs[3],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[4]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 2
        fold = {}
        train_index = np.concatenate((fold_indexs[4], fold_indexs[1], fold_indexs[2], fold_indexs[3],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[0]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 3
        fold = {}
        train_index = np.concatenate((fold_indexs[4], fold_indexs[0], fold_indexs[2], fold_indexs[3],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[1]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 4
        fold = {}
        train_index = np.concatenate((fold_indexs[4], fold_indexs[0], fold_indexs[1], fold_indexs[3],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[2]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 5
        fold = {}
        train_index = np.concatenate((fold_indexs[4], fold_indexs[0], fold_indexs[1], fold_indexs[2],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[3]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        return folds

    def __gen_file_name(self):
        nameset = []
        mat = sio.loadmat('./datasets/JVQD2.mat')
        name = mat['videos']
        for i in range(60):
            tmp=name[i]
            tmp = tmp.replace(' ', '').replace('\'', '')
            nameset.append(tmp)
        return nameset

    def __get_dmos(self):
        mat = sio.loadmat('./datasets/JVQD2.mat')
        dmos = mat['Jmos']
        dmos = np.array(dmos).flatten()/100.0
        return dmos

    def __image_indexing(self, index):
        image_new = []
        for i in index:
            image_new.append(self.files[i])
        return image_new

class SVQDSource():
    def __init__(self):
        self.files = self.__gen_file_name()
        self.__dmos = self.__get_dmos()
        self.fiveFolds = self.train_test_5fold()

    def train_test_5fold(self):

        num_videos=len(self.files)
        index = np.array(range(num_videos), dtype=np.int32)
        index = index.astype(int).flatten()
        folds=[]
        fold_indexs=[]

        fold_indexs.append(index[0::5])
        fold_indexs.append(index[1::5])
        fold_indexs.append(index[2::5])
        fold_indexs.append(index[3::5])
        fold_indexs.append(index[4::5])
        del index

        # fold 1
        # fold 1
        fold = {}
        train_index = np.concatenate((fold_indexs[0], fold_indexs[1], fold_indexs[2], fold_indexs[3],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[4]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 2
        fold = {}
        train_index = np.concatenate((fold_indexs[4], fold_indexs[1], fold_indexs[2], fold_indexs[3],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[0]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 3
        fold = {}
        train_index = np.concatenate((fold_indexs[4], fold_indexs[0], fold_indexs[2], fold_indexs[3],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[1]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 4
        fold = {}
        train_index = np.concatenate((fold_indexs[4], fold_indexs[0], fold_indexs[1], fold_indexs[3],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[2]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 5
        fold = {}
        train_index = np.concatenate((fold_indexs[4], fold_indexs[0], fold_indexs[1], fold_indexs[2],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[3]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        return folds

    def __gen_file_name(self):
        nameset = []
        mat = sio.loadmat('./datasets/360SVQD.mat')
        name = mat['name']
        for i in range(64):
            tmp=name[i]
            tmp = tmp+'.mkv'
            nameset.append(tmp)
        return nameset

    def __get_dmos(self):
        mat = sio.loadmat('./datasets/360SVQD.mat')
        dmos = mat['mos']
        dmos = np.array(dmos).flatten()/5.0
        return dmos

    def __image_indexing(self, index):
        image_new = []
        for i in index:
            image_new.append(self.files[i])
        return image_new

class VRVQWSource():
    def __init__(self):
        self.__files = self.__gen_file_name()
        self.__files=np.array(self.__files)
        self.__dmos = self.__get_dmos()
        index=np.argsort(self.__dmos)
        self.__files=self.__files[index]
        self.__dmos=self.__dmos[index]
        self.fiveFolds=self.train_test_5fold()
        self.files=self.__files
        self.dmos=self.__dmos

    def train_test_5fold(self):

        num_videos=len(self.__files)
        index = np.array(range(num_videos), dtype=np.int32)
        index = index.astype(int).flatten()
        folds=[]
        fold_indexs=[]

        fold_indexs.append(index[0::5])
        fold_indexs.append(index[1::5])
        fold_indexs.append(index[2::5])
        fold_indexs.append(index[3::5])
        fold_indexs.append(index[4::5])
        del index

        # fold 1
        fold={}
        train_index = np.concatenate((fold_indexs[0],fold_indexs[1],fold_indexs[2],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[4]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 2
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[1],fold_indexs[2],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[0]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 3
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[0],fold_indexs[2],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[1]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 4
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[0],fold_indexs[1],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[2]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 5
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[0],fold_indexs[1],fold_indexs[2],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[3]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        return folds

    def __gen_file_name(self):
        nameset = []
        mat = pickle.load(open('./datasets/vrvqw.pkl','rb'))
        name = mat['vname']
        for i in range(502):
            tmp=name[i]
            tmp = tmp.replace(' ', '')
            tmp=tmp.replace('.yuv', '.mp4')
            nameset.append(tmp)
        return nameset

    def __get_dmos(self):
        mat =  pickle.load(open('./datasets/vrvqw.pkl','rb'))
        dmosA = mat['mosA']
        dmosA = np.array(dmosA).flatten()/5.0
        dmosB = mat['modB']
        dmosB = np.array(dmosB).flatten() / 5.0
        return (dmosA+dmosB)/2

    def __image_indexing(self, index):
        image_new = []
        for i in index:
            image_new.append(self.__files[i])
        return image_new
