import numpy as np
import h5py
import pdb 
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import random

def random_sample(list1, list2, list3):
    combined = list(zip(list1, list2, list3))
    random.shuffle(combined)
    list1, list2, list3 = zip(*combined)
    
    list1 = list(list1)
    list2 = list(list2)
    list3 = list(list3)

    return list1, list2, list3

class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

def get_clean_and_noisy_index(dataset,noise_rate):
    
    if dataset == 'wiki':
        noise = h5py.File('./data/partial/WIKI-lall-partial_{}.h5'.format(noise_rate))
    elif dataset == "inria":
        noise = h5py.File('./data/partial/INRIA-lall-partial_{}.h5'.format(noise_rate))
    elif dataset == "nus":
        noise = h5py.File('./data/partial/NUS-lall-partial_{}.h5'.format(noise_rate))
    elif dataset == "xmedia":
        noise = h5py.File('./data/partial/XMEDIA-lall-partial_{}.h5'.format(noise_rate))
    

    fl = list(noise['True'])
    ffl = list(noise['result'])
    clean_index = []
    noisy_index = []

    for i in range(len(fl)):
        equal = True
        #pdb.set_trace()
        for j in range(len(fl[i])):
            if fl[i][j] != ffl[i][j]:
                equal=False
        if equal:
            clean_index.append(i)
            
        else:
            noisy_index.append(i)

    #pdb.set_trace()
    return clean_index, noisy_index

class DataList(object):
    def __init__(self, dataset, data_type, transform, noise_type, noise_rate, random_state):
        self.data_type = data_type
        if dataset == "wiki":
            data = h5py.File("./data/tool1215/WIKI_partial.h5", "r")
            noise = h5py.File(
                "./data/partial/WIKI-lall-partial_{}.h5".format(noise_rate)
            )
        elif dataset == "nus":
            data = h5py.File("./data/tool1215/NUS_partial.h5", "r", driver="core")
            noise = h5py.File(
                "./data/partial/NUS-lall-partial_{}.h5".format(noise_rate)
            )
            
        elif dataset == "inria":
            data = h5py.File("./data/tool1215/INRIA_partial.h5", "r", driver="core")
            noise = h5py.File(
                "./data/partial/INRIA-lall-partial_{}.h5".format(noise_rate)
            )
        elif dataset == "xmedia":
            data = h5py.File("./data/tool1215/XMEDIA_partial.h5", "r", driver="core")
            noise = h5py.File(
                "./data/partial/XMEDIA-lall-partial_{}.h5".format(noise_rate)
            )
        ##########################################################################
        if data_type == "train":
            fi = list(data['ImgTrain'])
            fl = list(data['LabTrain'])
            ffl = list(noise['result'])
            ft = list(data['TagTrain'])
            self.imgs = fi
            self.labs = fl
            self.flabs = ffl
            self.tags = ft
            lab = self.labs[1]
            lab = lab.astype(int)
        elif data_type == "test":
            fi = list(data['ImgQuery'])
            fl = list(data['LabQuery'])
            ft = list(data['TagQuery'])
            self.imgs = fi
            self.labs = fl
            self.tags = ft
        elif data_type == "database":
            fi = list(data['ImgDataBase'])
            fl = list(data['LabDataBase'])
            ft = list(data['TagDataBase'])
            self.imgs = fi
            self.labs = fl
            self.tags = ft
        self.transform = transform
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.random_state = random_state

    def __getitem__(self, index):
        img = self.imgs[index]
        img = img.astype(np.float32)
        lab = self.labs[index]
        lab = lab.astype(int)
        tlab = lab
        if self.data_type == "train":
            lab = self.flabs[index]
            lab = lab.astype(int)
        tag = self.tags[index]
        tag = tag.astype(np.float32)
        return img, tag, tlab, lab, index

    def __len__(self):
        return len(self.imgs)



def SaveH5File_WIKI(resize_size):
    database_size = 2173 + 462
    train_size = 2173
    query_size = 231

    root = './data/tool/wiki/'
    path = root + 'wiki_deep_doc2vec_data_corr_ae.h5py'
    data = h5py.File(path)
    fi = np.concatenate([data['train_imgs_deep'][()], data['test_imgs_deep'][()]], axis=0)
    fl = np.concatenate([data['train_imgs_labels'][()], data['test_imgs_labels'][()]], axis=0)
    ft = np.concatenate([data['train_text'][()], data['test_text'][()]], axis=0)

    FL = []
    for item in fl:
        one_hot = np.zeros(10)
        one_hot[item] = 1
        FL.append(one_hot)
    fl = FL

    fi, fl, ft = random_sample(fi, fl, ft)

    imgs = list(fi[query_size: query_size + train_size])
    labs = list(fl[query_size: query_size + train_size])
    tags = list(ft[query_size: query_size + train_size])

    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,10])
    Tag = np.zeros([n,300])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf = h5py.File('./data/tool/WIKI_partial.h5','w')
    hf.create_dataset('ImgTrain', data = Img)
    hf.create_dataset('TagTrain', data = Tag)
    hf.create_dataset('LabTrain', data = Lab)

    ##测试
    imgs = list(fi[: query_size])
    labs = list(fl[: query_size])
    tags = list(ft[: query_size])

    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,10])
    Tag = np.zeros([n,300])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        #pdb.set_trace()
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgQuery', data = Img)
    hf.create_dataset('TagQuery', data = Tag)
    hf.create_dataset('LabQuery', data = Lab)

    imgs = list(fi[query_size::])
    labs = list(fl[query_size::])
    tags = list(ft[query_size::])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,10])
    Tag = np.zeros([n,300])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        #pdb.set_trace()
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgDataBase', data = Img)
    hf.create_dataset('TagDataBase', data = Tag)
    hf.create_dataset('LabDataBase', data = Lab)
    hf.close()

def SaveH5File_NUS(resize_size):
    database_size = 42941 + 23661
    train_size = 23661
    query_size = 5000

    root = './data/tool/nus/'
    path = root + 'nus_wide_deep_doc2vec_data_42941.h5py'
    data = h5py.File(path)
    fi = np.concatenate([data['train_imgs_deep'][()], data['test_imgs_deep'][()]], axis=0)
    fl = np.concatenate([data['train_imgs_labels'][()], data['test_imgs_labels'][()]], axis=0)
    ft = np.concatenate([data['train_text'][()], data['test_text'][()]], axis=0)

    FL = []
    for item in fl:
        one_hot = np.zeros(10)
        one_hot[item-1] = 1
        FL.append(one_hot)
    fl = FL

    fi, fl, ft = random_sample(fi, fl, ft)

    imgs = list(fi[query_size: query_size + train_size])
    labs = list(fl[query_size: query_size + train_size])
    tags = list(ft[query_size: query_size + train_size])

    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,10])
    Tag = np.zeros([n,300])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf = h5py.File('./data/tool/NUS_partial.h5','w')
    hf.create_dataset('ImgTrain', data = Img)
    hf.create_dataset('TagTrain', data = Tag)
    hf.create_dataset('LabTrain', data = Lab)

    imgs = list(fi[: query_size])
    labs = list(fl[: query_size])
    tags = list(ft[: query_size])

    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,10])
    Tag = np.zeros([n,300])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        #pdb.set_trace()
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgQuery', data = Img)
    hf.create_dataset('TagQuery', data = Tag)
    hf.create_dataset('LabQuery', data = Lab)

    imgs = list(fi[query_size::])
    labs = list(fl[query_size::])
    tags = list(ft[query_size::])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,10])
    Tag = np.zeros([n,300])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        #pdb.set_trace()
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgDataBase', data = Img)
    hf.create_dataset('TagDataBase', data = Tag)
    hf.create_dataset('LabDataBase', data = Lab)
    hf.close()

def SaveH5File_INRIA(resize_size):
    database_size = 9000+4366
    train_size = 4366
    query_size = 1332

    path = './data/tool/inria/INRIA-Websearch.mat'
    data = sio.loadmat(path)

    fi = np.concatenate([data['tr_img'].astype('float32'),
                         data['val_img'].astype('float32'),
                         data['te_img'].astype('float32') ], axis=0)
    fl = np.concatenate([data['tr_img_lab'].reshape([-1]).astype('int64'),
                         data['val_img_lab'].reshape([-1]).astype('int64'),
                         data['te_img_lab'].reshape([-1]).astype('int64') ], axis=0)
    ft = np.concatenate([data['tr_txt'].astype('float32'),
                         data['val_txt'].astype('float32'),
                         data['te_txt'].astype('float32')], axis=0)
    # fi = np.concatenate([data['train_imgs_deep'][()], data['test_imgs_deep'][()]], axis=0)
    # fl = np.concatenate([data['train_imgs_labels'][()], data['test_imgs_labels'][()]], axis=0)
    # fl = one_hot_labels(fl)
    FL = []
    for item in fl:
        one_hot = np.zeros(100)
        one_hot[item] = 1
        FL.append(one_hot)
    fl = FL

    fi, fl, ft = random_sample(fi, fl, ft)

    imgs = list(fi[query_size: query_size + train_size])
    labs = list(fl[query_size: query_size + train_size])
    tags = list(ft[query_size: query_size + train_size])

    # LABS = []
    # for item in labs:
    #     one_hot = np.zeros(200)
    #     one_hot[item-1] = 1
    #     LABS.append(one_hot)
    # labs = LABS

    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,100])
    Tag = np.zeros([n,1000])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf = h5py.File('./data/tool/INRIA_partial.h5','w')
    hf.create_dataset('ImgTrain', data = Img)
    hf.create_dataset('TagTrain', data = Tag)
    hf.create_dataset('LabTrain', data = Lab)

    imgs = list(fi[: query_size])
    labs = list(fl[: query_size])
    tags = list(ft[: query_size])

    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,100])
    Tag = np.zeros([n,1000])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        #pdb.set_trace()
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgQuery', data = Img)
    hf.create_dataset('TagQuery', data = Tag)
    hf.create_dataset('LabQuery', data = Lab)

    imgs = list(fi[query_size::])
    labs = list(fl[query_size::])
    tags = list(ft[query_size::])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,100])
    Tag = np.zeros([n,1000])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        #pdb.set_trace()
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgDataBase', data = Img)
    hf.create_dataset('TagDataBase', data = Tag)
    hf.create_dataset('LabDataBase', data = Lab)
    hf.close()

def SaveH5File_XM(resize_size):
    database_size = 4000+500
    train_size = 4000
    query_size = 500

    
    path = './data/tool/xmedia/XMediaFeatures.mat'
    data = sio.loadmat(path)

    fi = np.concatenate([data['I_te_CNN'].astype('float32'),
                         data['I_tr_CNN'].astype('float32')], axis=0)
    
    fl = np.concatenate([data['teImgCat'].reshape([-1]).astype('int64'),
                         data['trImgCat'].reshape([-1]).astype('int64')], axis=0)
    fl = fl-1
    
    ft = np.concatenate([data['T_te_BOW'].astype('float32'),
                         data['T_tr_BOW'].astype('float32')], axis=0)
    # fi = np.concatenate([data['train_imgs_deep'][()], data['test_imgs_deep'][()]], axis=0)
    # fl = np.concatenate([data['train_imgs_labels'][()], data['test_imgs_labels'][()]], axis=0)
    # fl = one_hot_labels(fl)
    FL = []
    for item in fl:
        one_hot = np.zeros(20)
        one_hot[item] = 1
        FL.append(one_hot)
    fl = FL

    fi, fl, ft = random_sample(fi, fl, ft)

    imgs = list(fi[query_size: query_size + train_size])
    labs = list(fl[query_size: query_size + train_size])
    tags = list(ft[query_size: query_size + train_size])

    # LABS = []
    # for item in labs:
    #     one_hot = np.zeros(200)
    #     one_hot[item-1] = 1
    #     LABS.append(one_hot)
    # labs = LABS

    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,20])
    Tag = np.zeros([n,3000])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf = h5py.File('./data/tool/XMEDIA_partial.h5','w')
    hf.create_dataset('ImgTrain', data = Img)
    hf.create_dataset('TagTrain', data = Tag)
    hf.create_dataset('LabTrain', data = Lab)

    imgs = list(fi[: query_size])
    labs = list(fl[: query_size])
    tags = list(ft[: query_size])

    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,20])
    Tag = np.zeros([n,3000])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        #pdb.set_trace()
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgQuery', data = Img)
    hf.create_dataset('TagQuery', data = Tag)
    hf.create_dataset('LabQuery', data = Lab)

    imgs = list(fi[query_size::])
    labs = list(fl[query_size::])
    tags = list(ft[query_size::])
    n = len(imgs)
    Img = np.zeros([n,4096])
    Lab = np.zeros([n,20])
    Tag = np.zeros([n,3000])
    for i in tqdm(range(n)):
        img_i = imgs[i]
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:] = img_i
        #pdb.set_trace()
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgDataBase', data = Img)
    hf.create_dataset('TagDataBase', data = Tag)
    hf.create_dataset('LabDataBase', data = Lab)
    hf.close()
    ##################################################################################
    ##################################################################################
    ##################################################################################

def get_data(config):
    dsets = {}
    dset_loaders = {}

    for data_type in ["train", "test", "database"]:
        dsets[data_type] = DataList(config["dataset"], data_type,
                                    transforms.ToTensor(), config["noise_type"], config["noise_rate"], config["random_state"])
        print(data_type, len(dsets[data_type]))
        dset_loaders[data_type] = util_data.DataLoader(dsets[data_type],
                                                      batch_size=config["batch_size"],
                                                      shuffle=True, num_workers=2)

    return dset_loaders["train"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train"]), len(dsets["test"]), len(dsets["database"])

def compute_img_result(dataloader, net, device):
    bs, tclses, clses = [], [], []
    net.eval()
    for img, tag, tcls, cls, _ in tqdm(dataloader):
        tclses.append(tcls)
        clses.append(cls)
        bs.append((net(img.to('cuda')))[0].data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)


def compute_tag_result(dataloader, net, device):
    bs, tclses, clses = [], [], []
    net.eval()
    for img, tag, tcls, cls, _ in tqdm(dataloader):
        tclses.append(tcls)
        clses.append(cls)
        tag = tag.float()
        bs.append((net(tag.to('cuda')))[0].data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)


def CalcHammingDist(B1, B2):
    # B1=B1.cpu()
    # B2=B2.cpu()
    q = B2.shape[1]
    distH = 0.5 * (q - torch.matmul(B1, B2.transpose(0, 1)))

    return distH


def calc_map_k(
    rB, qB, retrieval_label, query_label, k=None, device=torch.device("cuda:0")
):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # sim: {0, 1}^{mxn}
    # check if qB is a numpy array and convert it to a torch tensor if it is

    if isinstance(qB, np.ndarray):
        qB = torch.from_numpy(qB)
    # check if rB is a numpy array and convert it to a torch tensor if it is
    if isinstance(rB, np.ndarray):
        rB = torch.from_numpy(rB)
    # check if query_label is a numpy array and convert it to a torch tensor if it is
    if isinstance(query_label, np.ndarray):
        query_label = torch.from_numpy(query_label)
    # check if retrieval_label is a numpy array and convert it to a torch tensor if it is
    if isinstance(retrieval_label, np.ndarray):
        retrieval_label = torch.from_numpy(retrieval_label)
    qB, rB, query_label, retrieval_label = [
        i.to('cuda').to(dtype=torch.float64)
        for i in [qB, rB, query_label, retrieval_label]
    ]
    num_query = query_label.shape[0]
    map = 0.0
    GND = (
        (query_label.mm(retrieval_label.t()) > 0).type(torch.float).squeeze().to('cuda')
    )
    if k is None:
        k = retrieval_label.shape[0]
    sum_query = num_query
    for iter in tqdm(range(num_query)):
        # gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        gnd = GND[iter, :]
        tsum = torch.sum(gnd)
        if tsum == 0:
            sum_query -= 1
            continue
        hamm = CalcHammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        # count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        count = torch.arange(1, total + 1).type(torch.float)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count.to('cuda') / tindex.to('cuda'))
    map = map / sum_query
    return map

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device=torch.device("cuda:0")
def pr_curve(rB, qB, retrieval_L, query_L, device=torch.device("cuda:0")):
    if isinstance(qB, np.ndarray):
        qB = torch.from_numpy(qB)
    # check if rB is a numpy array and convert it to a torch tensor if it is
    if isinstance(rB, np.ndarray):
        rB = torch.from_numpy(rB)
    # check if query_label is a numpy array and convert it to a torch tensor if it is
    if isinstance(query_L, np.ndarray):
        query_L = torch.from_numpy(query_L)
    # check if retrieval_label is a numpy array and convert it to a torch tensor if it is
    if isinstance(retrieval_L, np.ndarray):
        retrieval_L = torch.from_numpy(retrieval_L)
    qB, rB, query_L, retrieval_L = [
        i.to('cuda').to(dtype=torch.float64) for i in [qB, rB, query_L, retrieval_L]
    ]
    num_query = query_L.shape[0]
    topK = retrieval_L.shape[0]
    query_L = query_L.float().to(device)
    retrieval_L = retrieval_L.float().to(device)
    qB = qB.float().to(device)
    rB = rB.float().to(device)
    GND = (query_L.mm(retrieval_L.t()) > 0).type(torch.float).squeeze()
    # hamm = calc_hamming_dist(qB, rB)
    # _, ind = torch.sort(hamm,dim=1)
    P, R = [], []
    # np.linspace(1,topK+1,20)
    # tqdm(range(num_query))
    for k in tqdm(np.linspace(1, topK + 1, 50)):
        k = int(k)
        # ground-truth: 1 vs all
        p = torch.zeros(num_query)
        r = torch.zeros(num_query)
        for it in range(num_query):
            hamm = CalcHammingDist(qB[it, :], rB)
            _, ind = torch.sort(hamm)
            ind = ind.squeeze()
            p[it] = (GND[it][ind[:k]] != 0).sum() / k
            r[it] = (GND[it][ind[:k]] != 0).sum() / (GND[it] != 0).sum()
            if (GND[it] != 0).sum() == 0:
                # print(1)
                pass
        P.append((p.mean()).item())
        R.append((r.mean()).item())
    # return R,P
    return R, P


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


def TCalcTopMap(rB, qB, retrievalL, queryL, topk, tretrievalL, tqueryL):
    num_query = queryL.shape[0]
    topkmap = 0
    temp_ind = 0
    for iter in tqdm(range(num_query)):
        if np.dot(tqueryL[iter, :], queryL[iter, :].transpose()) > 0:
            gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(
                np.float32
            )
            # Cgnd = (np.dot(tqueryL[iter, :], tretrievalL.transpose()) > 0).astype(np.float32)
            hamm = CalcHammingDist(qB[iter, :], rB)
            ind = np.argsort(hamm)
            gnd = gnd[ind]
            # cgnd = Cgnd[ind]

            tgnd = gnd[0:topk]
            # Ntgnd = Ngnd[0:topk]
            tsum = np.sum(tgnd).astype(int)
            if tsum == 0:
                continue
            count = np.linspace(1, tsum, tsum)

            tindex = np.asarray(np.where(tgnd == 1)) + 1.0
            topkmap_ = np.mean(count / (tindex))
            topkmap = topkmap + topkmap_
            temp_ind += 1
    cor_topkmap = topkmap / temp_ind

    topkmap = 0
    temp_ind = 0
    for iter in tqdm(range(num_query)):
        if np.dot(tqueryL[iter, :], queryL[iter, :].transpose()) == 0:
            gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(
                np.float32
            )
            hamm = CalcHammingDist(qB[iter, :], rB)
            ind = np.argsort(hamm)
            gnd = gnd[ind]

            tgnd = gnd[0:topk]
            tsum = np.sum(tgnd).astype(int)
            if tsum == 0:
                continue
            count = np.linspace(1, tsum, tsum)

            tindex = np.asarray(np.where(tgnd == 1)) + 1.0
            topkmap_ = np.mean(count / (tindex))
            topkmap = topkmap + topkmap_
            temp_ind += 1
    oth_topkmap = topkmap / (temp_ind + 0.0001)
    return cor_topkmap, oth_topkmap



if __name__ == "__main__":
    SaveH5File_WIKI(256)
    SaveH5File_NUS(256)
    SaveH5File_INRIA(256)
    SaveH5File_XM(256)
