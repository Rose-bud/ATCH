import os
import re
import argparse
from utils.tools_partial import *
import itertools
from scipy.linalg import hadamard
from network import *
import pdb
import os

import torch
import torch.optim as optim
import time
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pickle
import ATCH_loss


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description="manual to this script")
parser.add_argument("--gpus", type=str, default="0")
parser.add_argument("--hash_dim", type=int, default=32)
parser.add_argument("--noise_rate", type=float, default=1.0)
parser.add_argument("--dataset", type=str, default="flickr")
parser.add_argument("--Lambda", type=float, default=0.6)
parser.add_argument("--num_gradual", type=int, default=100)
parser.add_argument("--log", type=str, default="default")

# parser.add_argument("--w1", type=float, default=1)
# # parser.add_argument("--w2", type=float, default=0)
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

bit_len = args.hash_dim
noise_rate = args.noise_rate
dataset = args.dataset
Lambda = args.Lambda
num_gradual = args.num_gradual

train_size = 0
n_class = 0
tag_len = 0
torch.multiprocessing.set_sharing_strategy("file_system")


def get_config(datasetA):
    if datasetA == "nus":
        config = {
            "optimizer": {
                "type": optim.RMSprop,
                "optim_params": {"lr": 1e-4, "weight_decay": 1e-6},
            },
            "txt_optimizer": {
                "type": optim.RMSprop,
                "optim_params": {"lr": 1e-4, "weight_decay": 1e-6},
            },
            "info": "[CSQ]",
            "resize_size": 256,
            "crop_size": 224,
            "batch_size": 128,
            "dataset": dataset,
            "epoch": 50,
            "device": torch.device("cuda:"+args.gpus),
            "bit_len": bit_len,
            "noise_type": "symmetric",
            "noise_rate": noise_rate,
            "random_state": 1,
            "n_class": n_class,
            "lambda": Lambda,
            "tag_len": tag_len,
            "train_size": train_size,
            "threshold_rate": 0.3,
        }
    elif datasetA == "xmedia":
        config = {
            "optimizer": {
                "type": optim.RMSprop,
                "optim_params": {"lr": 1e-5, "weight_decay": 1e-6},
            },
            "txt_optimizer": {
                "type": optim.RMSprop,
                "optim_params": {"lr": 1e-5, "weight_decay": 1e-6},
            },
            "info": "[CSQ]",
            "resize_size": 256,
            "crop_size": 224,
            "batch_size": 128,
            "dataset": dataset,
            "epoch": 50,
            "device": torch.device("cuda:"+args.gpus),
            "bit_len": bit_len,
            "noise_type": "symmetric",
            "noise_rate": noise_rate,
            "random_state": 1,
            "n_class": n_class,
            "lambda": Lambda,
            "tag_len": tag_len,
            "train_size": train_size,
            "threshold_rate": 0.3,
        }
    elif datasetA == "inria":
        config = {
            "optimizer": {
                "type": optim.RMSprop,
                "optim_params": {"lr": 5e-4, "weight_decay": 1e-6},
            },
            "txt_optimizer": {
                "type": optim.RMSprop,
                "optim_params": {"lr": 5e-4, "weight_decay": 1e-6},
            },
            "info": "[CSQ]",
            "resize_size": 256,
            "crop_size": 224,
            "batch_size": 128,
            "dataset": dataset,
            "epoch": 50,
            "device": torch.device("cuda:"+args.gpus),
            "bit_len": bit_len,
            "noise_type": "symmetric",
            "noise_rate": noise_rate,
            "random_state": 1,
            "n_class": n_class,
            "lambda": Lambda,
            "tag_len": tag_len,
            "train_size": train_size,
            "threshold_rate": 0.3,
        }
    else:
        config = {
            "optimizer": {
                "type": optim.RMSprop,
                "optim_params": {"lr": 1e-4, "weight_decay": 1e-6},
            },
            "txt_optimizer": {
                "type": optim.RMSprop,
                "optim_params": {"lr": 1e-4, "weight_decay": 1e-6},
            },
            "info": "[CSQ]",
            "resize_size": 256,
            "crop_size": 224,
            "batch_size": 128,
            "dataset": dataset,
            "epoch": 100,
            "device": torch.device("cuda:"+args.gpus),
            "bit_len": bit_len,
            "noise_type": "symmetric",
            "noise_rate": noise_rate,
            "random_state": 1,
            "n_class": n_class,
            "lambda": Lambda,
            "tag_len": tag_len,
            "train_size": train_size,
            "threshold_rate": 0.3,
        }
    return config



def train(config, bit, seed):
    device = config["device"]
    # alpha = config["alpha"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = (
        get_data(config)
    )
    config["num_train"] = num_train
    net = ImgModule(y_dim=4096, bit=bit, hiden_layer=3, num_classes=n_class).to('cuda')
    txt_net = TxtModule(y_dim=tag_len, bit=bit, hiden_layer=2, num_classes=n_class).to('cuda')
    
    W = torch.Tensor(n_class, bit_len)
    W = torch.nn.init.orthogonal_(W, gain=1)
    W = torch.tensor(W, requires_grad=True).to('cuda')
    W = torch.nn.Parameter(W)
    net.register_parameter("W", W)  # regist W into the image net
    get_grad_params = lambda model: [x for x in model.parameters() if x.requires_grad]
    params_dnet = get_grad_params(net)
    optimizer = config["optimizer"]["type"](
        params_dnet, **(config["optimizer"]["optim_params"])
    )
    txt_optimizer = config["txt_optimizer"]["type"](
        txt_net.parameters(), **(config["txt_optimizer"]["optim_params"])
    )
    
    i2t_mAP_list = []
    t2i_mAP_list = []
    epoch_list = []
    precision_list = []
    bestt2i = 0
    besti2t = 0
    n = 0

    os.makedirs("./ATCH/checkpoint", exist_ok=True)
    os.makedirs("./ATCH/LOG/"+str(args.log), exist_ok=True)
    os.makedirs(".ATCH/PR", exist_ok=True)
    os.makedirs("./ATCH/map", exist_ok=True)



    with open(
        "./ATCH/LOG/"+str(args.log)+"/data_{}_seed_{}_noiseRate_{}_bit_{}.txt".format(
            config["dataset"],
            seed,
            config["noise_rate"],
            bit,
        ),
        "w",
    ) as f:
##############################################################
##############################################################
##############################################################
        label_sum, tlable_sum, look_sum= 0, 0, 0

        for image, tag, tlabel, label, ind in train_loader:
            look = label*tlabel
            label_sum += torch.sum(label).item()
            tlable_sum += torch.sum(tlabel).item()
            look_sum += torch.sum(look).item()
            
        uniform_confidence =  torch.zeros((look_sum, n_class))
        all_partial_label =  torch.zeros((look_sum, n_class))
        all_true_label =  torch.zeros((look_sum, n_class))

        all_u_partial = torch.zeros((look_sum, n_class))
        all_v_partial = torch.zeros((look_sum, n_class))

        with torch.no_grad():
            for image, tag, tlabel, label, ind in train_loader:
                all_partial_label[ind] = label.float()
                all_true_label[ind] = tlabel.float()

        all_partial_label = all_partial_label.detach()
        all_true_label = all_true_label.detach()

        with torch.no_grad():
            for img, txt, _, _, ind in train_loader:
                img = img.to('cuda')
                txt = txt.to('cuda')
                _, u_p = net(img)
                _, v_p = txt_net(txt)
                all_u_partial[ind] = u_p.detach().cpu()
                all_v_partial[ind] = v_p.detach().cpu()
    
        uniform_confidence = all_partial_label / torch.sum(all_partial_label, dim=1).reshape(-1, 1)
            
        uniform_confidence = uniform_confidence.to('cuda')
        confidence_img = uniform_confidence.to('cuda')

        all_partial_label, all_true_label, all_u_partial, all_v_partial = \
            all_partial_label.to('cuda'), all_true_label.to('cuda'), all_u_partial.to('cuda'), all_v_partial.to('cuda')
        
        all_partial_label_zero = all_partial_label.detach().clone()
        same_count_list = []
        different_count_list = []

        for epoch in range(config["epoch"]):
            current_time = time.strftime("%H:%M:%S", time.localtime(time.time()))
            print(
                "%s[%2d/%2d][%s] bit:%d, dataset:%s, training...."
                % (
                    config["info"],
                    epoch + 1,
                    config["epoch"],
                    current_time,
                    bit,
                    config["dataset"],
                ),
                end="",
            )
            net.eval()
            txt_net.eval()
            net.train()
            txt_net.train()
            train_loss = 0
            if (epoch + 1) % 1 == 0:
                print("calculating test binary code......")
                img_tst_binary, img_tst_label = compute_img_result(
                    test_loader, net, device=device
                )
                print("calculating dataset binary code.......")
                img_trn_binary, img_trn_label = compute_img_result(
                    dataset_loader, net, device=device
                )
                txt_tst_binary, txt_tst_label = compute_tag_result(
                    test_loader, txt_net, device=device
                )
                txt_trn_binary, txt_trn_label = compute_tag_result(
                    dataset_loader, txt_net, device=device
                )
                print("calculating map.......")
                ######################################################
                t2i_mAP = calc_map_k(
                    img_trn_binary.numpy(),
                    txt_tst_binary.numpy(),
                    img_trn_label.numpy(),
                    txt_tst_label.numpy(),
                    device=device,
                )

                i2t_mAP = calc_map_k(
                    txt_trn_binary.numpy(),
                    img_tst_binary.numpy(),
                    txt_trn_label.numpy(),
                    img_tst_label.numpy(),
                    device=device,
                )

                if t2i_mAP + i2t_mAP > bestt2i + besti2t:
                    bestt2i = t2i_mAP
                    besti2t = i2t_mAP
 
                t2i_mAP_list.append(t2i_mAP.item())
                i2t_mAP_list.append(i2t_mAP.item())
                ########################
                #######################
                # t2i_mAP = calc_map_k(
                #     img_trn_binary.numpy(),
                #     txt_tst_binary.numpy(),
                #     img_trn_label.numpy(),
                #     txt_tst_label.numpy(),
                #     device=device,
                # )
                                
                # t2i_r, t2i_p = pr_curve(
                #     img_trn_binary.numpy(),
                #     txt_tst_binary.numpy(),
                #     img_trn_label.numpy(),
                #     txt_tst_label.numpy(),
                #     device=device,
                # )
                # i2t_mAP = calc_map_k(
                #     txt_trn_binary.numpy(),
                #     img_tst_binary.numpy(),
                #     txt_trn_label.numpy(),
                #     img_tst_label.numpy(),
                #     device=device,
                # )
                # i2t_r, i2t_p = pr_curve(
                #     txt_trn_binary.numpy(),
                #     img_tst_binary.numpy(),
                #     txt_trn_label.numpy(),
                #     img_tst_label.numpy(),
                #     device=device,
                # )
                # if t2i_mAP + i2t_mAP > bestt2i + besti2t:
                #     bestt2i = t2i_mAP
                #     besti2t = i2t_mAP
                #     bestt2i_r = t2i_r
                #     bestt2i_p = t2i_p
                #     besti2t_r = i2t_r
                #     besti2t_p = i2t_p
                #     data_to_save = {
                #         "bestt2i_r": bestt2i_r,
                #         "bestt2i_p": bestt2i_p,
                #         "besti2t_r": besti2t_r,
                #         "besti2t_p": besti2t_p,
                #     }
                #     with open(
                #         "./ATCH/PR/data_{}_seed_{}_noiseRate_{}_bit_{}_best_PR.pkl".format(
                #             config["dataset"],
                #             seed,
                #             config["noise_rate"],
                #             bit,
                #         ),
                #         "wb",
                #     ) as f1:
                #         pickle.dump(data_to_save, f1)
                # t2i_mAP_list.append(t2i_mAP.item())
                # i2t_mAP_list.append(i2t_mAP.item())
                ####################
                ####################
                epoch_list.append(epoch)
                print(
                    "%s epoch:%d, bit:%d, dataset:%s,noise_rate:%.2f,t2i_mAP:%.3f, i2t_mAP:%.3f"
                    % (
                        config["info"],
                        epoch + 1,
                        bit,
                        config["dataset"],
                        config["noise_rate"],
                        t2i_mAP,
                        i2t_mAP,
                    )
                )
                f.writelines(
                    "%s epoch:%d, bit:%d, dataset:%s,noise_rate:%.2f,t2i_mAP:%.3f, i2t_mAP:%.3f\n"
                    % (
                        config["info"],
                        epoch + 1,
                        bit,
                        config["dataset"],
                        config["noise_rate"],
                        t2i_mAP,
                        i2t_mAP,
                    )
                )
            

            confidence_img_max = torch.argmax(confidence_img, dim=1)
            all_true_label_max = torch.argmax(all_true_label, dim=1)
            

            for image, tag, tlabel, label, ind in train_loader:
                ind_np = ind.cpu().numpy()
                if 1==1:
                    image = image.to('cuda')
                    image = image.float()
                    tag = tag.to('cuda')
                    tag = tag.float()
                    label = label.to('cuda')
                    optimizer.zero_grad()
                    txt_optimizer.zero_grad()
                    u, u_partial = net(image)
                    v, v_partial = txt_net(tag)
                    
                    loss1, loss2 = 0, 0

                    u,v,u_partial, v_partial = u.to('cuda'), v.to('cuda'), u_partial.to('cuda'), v_partial.to('cuda')
                    all_u_partial[ind_np] = u_partial.detach()
                    all_v_partial[ind_np] = v_partial.detach()

                    confidence_img[ind_np]= ATCH_loss.final_Confidence_Upadate(u_partial, v_partial, all_u_partial, all_v_partial, all_partial_label, confidence_img, ind_np)
                                                      
                    LCD_loss, weights1 = ATCH_loss.LCD_loss(u_partial, v_partial, all_partial_label, confidence_img, ind_np)
                    
                    my_nrch_loss = ATCH_loss.CACH_loss(config,bit)
                    CACH_loss = my_nrch_loss(u, v, all_partial_label[ind], confidence_img[ind], config)*(0.9+0.1*weights1)

                    loss1 = LCD_loss.mean()
                    loss2 = CACH_loss.mean()
                    
                    loss = 1*loss1 + 1*loss2
                    
                    train_loss += loss
                    loss.backward()
                    optimizer.step()
                    txt_optimizer.step()

            train_loss = train_loss / len(train_loader)
            print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))
            precision_list.append(train_loss)
            print(
                "%s epoch:%d, bit:%d, dataset:%s,noise_rate:%.2f\n"
                % (
                    config["info"],
                    epoch + 1,
                    bit,
                    config["dataset"],
                    config["noise_rate"],
                )
            )

            print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        f.writelines(
            f"best result : bit:{bit}, dataset:{config['dataset']}, noise_rate:{config['noise_rate']:.2f}, t2i_mAP:{bestt2i:.3f}, i2t_mAP:{besti2t:.3f}, average:{(besti2t + bestt2i) / 2.0 * 100.0:.3f}\n"
        )

        
        with open("./ATCH/LOG/"+str(args.log)+"/LOSS__data_{}_seed_{}_noiseRate_{}_bit_{}.txt".format(
            config["dataset"],
            seed,
            config["noise_rate"],
            bit,
        ),
        "w",
    ) as f:
            f.write("train loss: " + str(precision_list) + "\n")
    original_filename = "./ATCH/LOG/{}/data_{}_seed_{}_noiseRate_{}_bit_{}.txt".format(
        args.log, config["dataset"], seed, config["noise_rate"], bit
    )

    if not os.path.exists(original_filename):
        print(f"file does not exit: {original_filename}")
        exit()


    with open(original_filename, 'r') as f:
        content = f.read()
        match = re.search(r'average:\s*([\d.]+)', content)
        if not match:
            print(f"no average value: {original_filename}")
            exit()
        average_value = match.group(1).strip()

    
    base, ext = os.path.splitext(original_filename)
    new_filename = f"{base}_average_{average_value}{ext}"

    os.rename(original_filename, new_filename)
    print(f"Renamed: {original_filename} -> {new_filename}")   

 
        
def test(config, bit, model_path="./ATCH/checkpoint/best_model1.pth"):

    device = config["device"]
    _, test_loader, dataset_loader, _, _, _ = get_data(config)
    net = ImgModule(y_dim=4096, bit=bit, hiden_layer=3).to('cuda')
    txt_net = TxtModule(y_dim=tag_len, bit=bit, hiden_layer=2).to('cuda')
    W = torch.Tensor(n_class, bit_len)
    W = torch.nn.init.orthogonal_(W, gain=1)
    W = torch.tensor(W, requires_grad=True).to('cuda')
    W = torch.nn.Parameter(W)
    net.register_parameter("W", W)
    # Load the saved models
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint["net_state_dict"])
    txt_net.load_state_dict(checkpoint["txt_net_state_dict"])
    net.eval()
    txt_net.eval()
    print("calculating test binary code......")
    print("calculating test binary code......")
    img_tst_binary, img_tst_label = compute_img_result(test_loader, net, device=device)
    print("calculating dataset binary code.......")
    img_trn_binary, img_trn_label = compute_img_result(
        dataset_loader, net, device=device
    )
    txt_tst_binary, txt_tst_label = compute_tag_result(
        test_loader, txt_net, device=device
    )
    txt_trn_binary, txt_trn_label = compute_tag_result(
        dataset_loader, txt_net, device=device
    )
    print("calculating map.......")
    t2i_mAP = calc_map_k(
        img_trn_binary.numpy(),
        txt_tst_binary.numpy(),
        img_trn_label.numpy(),
        txt_tst_label.numpy(),
        device=device,
    )
    i2t_mAP = calc_map_k(
        txt_trn_binary.numpy(),
        img_tst_binary.numpy(),
        txt_trn_label.numpy(),
        img_tst_label.numpy(),
        device=device,
    )
    print("Test Results: t2i_mAP: %.3f, i2t_mAP: %.3f" % (t2i_mAP, i2t_mAP))


if __name__ == "__main__":
    
    data_name_list = ["wiki", "xmedia", "inria"]
    bit_list = [16,32,64,128]
    noise_rate_list = [0.1]

    for data_name in data_name_list:
        if data_name in ["wiki", "nus", "xmedia"]:
            noise_rate_list = [0.1,0.3,0.5]
        else:
            noise_rate_list = [0.01,0.03,0.05]
        for rand_num in [123]:
            for rate in noise_rate_list:
                for bit in bit_list:
                    setup_seed(rand_num)

                    bit_len = bit
                    noise_rate = rate
                    dataset = data_name
                    if dataset == 'xmedianet2views':
                        n_class = 200
                        tag_len = 300
                    elif dataset == 'wiki':
                        n_class = 10
                        tag_len = 300
                    elif dataset == 'nus':
                        n_class = 10
                        tag_len = 300

                    elif dataset == 'inria':
                        n_class = 100
                        tag_len = 1000

                    elif dataset == 'xmedia':
                        n_class = 20
                        tag_len = 3000

                    config = get_config(dataset)
                    print(config)
                    train(config, bit, rand_num)
                    # test(config, bit)
