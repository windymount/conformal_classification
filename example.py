import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt 
import time
from tqdm import tqdm 
from utils import * 
from conformal import ConformalModel
import torch.backends.cudnn as cudnn
import random

parser = argparse.ArgumentParser(description='Conformalize Torchvision Model on Imagenet')
parser.add_argument('data', metavar='IMAGENETVALDIR', help='path to Imagenet Val')
parser.add_argument('--batch_size', metavar='BSZ', type=int, help='batch size', default=128)
parser.add_argument('--arch', type=str, default="ResNet50")
parser.add_argument('--val_batch_size', metavar='VBSZ', type=int, help='batch size', default=128)
parser.add_argument('--num_workers', metavar='NW', help='number of workers', default=0)
parser.add_argument('--num_calib', metavar='NCALIB', type=int, help='number of calibration points', default=10000)
parser.add_argument('--seed', metavar='SEED', help='random seed', default=0)
parser.add_argument('--eps', type=float, default=2/255)

if __name__ == "__main__":
    args = parser.parse_args()
    ### Fix randomness 
    np.random.seed(seed=args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    # Transform as in https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L92 
    transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std= [0.229, 0.224, 0.225])
                ])

    # Get the conformal calibration dataset
    imagenet_calib_data, imagenet_val_data = torch.utils.data.random_split(torchvision.datasets.ImageFolder(args.data, transform), [args.num_calib,50000-args.num_calib])
    # Initialize loaders 
    val_data_subset, _ = torch.utils.data.random_split(imagenet_val_data, [5000, len(imagenet_val_data) - 5000])
    calib_loader = torch.utils.data.DataLoader(imagenet_calib_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data_subset, batch_size=args.val_batch_size, shuffle=True, pin_memory=True)

    cudnn.benchmark = True

    # Get the model 
    model = get_model(args.arch)
    model = torch.nn.DataParallel(model) 
    model.eval()

    def adv_collate(batch):
        img, target = zip(*batch)
        img = torch.stack(img)
        target = torch.Tensor(target).cuda().type(torch.long)
        adv_img = pgd_attack(model, img, target, torch.device('cuda'), eps=args.eps)
        return adv_img, target

    adv_loader = torch.utils.data.DataLoader(val_data_subset, batch_size=args.val_batch_size, shuffle=True, collate_fn=adv_collate)
    adv_calib_loader = torch.utils.data.DataLoader(imagenet_calib_data, batch_size=args.val_batch_size, shuffle=True, collate_fn=adv_collate)

    # Sanity check
    def evaluate(model, dataloader):
        num_imgs, correct = 0, 0
        for img, label in tqdm(dataloader):
            img, label = img.cuda(), label.cuda()
            with torch.no_grad():
                prediction = torch.argmax(model(img), dim=1)
            correct += torch.sum(prediction == label)
            num_imgs += label.shape[0]
        print("Avg acc: {:.3f}".format(correct/num_imgs))


    # optimize for 'size' or 'adaptiveness'
    lamda_criterion = 'size'
    # allow sets of size zero
    allow_zero_sets = False 
    # use the randomized version of conformal
    randomized = True 

    # Conformalize model
    cmodel = ConformalModel(model, adv_calib_loader, alpha=0.1, lamda_criterion='size')
    print("Model calibrated and conformalized! Now evaluate over remaining data.")
    # validate(val_loader, cmodel, print_bool=True)
    validate(adv_loader, cmodel, print_bool=True)
    print("Complete!")
