import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import cv2

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory


import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models_deptha_distill_vim import DPT_Vim, DepthAnything

from engine_distill_deptha import train_one_epoch

from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'])

    parser.add_argument('--resume', default='/home/ypf/workspace2/code/DMAE/work_dirs/distill_base_model_0516/checkpoint-5.pth',
                        help='resume from checkpoint')
    
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    margin_width = 50
    caption_height = 60
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # DEVICE = 'cpu'
    print('Using device:', DEVICE)

    device = torch.device(DEVICE)

    model = DPT_Vim(encoder='vits', features=64, out_channels=[48,96,192,384])
    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print("Resume checkpoint %s" % args.resume)

    # optimizer = optim_factory.create_optimizer_v2(model_or_params = model, opt = 'adamw', lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    loss_scaler = NativeScaler()
    # misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)
 
    
    total_params = sum(param.numel() for param in model.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    # encoder params
    encoder_params = 0.
    for name, param in model.encoder.named_parameters():
        encoder_params += param.numel()

    decoder_params = 0.
    for name, param in model.depth_head.named_parameters():
        decoder_params += param.numel()

    print('Encoder parameters: {:.2f}M'.format(encoder_params / 1e6))
    print('Decoder parameters: {:.2f}M'.format(decoder_params / 1e6))
    
    transform = transforms.Compose([
        Resize(
            width=224,
            height=224,
            resize_target=False,
            keep_aspect_ratio=False,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    # transform_train = transforms.Compose([
    #     transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     PrepareForNet()]
    #     )
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = os.listdir(args.img_path)
        filenames = [os.path.join(args.img_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    os.makedirs(args.outdir, exist_ok=True)
   

    for filename in tqdm(filenames):
        raw_image = cv2.imread(filename)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        h, w = image.shape[:2]
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
        total_time = 0
        with torch.no_grad():
            # for i in range(110):
                depth = model.infer(image)

        # total_time = time.perf_counter() - start
        # print('Elapsed time: {:.6f}s'.format(total_time))
        # avg_time = total_time / 100
        # print('Average time: {:.6f}s'.format(avg_time))
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        
        depth = depth.cpu().numpy().astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        
        filename = os.path.basename(filename)
        
        if args.pred_only:
            cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_depth.png'), depth)
        else:
            split_region = np.ones((raw_image.shape[0], margin_width, 3), dtype=np.uint8) * 255
            combined_results = cv2.hconcat([raw_image, split_region, depth])
            
            caption_space = np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8) * 255
            captions = ['Raw image', 'Depth Anything']
            segment_width = w + margin_width
            
            for i, caption in enumerate(captions):
                # Calculate text size
                text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

                # Calculate x-coordinate to center the text
                text_x = int((segment_width * i) + (w - text_size[0]) / 2)

                # Add text caption
                cv2.putText(caption_space, caption, (text_x, 40), font, font_scale, (0, 0, 0), font_thickness)
            
            final_result = cv2.vconcat([caption_space, combined_results])
            
            cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_img_depth.png'), final_result)
        