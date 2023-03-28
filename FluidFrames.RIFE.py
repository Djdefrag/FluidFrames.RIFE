import ctypes
import itertools
import multiprocessing
import os
import os.path
import platform
import shutil
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkFont
import webbrowser
from multiprocessing.pool import ThreadPool
from timeit import default_timer as timer
from tkinter import PhotoImage, ttk

import cv2
import numpy as np
import tkinterDnD
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml
from moviepy.editor import VideoFileClip
from moviepy.video.io import ImageSequenceClip
from PIL import Image
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from win32mica import MICAMODE, ApplyMica

import sv_ttk

version  = "1.13"

# NEW
# - Added the ability to create slowmotion videos, selectable from the 'AI generation' widget

# GUI
# - Updated info widget descriptions

# BUGFIX / IMPROVEMENTS
# - The app will save thr generated video different tags (x2, x4, x2-slowmotion, x4-slowmotion) according to the user's choice
# - Setted .log file permissions to 777 (maximum permissions), this should solve the problem of reading and writing this file
# - Setted temp folder permissions to 777 (maximum permissions), this should solve the problem of reading and writing in this folder
# - General bugfix and improvements
# - Updated dependencies

global app_name
app_name     = "FluidFrames"
second_title = "RIFE"

models_array             = [ 'RIFE_HDv3' ]
AI_model                 = models_array[0]
generation_factors_array = ['x2', 'x4', 'x2-slowmotion', 'x4-slowmotion']
generation_factor        = 2
slowmotion               = False

image_path            = "none"
device                = 0
input_video_path      = ""
target_file_extension = ".png"
file_extension_list   = [ '.png', '.jpg', '.jp2', '.bmp', '.tiff' ]
half_precision        = False
single_file           = False
multiple_files        = False
video_files           = False
video_frames_list     = []
cpu_number            = 4
windows_subversion    = int(platform.version().split('.')[2])
resize_algorithm      = cv2.INTER_AREA
compatible_gpus       = torch_directml.device_count()

device_list_names     = []
device_list           = []

class Gpu:
    def __init__(self, name, index):
        self.name = name
        self.index = index

for index in range(compatible_gpus): 
    gpu = Gpu(name = torch_directml.device_name(index), index = index)
    device_list.append(gpu)
    device_list_names.append(gpu.name)

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
if sys.stdout is None: sys.stdout = open(os.devnull, "w")
if sys.stderr is None: sys.stderr = open(os.devnull, "w")

githubme = "https://github.com/Djdefrag/FluidFrames.RIFE"
itchme   = "https://jangystudio.itch.io/fluidframesrife"

default_font          = 'Segoe UI'
background_color      = "#181818"
text_color            = "#DCDCDC"
selected_button_color = "#ffbf00"
window_width          = 1300
window_height         = 850
left_bar_width        = 410
left_bar_height       = window_height
drag_drop_width       = window_width - left_bar_width
drag_drop_height      = window_height
show_image_width      = drag_drop_width * 0.8
show_image_height     = drag_drop_width * 0.6
image_text_width      = drag_drop_width * 0.8
support_button_height = 95 
button1_y             = 170
button2_y             = button1_y + 90
button3_y             = button2_y + 90
button4_y             = button3_y + 90
button5_y             = button4_y + 90
button6_y             = button5_y + 90


supported_file_list     = ['.jpg', '.jpeg', '.JPG', '.JPEG',
                            '.png', '.PNG',
                            '.webp', '.WEBP',
                            '.bmp', '.BMP',
                            '.tif', '.tiff', '.TIF', '.TIFF',
                            '.mp4', '.MP4',
                            '.webm', '.WEBM',
                            '.mkv', '.MKV',
                            '.flv', '.FLV',
                            '.gif', '.GIF',
                            '.m4v', ',M4V',
                            '.avi', '.AVI',
                            '.mov', '.MOV',
                            '.qt', '.3gp', '.mpg', '.mpeg']

supported_video_list    = ['.mp4', '.MP4',
                            '.webm', '.WEBM',
                            '.mkv', '.MKV',
                            '.flv', '.FLV',
                            '.gif', '.GIF',
                            '.m4v', ',M4V',
                            '.avi', '.AVI',
                            '.mov', '.MOV',
                            '.qt',
                            '.3gp', '.mpg', '.mpeg']

not_supported_file_list = ['.txt', '.exe', '.xls', '.xlsx', '.pdf',
                           '.odt', '.html', '.htm', '.doc', '.docx',
                           '.ods', '.ppt', '.pptx', '.aiff', '.aif',
                           '.au', '.bat', '.java', '.class',
                           '.csv', '.cvs', '.dbf', '.dif', '.eps',
                           '.fm3', '.psd', '.psp', '.qxd',
                           '.ra', '.rtf', '.sit', '.tar', '.zip',
                           '.7zip', '.wav', '.mp3', '.rar', '.aac',
                           '.adt', '.adts', '.bin', '.dll', '.dot',
                           '.eml', '.iso', '.jar', '.py',
                           '.m4a', '.msi', '.ini', '.pps', '.potx',
                           '.ppam', '.ppsx', '.pptm', '.pst', '.pub',
                           '.sys', '.tmp', '.xlt', '.avif']

ctypes.windll.shcore.SetProcessDpiAwareness(True)
scaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100
font_scale = round(1/scaleFactor, 1)


# ------------------------ Utils ------------------------


def image_write(path, image_data):
    _, file_extension = os.path.splitext(path)
    r, buff = cv2.imencode(file_extension, image_data)
    buff.tofile(path)

def image_read(image_to_prepare, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(image_to_prepare, dtype=np.uint8), flags)

def remove_file(name_file):
    if os.path.exists(name_file): os.remove(name_file)

def create_temp_dir(name_dir):
    if os.path.exists(name_dir): shutil.rmtree(name_dir)
    if not os.path.exists(name_dir): os.makedirs(name_dir, mode=0o777)

def find_by_relative_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def adapt_image_to_show(image_to_prepare):
    old_image     = image_read(image_to_prepare)
    actual_width  = old_image.shape[1]
    actual_height = old_image.shape[0]

    if actual_width >= actual_height:
        max_val = actual_width
        max_photo_resolution = show_image_width
    else:
        max_val = actual_height
        max_photo_resolution = show_image_height

    if max_val >= max_photo_resolution:
        downscale_factor = max_val/max_photo_resolution
        new_width        = round(old_image.shape[1]/downscale_factor)
        new_height       = round(old_image.shape[0]/downscale_factor)
        resized_image    = cv2.resize(old_image,
                                   (new_width, new_height),
                                   interpolation = cv2.INTER_NEAREST)
        image_write("temp.png", resized_image)
        return "temp.png"
    else:
        new_width        = round(old_image.shape[1])
        new_height       = round(old_image.shape[0])
        resized_image    = cv2.resize(old_image,
                                   (new_width, new_height),
                                   interpolation = cv2.INTER_NEAREST)
        image_write("temp.png", resized_image)
        return "temp.png"

def delete_list_of_files(list_to_delete):
    if len(list_to_delete) > 0:
        for to_delete in list_to_delete:
            if os.path.exists(to_delete):
                os.remove(to_delete)

def write_in_log_file(text_to_insert):
    log_file_name = app_name + ".log"
    with open(log_file_name,'w') as log_file: 
        os.chmod(log_file_name, 0o777)
        log_file.write(text_to_insert) 
    log_file.close()

def read_log_file():
    log_file_name = app_name + ".log"
    with open(log_file_name,'r') as log_file: 
        os.chmod(log_file_name, 0o777)
        step = log_file.readline()
    log_file.close()
    return step


#VIDEO

def extract_frames_from_video(video_path):
    video_frames_list = []
    cap          = cv2.VideoCapture(video_path)
    frame_rate   = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    # extract frames
    video = VideoFileClip(video_path)
    img_sequence = app_name + "_temp" + os.sep + "frame_%01d" + '.jpg'
    video_frames_list = video.write_images_sequence(img_sequence, logger = 'bar', fps = frame_rate)
    
    # extract audio
    try:
        video.audio.write_audiofile(app_name + "_temp" + os.sep + "audio.mp3")
    except Exception as e:
        pass

    return video_frames_list

def get_upscaled_video_filepath(input_video_path, slowmotion, AI_model, generation_factor):
    path_as_list = input_video_path.split("/")
    video_name   = str(path_as_list[-1])
    only_path    = input_video_path.replace(video_name, "")
    for video_type in supported_video_list: video_name = video_name.replace(video_type, "")

    if slowmotion: upscaled_video_path = (only_path + video_name + "_"  + AI_model + "x" + str(generation_factor) + "-slowmotion" + ".mp4")
    else: upscaled_video_path = (only_path + video_name + "_"  + AI_model + "x" + str(generation_factor) + ".mp4")

    return upscaled_video_path

def video_reconstruction_by_frames(input_video_path, 
                                   all_files_list, 
                                   AI_model, 
                                   cpu_number, 
                                   generation_factor,
                                   slowmotion):
    
    cap = cv2.VideoCapture(input_video_path)
    if slowmotion: frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    else: frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) * generation_factor
    cap.release()

    upscaled_video_path = get_upscaled_video_filepath(input_video_path, slowmotion, AI_model, generation_factor)

    audio_file = app_name + "_temp" + os.sep + "audio.mp3"

    clip = ImageSequenceClip.ImageSequenceClip(all_files_list, fps = frame_rate)
    if slowmotion:
        clip.write_videofile(upscaled_video_path,
                            fps     = frame_rate,
                            threads = cpu_number)  
    else:
        if os.path.exists(audio_file):
            clip.write_videofile(upscaled_video_path,
                                fps     = frame_rate,
                                audio   = audio_file,
                                threads = cpu_number)
        else:
            clip.write_videofile(upscaled_video_path,
                                fps     = frame_rate,
                                threads = cpu_number)   

def resize_frame(image_path, new_width, new_height, target_file_extension):
    new_image_path = image_path.replace('.jpg', "" + target_file_extension)
    
    old_image = image_read(image_path.strip(), cv2.IMREAD_UNCHANGED)

    resized_image = cv2.resize(old_image, (new_width, new_height), 
                                interpolation = resize_algorithm)    
    image_write(new_image_path, resized_image)

def resize_frame_list(image_list, resize_factor, target_file_extension, cpu_number):
    downscaled_images = []

    old_image = Image.open(image_list[1])
    new_width, new_height = old_image.size
    new_width = int(new_width * resize_factor)
    new_height = int(new_height * resize_factor)
    
    with ThreadPool(cpu_number) as pool:
        pool.starmap(resize_frame, zip(image_list, 
                                    itertools.repeat(new_width), 
                                    itertools.repeat(new_height), 
                                    itertools.repeat(target_file_extension)))

    for image in image_list:
        resized_image_path = image.replace('.jpg', "" + target_file_extension)
        downscaled_images.append(resized_image_path)

    return downscaled_images


# ----------------------- /Utils ------------------------


# ------------------ AI ------------------


backwarp_tenGrid = {}

def warp(tenInput, tenFlow, backend):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=backend).view(
                        1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=backend).view(
                        1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
                        [tenHorizontal, tenVertical], 1).to(backend, non_blocking = True)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', 
                                            padding_mode='border', align_corners=True)

class EPE(nn.Module):
    def __init__(self):
        super(EPE, self).__init__()

    def forward(self, flow, gt, loss_mask):
        loss_map = (flow - gt.detach()) ** 2
        loss_map = (loss_map.sum(1, True) + 1e-6) ** 0.5
        return (loss_map * loss_mask)

class Ternary(nn.Module):
    def __init__(self):
        super(Ternary, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape(
            (patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().to(device)

    def transform(self, img):
        patches = F.conv2d(img, self.w, padding=3, bias=None)
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf**2)
        return transf_norm

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        return dist_norm

    def valid_mask(self, t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, img0, img1):
        img0 = self.transform(self.rgb2gray(img0))
        img1 = self.transform(self.rgb2gray(img1))
        return self.hamming(img0, img1) * self.valid_mask(img0, 1)

class SOBEL(nn.Module):
    def __init__(self, backend):
        super(SOBEL, self).__init__()
        self.kernelX = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]).float()
        self.kernelY = self.kernelX.clone().T
        self.kernelX = self.kernelX.unsqueeze(0).unsqueeze(0).to(backend, non_blocking = True)
        self.kernelY = self.kernelY.unsqueeze(0).unsqueeze(0).to(backend, non_blocking = True)

    def forward(self, pred, gt):
        N, C, H, W = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
        img_stack = torch.cat(
            [pred.reshape(N*C, 1, H, W), gt.reshape(N*C, 1, H, W)], 0)
        sobel_stack_x = F.conv2d(img_stack, self.kernelX, padding=1)
        sobel_stack_y = F.conv2d(img_stack, self.kernelY, padding=1)
        pred_X, gt_X = sobel_stack_x[:N*C], sobel_stack_x[N*C:]
        pred_Y, gt_Y = sobel_stack_y[:N*C], sobel_stack_y[N*C:]

        L1X, L1Y = torch.abs(pred_X-gt_X), torch.abs(pred_Y-gt_Y)
        loss = (L1X+L1Y)
        return loss

class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False
            
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),        
        nn.PReLU(out_planes)
    )

def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(out_planes)
    )

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock0 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.convblock1 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.convblock2 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.convblock3 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(c, c//2, 4, 2, 1),
            nn.PReLU(c//2),
            nn.ConvTranspose2d(c//2, 4, 4, 2, 1),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(c, c//2, 4, 2, 1),
            nn.PReLU(c//2),
            nn.ConvTranspose2d(c//2, 1, 4, 2, 1),
        )

    def forward(self, x, flow, scale=1):
        x = F.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        flow = F.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 1. / scale
        feat = self.conv0(torch.cat((x, flow), 1))
        feat = self.convblock0(feat) + feat
        feat = self.convblock1(feat) + feat
        feat = self.convblock2(feat) + feat
        feat = self.convblock3(feat) + feat        
        flow = self.conv1(feat)
        mask = self.conv2(feat)
        flow = F.interpolate(flow, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) * scale
        mask = F.interpolate(mask, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        return flow, mask
        
class IFNet(nn.Module):
    def __init__(self, backend):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7+4, c=90)
        self.block1 = IFBlock(7+4, c=90)
        self.block2 = IFBlock(7+4, c=90)
        self.block_tea = IFBlock(10+4, c=90)

        self.backend = backend

    def forward(self, x, scale_list=[4, 2, 1], training=False):
        if training == False:
            channel = x.shape[1] // 2
            img0 = x[:, :channel]
            img1 = x[:, channel:]
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = (x[:, :4]).detach() * 0
        mask = (x[:, :1]).detach() * 0
        loss_cons = 0
        block = [self.block0, self.block1, self.block2]
        for i in range(3):
            f0, m0 = block[i](torch.cat((warped_img0[:, :3], warped_img1[:, :3], mask), 1), flow, scale=scale_list[i])
            f1, m1 = block[i](torch.cat((warped_img1[:, :3], warped_img0[:, :3], -mask), 1), torch.cat((flow[:, 2:4], flow[:, :2]), 1), scale=scale_list[i])
            flow = flow + (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
            mask = mask + (m0 + (-m1)) / 2
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2], self.backend)
            warped_img1 = warp(img1, flow[:, 2:4], self.backend)
            merged.append((warped_img0, warped_img1))
        '''
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, 1:4] * 2 - 1
        '''
        for i in range(3):
            mask_list[i] = torch.sigmoid(mask_list[i])
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            # merged[i] = torch.clamp(merged[i] + res, 0, 1)        
        return flow_list, mask_list[2], merged

class RIFEv3:
    def __init__(self, backend, local_rank=-1):
        self.flownet = IFNet(backend)
        self.device(backend)
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)
        self.epe = EPE()
        self.sobel = SOBEL(backend)
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def eval(self): self.flownet.eval()

    def device(self, backend): self.flownet.to(backend, non_blocking = True)

    def inference(self, img0, img1, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [4/scale, 2/scale, 1/scale]
        _ , _ , merged = self.flownet(imgs, scale_list)
        return merged[2]
    

# ------------------ /AI ------------------


# ----------------------- Core ------------------------


def thread_check_steps_for_videos( not_used_var, not_used_var2 ):
    time.sleep(3)
    try:
        while True:
            step = read_log_file()
            if "completed" in step or "Error" in step or "Stopped" in step:
                info_string.set(step)
                stop = 1 + "x"
            info_string.set(step)
            time.sleep(2)
    except:
        place_start_button()


def prepare_model(AI_model, device, half_precision):
    def convert(param):
        return {
            k.replace("module.", ""): v
            for k, v in param.items()
            if "module." in k
        }

    backend = torch.device(torch_directml.device(device))
    model_path = find_by_relative_path("AI" + os.sep + "RIFE_HDv3.pkl")
    model = RIFEv3(backend)

    model.flownet.load_state_dict(convert(torch.load(model_path, 
                                                     map_location ='cpu'))) # maibe to remove?    
    model.eval()
        
    if half_precision: model.flownet = model.flownet.half()
    model.flownet.to(backend, non_blocking = True)

    return model

def adapt_images(img_1, 
                 img_2, 
                 backend, 
                 half_precision):
    img_1 = image_read(img_1, cv2.IMREAD_UNCHANGED)
    img_2 = image_read(img_2, cv2.IMREAD_UNCHANGED)

    img_1 = (torch.tensor(img_1.transpose(2, 0, 1)).to(backend, non_blocking = True) / 255.).unsqueeze(0)
    img_2 = (torch.tensor(img_2.transpose(2, 0, 1)).to(backend, non_blocking = True) / 255.).unsqueeze(0)

    if half_precision:
        img_1 = img_1.half()
        img_2 = img_2.half()

    _ , _ , h, w = img_1.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)

    img_1 = F.pad(img_1, padding)
    img_2 = F.pad(img_2, padding)

    return img_1, img_2, h, w

def generate_middle_image(img_1, 
                          img_2, 
                          all_files_list,
                          model, 
                          target_file_extension, 
                          device, 
                          half_precision,
                          generation_factor):

    backend = torch.device(torch_directml.device(device))
    frames_to_generate = generation_factor - 1
    img_base_name = img_1.replace('.png', '').replace('.jpg','')

    with torch.no_grad():
        first_img, last_img, h, w = adapt_images(img_1, img_2, backend, half_precision)

        if frames_to_generate == 1: #x2
            mid_image = model.inference(first_img, last_img)

            created_image_name = img_base_name + '_middle' + target_file_extension
            created_image = (mid_image[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            image_write(created_image_name, created_image)

            all_files_list.append(img_1)
            all_files_list.append(created_image_name)
            all_files_list.append(img_2)

        elif frames_to_generate == 3: #x4
            mid_image             = model.inference(first_img, last_img)
            mid_image_after_first = model.inference(first_img, mid_image)
            mid_image_prelast     = model.inference(mid_image, last_img)
            
            middle_image_name = img_base_name + '_middle' + target_file_extension
            created_image_middle = (mid_image[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            image_write(middle_image_name, created_image_middle)

            afterfirst_image_name = img_base_name + '_afterfirst' + target_file_extension
            created_image_afterfirst = (mid_image_after_first[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            image_write(afterfirst_image_name, created_image_afterfirst)

            prelast_image_name = img_base_name + '_prelast' + target_file_extension
            created_image_prelast = (mid_image_prelast[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            image_write(prelast_image_name, created_image_prelast)

            all_files_list.append(img_1)
            all_files_list.append(afterfirst_image_name)
            all_files_list.append(middle_image_name)
            all_files_list.append(prelast_image_name)
            all_files_list.append(img_2)


    return all_files_list


def process_generate_video_frames(input_video_path, 
                                  AI_model, 
                                  resize_factor, 
                                  device,
                                  generation_factor, 
                                  target_file_extension, 
                                  cpu_number,
                                  half_precision,
                                  slowmotion):
    try:
        start = timer()

        create_temp_dir(app_name + "_temp")

        write_in_log_file('...')
      
        write_in_log_file('Extracting video frames...')
        image_list = extract_frames_from_video(input_video_path)
        print(' > Extracted: ' + str(len(image_list)) + ' frames')
        
        if resize_factor != 1:
            write_in_log_file('Resizing video frames...')
            image_list  = resize_frame_list(image_list, 
                                            resize_factor, 
                                            target_file_extension, 
                                            cpu_number)
            print(' > Resized: ' + str(len(image_list)) + ' frames')


        write_in_log_file('Starting...')
        how_many_images  = len(image_list)
        done_images      = 0
        all_files_list   = []

        model = prepare_model(AI_model, device, half_precision)

        for index in range(how_many_images):
            try:
                all_files_list = generate_middle_image(image_list[index], 
                                                        image_list[index + 1], 
                                                        all_files_list,
                                                        model, 
                                                        target_file_extension, 
                                                        device, 
                                                        half_precision,
                                                        generation_factor)
                done_images += 1
                write_in_log_file("Generating frame " + str(done_images) + "/" + str(how_many_images))
            except Exception as e: 
                print(str(e))
                pass

        write_in_log_file("Processing video...")
        all_files_list = list(dict.fromkeys(all_files_list))
        all_files_list.append(all_files_list[-1])

        video_reconstruction_by_frames(input_video_path, 
                                       all_files_list, 
                                       AI_model, 
                                       cpu_number, 
                                       generation_factor,
                                       slowmotion)

        write_in_log_file("Video completed [" + str(round(timer() - start)) + " sec.]")

        create_temp_dir(app_name + "_temp")

    except Exception as e:
        write_in_log_file('Error while upscaling' + '\n\n' + str(e)) 
        import tkinter as tk
        tk.messagebox.showerror(title   = 'Error', 
                                message = 'Process failed caused by:\n\n' +
                                            str(e) + '\n\n' +
                                            'Please report the error on Github.com or Itch.io.' +
                                            '\n\nThank you :)')


# ----------------------- /Core ------------------------

# ---------------------- GUI related ----------------------

def opengithub(): webbrowser.open(githubme, new=1)

def openitch(): webbrowser.open(itchme, new=1)

def open_info_generation_factor():
    info = """This widget allows you to choose between different generation factors: \n
- x2 | doubles video framerate | 30fps => 60fps
- x4 | quadruples video framerate | 30fps => 120fps
- x2-slowmotion | slowmotion effect by a factor of 2 | no audio
- x4-slowmotion | slowmotion effect by a factor of 4 | no audio""" 
    
    info_window = tk.Tk()
    info_window.withdraw()
    tk.messagebox.showinfo(title = 'AI generation', message = info )
    info_window.destroy()
    
def open_info_backend():
    info = """This widget allows you to choose the gpu 
on which to run your chosen AI. \n 
Keep in mind that the more powerful your gpu is, 
the faster the upscale will be. \n
If the list is empty it means the app couldn't find 
a compatible gpu, try updating your video card driver :)"""

    info_window = tk.Tk()
    info_window.withdraw()
    tk.messagebox.showinfo(title = 'AI device', message = info)
    info_window.destroy()

def open_info_file_extension():
    info = """This widget allows you to choose the extension of the file generated by AI.\n
- png | very good quality | supports transparent images
- jpg | good quality | very fast
- jpg2 (jpg2000) | very good quality | not very popular
- bmp | highest quality | slow
- tiff | highest quality | very slow"""

    info_window = tk.Tk()
    info_window.withdraw()
    tk.messagebox.showinfo(title = 'AI output extension', message = info)
    info_window.destroy()

def open_info_resize():
    info = """This widget allows you to choose the percentage of the resolution input to the AI.\n
For example for a 100x100px image:
- Input resolution 50% => input to AI 50x50px
- Input resolution 100% => input to AI 100x100px
- Input resolution 200% => input to AI 200x200px """

    info_window = tk.Tk()
    info_window.withdraw()
    tk.messagebox.showinfo(title = 'Input resolution %', message = info)
    info_window.destroy()

def open_info_cpu():
    info = """This widget allows you to choose how much cpu to devote to the app.\n
Where possible the app will use the number of processors you select, for example:
- Extracting frames from videos
- Resizing frames from videos
- Recostructing final video"""

    info_window = tk.Tk()
    info_window.withdraw()
    tk.messagebox.showinfo(title   = 'Cpu number', message = info)
    info_window.destroy() 



def user_input_checks():
    global resize_factor
    global cpu_number

    is_ready = True

    if compatible_gpus == 0:
        tk.messagebox.showerror(title   = 'Error', 
                                message = 'Sorry, your gpu is not compatible with QualityScaler :(')
        is_ready = False

    # resize factor
    try: resize_factor = int(float(str(selected_resize_factor.get())))
    except:
        info_string.set("Resize % must be a numeric value")
        is_ready = False

    #if resize_factor > 0 and resize_factor <= 100: resize_factor = resize_factor/100
    if resize_factor > 0: resize_factor = resize_factor/100
    else:
        info_string.set("Resize % must be a value > 0")
        is_ready = False
    
    # cpu number
    try: cpu_number = int(float(str(selected_cpu_number.get())))
    except:
        info_string.set("Cpu number must be a numeric value")
        is_ready = False 

    if cpu_number <= 0:         
        info_string.set("Cpu number value must be > 0")
        is_ready = False
    elif cpu_number == 1: cpu_number = 1
    else: cpu_number = int(cpu_number)

    return is_ready

def start_button_command():
    global image_path
    global multiple_files
    global process_fluid_frames
    global thread_wait
    global video_frames_list
    global video_files
    global input_video_path
    global device
    global target_file_extension
    global cpu_number
    global half_precision
    global generation_factor
    global slowmotion

    remove_file(app_name + ".log")

    info_string.set("Loading...")
    write_in_log_file("Loading...")

    is_ready = user_input_checks()

    if is_ready:
        if video_files:
            place_stop_button()

            process_fluid_frames = multiprocessing.Process(target = process_generate_video_frames,
                                                    args   = (input_video_path, 
                                                                AI_model, 
                                                                resize_factor, 
                                                                device,
                                                                generation_factor,
                                                                target_file_extension,
                                                                cpu_number,
                                                                half_precision, 
                                                                slowmotion))
            process_fluid_frames.start()

            thread_wait = threading.Thread(target = thread_check_steps_for_videos,
                                            args   = (1, 2), 
                                            daemon = True)
            thread_wait.start()

        elif multiple_files or single_file:
            info_string.set("Only video supported!")

        elif "none" in image_path:
            info_string.set("No file selected")
  
def stop_button_command():
    global process_fluid_frames
    process_fluid_frames.terminate()
    process_fluid_frames.join()
    
    # this will stop thread that check upscaling steps
    write_in_log_file("Stopped") 

def drop_event_to_image_list(event):
    image_list = str(event.data).replace("{", "").replace("}", "")

    for file_type in supported_file_list: image_list = image_list.replace(file_type, file_type+"\n")

    image_list = image_list.split("\n")
    image_list.pop() 

    return image_list

def file_drop_event(event):
    global image_path
    global multiple_files
    global video_files
    global single_file
    global input_video_path

    supported_file_dropped_number, not_supported_file_dropped_number, supported_video_dropped_number = count_files_dropped(event)
    all_supported, single_file, multiple_files, video_files, more_than_one_video = check_compatibility(supported_file_dropped_number, 
                                                                                                        not_supported_file_dropped_number, 
                                                                                                        supported_video_dropped_number)

    if video_files:
        # video section
        if not all_supported:
            info_string.set("Not supported video")
            return
        elif all_supported:
            if multiple_files:
                info_string.set("Only one video supported")
                return
            elif not multiple_files:
                if not more_than_one_video:
                    input_video_path = str(event.data).replace("{", "").replace("}", "")
                    
                    show_video_in_GUI(input_video_path)

                    # reset variable
                    image_path = "none"

                elif more_than_one_video:
                    info_string.set("Only one video supported")
                    return
    else:
        # image section
        info_string.set("Only video supported")
        


def check_compatibility(supported_file_dropped_number, 
                        not_supported_file_dropped_number, 
                        supported_video_dropped_number):
    all_supported  = True
    single_file    = False
    multiple_files = False
    video_files    = False
    more_than_one_video = False

    if not_supported_file_dropped_number > 0:
        all_supported = False

    if supported_file_dropped_number + not_supported_file_dropped_number == 1:
        single_file = True
    elif supported_file_dropped_number + not_supported_file_dropped_number > 1:
        multiple_files = True

    if supported_video_dropped_number == 1:
        video_files = True
        more_than_one_video = False
    elif supported_video_dropped_number > 1:
        video_files = True
        more_than_one_video = True

    return all_supported, single_file, multiple_files, video_files, more_than_one_video

def count_files_dropped(event):
    supported_file_dropped_number = 0
    not_supported_file_dropped_number = 0
    supported_video_dropped_number = 0

    # count compatible images files
    for file_type in supported_file_list:
        supported_file_dropped_number = supported_file_dropped_number + \
            str(event.data).count(file_type)

    # count compatible video files
    for file_type in supported_video_list:
        supported_video_dropped_number = supported_video_dropped_number + \
            str(event.data).count(file_type)

    # count not supported files
    for file_type in not_supported_file_list:
        not_supported_file_dropped_number = not_supported_file_dropped_number + \
            str(event.data).count(file_type)

    return supported_file_dropped_number, not_supported_file_dropped_number, supported_video_dropped_number

def clear_input_variables():
    global image_path
    global video_frames_list
    global single_file
    global multiple_files
    global video_files

    # reset variable
    image_path        = "none"
    video_frames_list = []
    single_file       = False
    multiple_files    = False
    video_files       = False

def clear_app_background():
    drag_drop = ttk.Label(root,
                          ondrop = file_drop_event,
                          relief = "flat",
                          background = background_color,
                          foreground = text_color)
    drag_drop.place(x = left_bar_width + 50, y=0,
                    width = drag_drop_width, height = drag_drop_height)

def show_video_in_GUI(video_path):
    clear_app_background()
    
    fist_frame   = "temp.jpg"
    cap          = cv2.VideoCapture(video_path)
    width        = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate   = cap.get(cv2.CAP_PROP_FPS)
    duration     = num_frames/frame_rate
    minutes      = int(duration/60)
    seconds      = duration % 60
    path_as_list = video_path.split("/")
    video_name   = str(path_as_list[-1])
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False: break
        image_write(fist_frame, frame)
        break
    cap.release()

    resized_image_to_show = adapt_image_to_show(fist_frame)

    global image
    image = tk.PhotoImage(file = resized_image_to_show)
    resized_image_to_show_width = round(image_read(resized_image_to_show).shape[1])
    resized_image_to_show_height = round(image_read(resized_image_to_show).shape[0])
    image_x_center = 30 + left_bar_width + drag_drop_width/2 - resized_image_to_show_width/2
    image_y_center = drag_drop_height/2 - resized_image_to_show_height/2

    image_container = ttk.Notebook(root)
    image_container.place(x = image_x_center - 20, 
                            y = image_y_center - 20, 
                            width  = resized_image_to_show_width + 40,
                            height = resized_image_to_show_height + 40)

    image_ = ttk.Label(root,
                        text    = "",
                        image   = image,
                        ondrop  = file_drop_event,
                        anchor  = "center",
                        relief  = "flat",
                        justify = "center",
                        background = background_color,
                        foreground = "#202020")
    image_.place(x = image_x_center,
                y = image_y_center,
                width  = resized_image_to_show_width,
                height = resized_image_to_show_height)

    image_info_label = ttk.Label(root,
                                  font       = bold11,
                                  text       = ( video_name + "\n" + "[" + str(width) + "x" + str(height) + "]" + " | " + str(minutes) + 'm:' + str(round(seconds)) + "s | " + str(num_frames) + "frames | " + str(round(frame_rate)) + "fps" ),
                                  relief     = "flat",
                                  justify    = "center",
                                  background = background_color,
                                  foreground = "#D3D3D3",
                                  anchor     = "center")

    image_info_label.place(x = 30 + left_bar_width + drag_drop_width/2 - image_text_width/2,
                            y = drag_drop_height - 85,
                            width  = image_text_width,
                            height = 40)
    clean_button = ttk.Button(root, 
                            text     = '  CLEAN',
                            image    = clear_icon,
                            compound = 'left',
                            style    = 'Bold.TButton')

    clean_button.place(x = image_x_center - 20,
                       y = image_y_center - 75,
                       width  = 175,
                       height = 40)
    clean_button["command"] = lambda: place_drag_drop_widget()          

    os.remove(fist_frame)
              
def show_list_images_in_GUI(image_list):
    clear_app_background()

    clean_button = ttk.Button(root, 
                            text     = '  CLEAN',
                            image    = clear_icon,
                            compound = 'left',
                            style    = 'Bold.TButton')

    clean_button.place(x = left_bar_width + drag_drop_width/2 - 175/2,
                       y = 125,
                       width  = 175,
                       height = 40)
    clean_button["command"] = lambda: place_drag_drop_widget()

    final_string = "\n"
    counter_img = 0
    for elem in image_list:
        counter_img += 1
        if counter_img <= 8:
            img     = image_read(elem.strip())
            width   = round(img.shape[1])
            height  = round(img.shape[0])
            img_name = str(elem.split("/")[-1])

            final_string += (str(counter_img) + ".  " + img_name + " | [" + str(width) + "x" + str(height) + "]" + "\n\n")
        else:
            final_string += "and others... \n"
            break

    list_height = 420
    list_width  = 750

    images_list_label = ttk.Label(root,
                            text    = final_string,
                            ondrop  = file_drop_event,
                            font    = bold12,
                            anchor  = "n",
                            relief  = "flat",
                            justify = "left",
                            background = background_color,
                            foreground = "#D3D3D3",
                            wraplength = list_width)

    images_list_label.place(x = left_bar_width + drag_drop_width/2 - list_width/2,
                               y = drag_drop_height/2 - list_height/2 -25,
                               width  = list_width,
                               height = list_height)

    images_counter = ttk.Entry(root, 
                                foreground = text_color,
                                ondrop     = file_drop_event,
                                font       = bold12, 
                                justify    = 'center')
    images_counter.insert(0, str(len(image_list)) + ' images')
    images_counter.configure(state='disabled')
    images_counter.place(x = left_bar_width + drag_drop_width/2 - 175/2,
                        y  = drag_drop_height/2 + 225,
                        width  = 200,
                        height = 42)

def show_image_in_GUI(original_image):
    clear_app_background()
    original_image = original_image.replace('{', '').replace('}', '')
    resized_image_to_show = adapt_image_to_show(original_image)

    global image
    image = tk.PhotoImage(file = resized_image_to_show)
    resized_image_to_show_width = round(image_read(resized_image_to_show).shape[1])
    resized_image_to_show_height = round(image_read(resized_image_to_show).shape[0])
    image_x_center = 30 + left_bar_width + drag_drop_width/2 - resized_image_to_show_width/2
    image_y_center = drag_drop_height/2 - resized_image_to_show_height/2

    image_container = ttk.Notebook(root)
    image_container.place(x = image_x_center - 20, 
                            y = image_y_center - 20, 
                            width  = resized_image_to_show_width + 40,
                            height = resized_image_to_show_height + 40)

    image_ = ttk.Label(root,
                        text    = "",
                        image   = image,
                        ondrop  = file_drop_event,
                        anchor  = "center",
                        relief  = "flat",
                        justify = "center",
                        background = background_color,
                        foreground = "#202020")
    image_.place(x = image_x_center,
                      y = image_y_center,
                      width  = resized_image_to_show_width,
                      height = resized_image_to_show_height)

    img_name     = str(original_image.split("/")[-1])
    width        = round(image_read(original_image).shape[1])
    height       = round(image_read(original_image).shape[0])

    image_info_label = ttk.Label(root,
                                  font       = bold11,
                                  text       = (img_name + " | [" + str(width) + "x" + str(height) + "]"),
                                  relief     = "flat",
                                  justify    = "center",
                                  background = background_color,
                                  foreground = "#D3D3D3",
                                  anchor     = "center")

    image_info_label.place(x = 30 + left_bar_width + drag_drop_width/2 - image_text_width/2,
                            y = drag_drop_height - 70,
                            width  = image_text_width,
                            height = 40)

    clean_button = ttk.Button(root, 
                            text     = '  CLEAN',
                            image    = clear_icon,
                            compound = 'left',
                            style    = 'Bold.TButton')

    clean_button.place(x = image_x_center - 20,
                       y = image_y_center - 75,
                       width  = 175,
                       height = 40)
    clean_button["command"] = lambda: place_drag_drop_widget()

def place_drag_drop_widget():
    clear_input_variables()

    clear_app_background()

    text_drop = (" DROP VIDEO HERE \n\n"
                + " ⥥ \n\n"
                + " VIDEO   - mp4 webm mkv flv gif avi mov mpg qt 3gp \n\n")

    drag_drop = ttk.Notebook(root, ondrop  = file_drop_event)

    x_center = 30 + left_bar_width + drag_drop_width/2 - (drag_drop_width * 0.75)/2
    y_center = drag_drop_height/2 - (drag_drop_height * 0.75)/2

    drag_drop.place(x = x_center, 
                    y = y_center, 
                    width  = drag_drop_width * 0.75, 
                    height = drag_drop_height * 0.75)

    drag_drop_text = ttk.Label(root,
                            text    = text_drop,
                            ondrop  = file_drop_event,
                            font    = bold12,
                            anchor  = "center",
                            relief  = 'flat',
                            justify = "center",
                            foreground = text_color)

    x_center = 30 + left_bar_width + drag_drop_width/2 - (drag_drop_width * 0.5)/2
    y_center = drag_drop_height/2 - (drag_drop_height * 0.5)/2
    
    drag_drop_text.place(x = x_center, 
                         y = y_center, 
                         width  = drag_drop_width * 0.50, 
                         height = drag_drop_height * 0.50)



def combobox_generation_factor_selection(event):
    global generation_factor
    global slowmotion

    selected = str(selected_generation_factor.get())

    if 'slowmotion' in selected:
        slowmotion = True
        generation_factor = int(selected.replace('x', '').replace('-slowmotion', ''))
    else:
        slowmotion = False
        generation_factor = int(selected.replace('x', ''))

    combo_box_generation_factor.set('')
    combo_box_generation_factor.set(selected)

def combobox_backend_selection(event):
    global device

    selected_option = str(selected_backend.get())
    combo_box_backend.set('')
    combo_box_backend.set(selected_option)

    for obj in device_list:
        if obj.name == selected_option:
            device = obj.index

def combobox_extension_selection(event):
    global target_file_extension
    selected = str(selected_file_extension.get()).strip()
    target_file_extension = selected
    combobox_file_extension.set('')
    combobox_file_extension.set(selected)


def place_generation_factor_combobox():
    generation_factor_container = ttk.Notebook(root)
    generation_factor_container.place(x = 45 + left_bar_width/2 - 370/2, 
                                        y = button1_y - 17, 
                                        width  = 370,
                                        height = 75)

    generation_factor_label = ttk.Label(root, 
                                    font       = bold11, 
                                    foreground = text_color, 
                                    justify    = 'left', 
                                    relief     = 'flat', 
                                    text       = " AI generation ")
    generation_factor_label.place(x = 90,
                                y = button1_y - 2,
                                width  = 175,
                                height = 42)

    global combo_box_generation_factor
    combo_box_generation_factor = ttk.Combobox(root, 
                                            textvariable = selected_generation_factor, 
                                            justify      = 'center',
                                            foreground   = text_color,
                                            values       = generation_factors_array,
                                            state        = 'readonly',
                                            takefocus    = False,
                                            font         = bold11)
    combo_box_generation_factor.place(x = 10 + left_bar_width/2, 
                                    y = button1_y, 
                                    width  = 200, 
                                    height = 40)
    combo_box_generation_factor.bind('<<ComboboxSelected>>', combobox_generation_factor_selection)
    combo_box_generation_factor.set(generation_factors_array[0])

    generation_factor_info_button = ttk.Button(root,
                                                padding = '0 0 0 0',
                                                text    = "i",
                                                compound = 'left',
                                                style    = 'Bold.TButton')
    generation_factor_info_button.place(x = 50,
                                    y = button1_y + 6,
                                    width  = 30,
                                    height = 30)
    generation_factor_info_button["command"] = lambda: open_info_generation_factor()

def place_backend_combobox():
    backend_container = ttk.Notebook(root)
    backend_container.place(x = 45 + left_bar_width/2 - 370/2, 
                            y = button2_y - 17, 
                            width  = 370,
                            height = 75)

    backend_label = ttk.Label(root, 
                            font       = bold11, 
                            foreground = text_color, 
                            justify    = 'left', 
                            relief     = 'flat', 
                            text       = " AI device ")
    backend_label.place(x = 90,
                        y = button2_y - 2,
                        width  = 155,
                        height = 42)

    global combo_box_backend
    combo_box_backend = ttk.Combobox(root, 
                            textvariable = selected_backend, 
                            justify      = 'center',
                            foreground   = text_color,
                            values       = device_list_names,
                            state        = 'readonly',
                            takefocus    = False,
                            font         = bold10)
    combo_box_backend.place(x = 10 + left_bar_width/2, 
                            y = button2_y, 
                            width  = 200, 
                            height = 40)
    combo_box_backend.bind('<<ComboboxSelected>>', combobox_backend_selection)
    combo_box_backend.set(device_list_names[0])

    backend_combobox_info_button = ttk.Button(root,
                               padding = '0 0 0 0',
                               text    = "i",
                               compound = 'left',
                               style    = 'Bold.TButton')
    backend_combobox_info_button.place(x = 50,
                                    y = button2_y + 6,
                                    width  = 30,
                                    height = 30)
    backend_combobox_info_button["command"] = lambda: open_info_backend()

def place_file_extension_combobox():
    file_extension_container = ttk.Notebook(root)
    file_extension_container.place(x = 45 + left_bar_width/2 - 370/2, 
                        y = button3_y - 17, 
                        width  = 370,
                        height = 75)

    file_extension_label = ttk.Label(root, 
                        font       = bold11, 
                        foreground = text_color, 
                        justify    = 'left', 
                        relief     = 'flat', 
                        text       = " AI output extension ")
    file_extension_label.place(x = 90,
                            y = button3_y - 2,
                            width  = 155,
                            height = 42)

    global combobox_file_extension
    combobox_file_extension = ttk.Combobox(root, 
                        textvariable = selected_file_extension, 
                        justify      = 'center',
                        foreground   = text_color,
                        values       = file_extension_list,
                        state        = 'readonly',
                        takefocus    = False,
                        font         = bold11)
    combobox_file_extension.place(x = 65 + left_bar_width/2, 
                        y = button3_y, 
                        width  = 145, 
                        height = 40)
    combobox_file_extension.bind('<<ComboboxSelected>>', combobox_extension_selection)
    combobox_file_extension.set(target_file_extension)

    file_extension_combobox_info_button = ttk.Button(root,
                               padding = '0 0 0 0',
                               text    = "i",
                               compound = 'left',
                               style    = 'Bold.TButton')
    file_extension_combobox_info_button.place(x = 50,
                                    y = button3_y + 6,
                                    width  = 30,
                                    height = 30)
    file_extension_combobox_info_button["command"] = lambda: open_info_file_extension()

def place_resize_factor_spinbox():
    resize_factor_container = ttk.Notebook(root)
    resize_factor_container.place(x = 45 + left_bar_width/2 - 370/2, 
                               y = button4_y - 17, 
                               width  = 370,
                               height = 75)

    global spinbox_resize_factor
    spinbox_resize_factor = ttk.Spinbox(root,  
                                        from_        = 1, 
                                        to           = 100, 
                                        increment    = 1,
                                        textvariable = selected_resize_factor, 
                                        justify      = 'center',
                                        foreground   = text_color,
                                        takefocus    = False,
                                        font         = bold12)
    spinbox_resize_factor.place(x = 65 + left_bar_width/2, 
                                y = button4_y, 
                                width  = 145, 
                                height = 40)
    spinbox_resize_factor.insert(0, '70')

    resize_factor_label = ttk.Label(root, 
                                    font       = bold11, 
                                    foreground = text_color, 
                                    justify    = 'left', 
                                    relief     = 'flat', 
                                    text       = " Input resolution | % ")
    resize_factor_label.place(x = 90,
                            y = button4_y - 2,
                            width  = 155,
                            height = 42)
    
    resize_spinbox_info_button = ttk.Button(root,
                               padding = '0 0 0 0',
                               text    = "i",
                               compound = 'left',
                               style    = 'Bold.TButton')
    resize_spinbox_info_button.place(x = 50,
                                    y = button4_y + 6,
                                    width  = 30,
                                    height = 30)
    resize_spinbox_info_button["command"] = lambda: open_info_resize()

def place_cpu_number_spinbox():
    cpu_number_container = ttk.Notebook(root)
    cpu_number_container.place(x = 45 + left_bar_width/2 - 370/2, 
                        y = button5_y - 17, 
                        width  = 370,
                        height = 75)

    global spinbox_cpus
    spinbox_cpus = ttk.Spinbox(root,  
                                from_     = 1, 
                                to        = 100, 
                                increment = 1,
                                textvariable = selected_cpu_number, 
                                justify      = 'center',
                                foreground   = text_color,
                                takefocus    = False,
                                font         = bold12)
    spinbox_cpus.place(x = 65 + left_bar_width/2, 
                        y = button5_y, 
                        width  = 145, 
                        height = 40)
    spinbox_cpus.insert(0, str(cpu_number))

    cpus_label = ttk.Label(root, 
                            font       = bold11, 
                            foreground = text_color, 
                            justify    = 'left', 
                            relief     = 'flat', 
                            text       = " Cpu number ")
    cpus_label.place(x = 90,
                    y = button5_y - 2,
                    width  = 155,
                    height = 42)
    
    cpu_spinbox_info_button = ttk.Button(root,
                               padding = '0 0 0 0',
                               text    = "i",
                               compound = 'left',
                               style    = 'Bold.TButton')
    cpu_spinbox_info_button.place(x = 50,
                                    y = button5_y + 6,
                                    width  = 30,
                                    height = 30)
    cpu_spinbox_info_button["command"] = lambda: open_info_cpu()




def place_clean_button():
    clean_button = ttk.Button(root, 
                            text     = '  CLEAN',
                            image    = clear_icon,
                            compound = 'left',
                            style    = 'Bold.TButton')

    clean_button.place(x = 45 + left_bar_width + drag_drop_width/2 - 175/2,
                       y = 25,
                       width  = 175,
                       height = 40)
    clean_button["command"] = lambda: place_drag_drop_widget()

def place_app_title():
    Title = ttk.Label(root, 
                      font       = bold20,
                      foreground = "#F08080", 
                      background = background_color,
                      anchor     = 'w', 
                      text       = app_name)
    Title.place(x = 60,
                y = 25,
                width  = 300,
                height = 55)

    Second_title = ttk.Label(root, 
                      font       = bold18,
                      foreground = "#0096FF", 
                      background = background_color,
                      anchor     = 'w', 
                      text       = second_title)
    Second_title.place(x = 195,
                y = 67,
                width  = 150,
                height = 35)

    version_button = ttk.Button(root,
                               image = logo_itch,
                               padding = '0 0 0 0',
                               text    = " " + version,
                               compound = 'left',
                               style    = 'Bold.TButton')
    version_button.place(x = (left_bar_width + 45) - (125 + 30),
                        y = 30,
                        width  = 125,
                        height = 35)
    version_button["command"] = lambda: openitch()

    ft = tkFont.Font(family = default_font)
    Butt_Style = ttk.Style()
    Butt_Style.configure("Bold.TButton", font = ft)

    github_button = ttk.Button(root,
                               image = logo_git,
                               padding = '0 0 0 0',
                               text    = ' Github',
                               compound = 'left',
                               style    = 'Bold.TButton')
    github_button.place(x = (left_bar_width + 45) - (125 + 30),
                        y = 75,
                        width  = 125,
                        height = 35)
    github_button["command"] = lambda: opengithub()

def place_message_box():
    message_label = ttk.Label(root,
                            font       = bold11,
                            textvar    = info_string,
                            relief     = "flat",
                            justify    = "center",
                            background = background_color,
                            foreground = "#ffbf00",
                            anchor     = "center")
    message_label.place(x = 45 + left_bar_width/2 - left_bar_width/2,
                        y = window_height - 120,
                        width  = left_bar_width,
                        height = 30)

def place_start_button():
    button_Style = ttk.Style()
    button_Style.configure("Bold.TButton", font = bold11, foreground = text_color)

    Upscale_button = ttk.Button(root, 
                                text  = ' START',
                                image = play_icon,
                                compound = tk.LEFT,
                                style = "Bold.TButton")

    Upscale_button.place(x      = 45 + left_bar_width/2 - 275/2,  
                         y      = left_bar_height - 80,
                         width  = 280,
                         height = 47)
    Upscale_button["command"] = lambda: start_button_command()

def place_stop_button():
    Upsc_Butt_Style = ttk.Style()
    Upsc_Butt_Style.configure("Bold.TButton", font = bold11)

    Stop_button = ttk.Button(root, 
                                text  = '  STOP',
                                image = stop_icon,
                                compound = tk.LEFT,
                                style    = 'Bold.TButton')

    Stop_button.place(x      = 45 + left_bar_width/2 - 275/2,  
                      y      = left_bar_height - 80,
                      width  = 280,
                      height = 47)

    Stop_button["command"] = lambda: stop_button_command()

def place_background():
    background = ttk.Label(root, background = background_color, relief = 'flat')
    background.place(x = 0, 
                     y = 0, 
                     width  = window_width,
                     height = window_height)


# ---------------------- /GUI related ----------------------


def apply_windows_dark_bar(window_root):
    window_root.update()
    DWMWA_USE_IMMERSIVE_DARK_MODE = 20
    set_window_attribute          = ctypes.windll.dwmapi.DwmSetWindowAttribute
    get_parent                    = ctypes.windll.user32.GetParent
    hwnd                          = get_parent(window_root.winfo_id())
    rendering_policy              = DWMWA_USE_IMMERSIVE_DARK_MODE
    value                         = 2
    value                         = ctypes.c_int(value)
    set_window_attribute(hwnd, rendering_policy, ctypes.byref(value), ctypes.sizeof(value))    

    #Changes the window size
    window_root.geometry(str(window_root.winfo_width()+1) + "x" + str(window_root.winfo_height()+1))
    #Returns to original size
    window_root.geometry(str(window_root.winfo_width()-1) + "x" + str(window_root.winfo_height()-1))

def apply_windows_transparency_effect(window_root):
    window_root.wm_attributes("-transparent", background_color)
    hwnd = ctypes.windll.user32.GetParent(window_root.winfo_id())
    ApplyMica(hwnd, MICAMODE.DARK )


class App:
    def __init__(self, root):
        sv_ttk.use_dark_theme()
        
        root.title('')
        width        = window_width
        height       = window_height
        screenwidth  = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr     = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)
        root.iconphoto(False, PhotoImage(file = find_by_relative_path("Assets"  + os.sep + "logo.png")))

        if windows_subversion >= 22000: apply_windows_transparency_effect(root) # Windows 11
        apply_windows_dark_bar(root)

        place_background()                                  # Background
        place_app_title()                                   # App title
        place_generation_factor_combobox()                  # AI models widget
        place_resize_factor_spinbox()                       
        place_backend_combobox()                            # Backend widget
        place_file_extension_combobox()
        place_cpu_number_spinbox()
        place_message_box()                                 # Message box
        place_start_button()                                # Button
        place_drag_drop_widget()                            # Drag&Drop widget
        
if __name__ == "__main__":
    multiprocessing.freeze_support()

    root        = tkinterDnD.Tk()
    info_string = tk.StringVar()
    selected_generation_factor = tk.StringVar()
    selected_resize_factor     = tk.StringVar()
    selected_backend           = tk.StringVar()
    selected_file_extension    = tk.StringVar()
    selected_cpu_number        = tk.StringVar()

    bold10 = tkFont.Font(family = default_font, size   = round(10 * font_scale), weight = 'bold')
    bold11 = tkFont.Font(family = default_font, size   = round(11 * font_scale), weight = 'bold')
    bold12 = tkFont.Font(family = default_font, size   = round(12 * font_scale), weight = 'bold')
    bold13 = tkFont.Font(family = default_font, size   = round(13 * font_scale), weight = 'bold')
    bold14 = tkFont.Font(family = default_font, size   = round(14 * font_scale), weight = 'bold')
    bold15 = tkFont.Font(family = default_font, size   = round(15 * font_scale), weight = 'bold')
    bold18 = tkFont.Font(family = default_font, size   = round(18 * font_scale), weight = 'bold')    
    bold20 = tkFont.Font(family = default_font, size   = round(20 * font_scale), weight = 'bold')
    bold21 = tkFont.Font(family = default_font, size   = round(21 * font_scale), weight = 'bold')

    global stop_icon
    global clear_icon
    global play_icon
    global logo_itch
    global logo_git
    logo_git      = tk.PhotoImage(file = find_by_relative_path("Assets" + os.sep + "github_logo.png"))
    logo_itch     = tk.PhotoImage(file = find_by_relative_path("Assets" + os.sep + "itch_logo.png"))
    stop_icon     = tk.PhotoImage(file = find_by_relative_path("Assets" + os.sep + "stop_icon.png"))
    play_icon     = tk.PhotoImage(file = find_by_relative_path("Assets" + os.sep + "upscale_icon.png"))
    clear_icon    = tk.PhotoImage(file = find_by_relative_path("Assets" + os.sep + "clear_icon.png"))

    app = App(root)
    root.update()
    root.mainloop()

