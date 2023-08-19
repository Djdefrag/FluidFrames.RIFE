import multiprocessing
import os.path
import shutil
import sys
import threading
import time
import tkinter as tk
import webbrowser
from timeit import default_timer as timer

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml
from customtkinter import (CTk, 
                           CTkButton, 
                           CTkEntry, 
                           CTkFont, 
                           CTkImage,
                           CTkLabel, 
                           CTkOptionMenu, 
                           CTkScrollableFrame,
                           filedialog, 
                           set_appearance_mode,
                           set_default_color_theme)
from moviepy.editor import VideoFileClip
from moviepy.video.io import ImageSequenceClip
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

app_name     = "FluidFrames"
second_title = "RIFE"
version      = "2.3"

githubme   = "https://github.com/Djdefrag/FluidFrames.RIFE"
itchme     = "https://jangystudio.itch.io/fluidframesrife"
telegramme = "https://linktr.ee/j3ngystudio"

half_precision           = False 
fluidity_options_list    = [
                            'x2', 
                            'x4', 
                            'x2-slowmotion', 
                            'x4-slowmotion'
                            ]

image_extension_list  = [ '.png', '.jpg', '.bmp', '.tiff' ]
video_extension_list  = [ '.mp4', '.avi' ]

device_list_names    = []
device_list          = []
resize_algorithm     = cv2.INTER_AREA 

offset_y_options = 0.1125
row1_y           = 0.705
row2_y           = row1_y + offset_y_options
row3_y           = row2_y + offset_y_options

app_name_color    = "#4169E1"
select_files_widget_color = "#080808"


# Classes and utils -------------------

class Gpu:
    def __init__(self, index, name):
        self.name   = name
        self.index  = index

class ScrollableImagesTextFrame(CTkScrollableFrame):
    def __init__(self, master, command=None, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.label_list  = []
        self.button_list = []
        self.file_list   = []

    def get_selected_file_list(self): 
        return self.file_list

    def add_clean_button(self):
        label = CTkLabel(self, text = "")
        button = CTkButton(self, 
                            font  = bold11,
                            text  = "CLEAN", 
                            fg_color   = "#282828",
                            text_color = "#E0E0E0",
                            image    = clear_icon,
                            compound = "left",
                            width    = 85, 
                            height   = 27,
                            corner_radius = 25)
        button.configure(command=lambda: self.clean_all_items())
        button.grid(row = len(self.button_list), column=1, pady=(0, 10), padx = 5)
        self.label_list.append(label)
        self.button_list.append(button)

    def add_item(self, text_to_show, file_element, image = None):
        label = CTkLabel(self, 
                        text  = text_to_show,
                        font  = bold11,
                        image = image, 
                        #fg_color   = "#282828",
                        text_color = "#E0E0E0",
                        compound = "left", 
                        padx     = 10,
                        pady     = 5,
                        corner_radius = 25,
                        anchor   = "center")
                        
        label.grid(row  = len(self.label_list), column = 0, 
                   pady = (3, 3), padx = (3, 3), sticky = "w")
        self.label_list.append(label)
        self.file_list.append(file_element)    

    def clean_all_items(self):
        self.label_list  = []
        self.button_list = []
        self.file_list   = []
        place_up_background()
        place_loadFile_section()

for index in range(torch_directml.device_count()): 
    gpu_name = torch_directml.device_name(index)

    gpu = Gpu(index = index, name = gpu_name)
    device_list.append(gpu)
    device_list_names.append(gpu.name)

supported_file_extensions = ['.mp4', '.MP4',
                            '.webm', '.WEBM',
                            '.mkv', '.MKV',
                            '.flv', '.FLV',
                            '.gif', '.GIF',
                            '.m4v', ',M4V',
                            '.avi', '.AVI',
                            '.mov', '.MOV',
                            '.qt', '.3gp', '.mpg', '.mpeg']

supported_video_extensions  = ['.mp4', '.MP4',
                                '.webm', '.WEBM',
                                '.mkv', '.MKV',
                                '.flv', '.FLV',
                                '.gif', '.GIF',
                                '.m4v', ',M4V',
                                '.avi', '.AVI',
                                '.mov', '.MOV',
                                '.qt',
                                '.3gp', '.mpg', '.mpeg']

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
if sys.stdout is None: sys.stdout = open(os.devnull, "w")
if sys.stderr is None: sys.stderr = open(os.devnull, "w")



# Utils functions ------------------------

def opengithub(): webbrowser.open(githubme, new=1)

def openitch(): webbrowser.open(itchme, new=1)

def opentelegram(): webbrowser.open(telegramme, new=1)

def image_write(path, image_data):
    cv2.imwrite(path, image_data)

def image_read(image_to_prepare, flags = cv2.IMREAD_UNCHANGED):
    return cv2.imread(image_to_prepare, flags)

def create_temp_dir(name_dir):
    if os.path.exists(name_dir): shutil.rmtree(name_dir)
    if not os.path.exists(name_dir): os.makedirs(name_dir, mode=0o777)

def remove_dir(name_dir):
    if os.path.exists(name_dir): shutil.rmtree(name_dir)

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

def find_by_relative_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def prepare_output_video_filename(video_path, 
                                  fluidification_factor, 
                                  slowmotion, 
                                  resize_factor, 
                                  selected_video_output = ".mp4"):
    
    result_video_path = os.path.splitext(video_path)[0]  # remove extension

    resize_percentage = str(int(resize_factor * 100)) + "%"
    
    if slowmotion: to_append = "_RIFEx" + str(fluidification_factor) + "_" + "slowmo" + "_" + resize_percentage + selected_video_output
    else:          to_append = "_RIFEx" + str(fluidification_factor) + "_" + resize_percentage + selected_video_output

    result_video_path = result_video_path + to_append

    return result_video_path

def delete_list_of_files(list_to_delete):
    if len(list_to_delete) > 0:
        for to_delete in list_to_delete:
            if os.path.exists(to_delete):
                os.remove(to_delete)

def resize_frames(frame_1, frame_2, target_width, target_height):

    frame_1_resized = cv2.resize(frame_1, (target_width, target_height), interpolation = resize_algorithm)    
    frame_2_resized = cv2.resize(frame_2, (target_width, target_height), interpolation = resize_algorithm)    

    return frame_1_resized, frame_2_resized

def remove_file(name_file):
    if os.path.exists(name_file): os.remove(name_file)

def show_error(exception):
    import tkinter as tk
    tk.messagebox.showerror(title   = 'Error', 
                            message = 'Upscale failed caused by:\n\n' +
                                        str(exception) + '\n\n' +
                                        'Please report the error on Github.com or Itch.io.' +
                                        '\n\nThank you :)')

def extract_frames_from_video(video_path):
    video_frames_list = []
    cap          = cv2.VideoCapture(video_path)
    frame_rate   = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # extract frames
    video = VideoFileClip(video_path)
    frame_sequence = app_name + "_temp" + os.sep + "frame_%01d" + '.jpg'
    video_frames_list = video.write_images_sequence(frame_sequence, 
                                                    verbose = False,
                                                    logger  = None, 
                                                    fps     = frame_rate)
    
    # extract audio
    try: video.audio.write_audiofile(app_name + "_temp" + os.sep + "audio.mp3",
                                    verbose = False,
                                    logger  = None)
    except: pass

    return video_frames_list

def video_reconstruction_by_frames(input_video_path, 
                                    all_files_list, 
                                    fluidification_factor,
                                    slowmotion,
                                    resize_factor,
                                    cpu_number,
                                    selected_video_extension):
    
    # Find original video FPS
    cap          = cv2.VideoCapture(input_video_path)
    if slowmotion: frame_rate = cap.get(cv2.CAP_PROP_FPS)
    else: frame_rate = cap.get(cv2.CAP_PROP_FPS) * fluidification_factor
    cap.release()

    # Choose the appropriate codec
    if selected_video_extension == '.mp4':
        extension = '.mp4'
        codec = 'libx264'
    elif selected_video_extension == '.avi':
        extension = '.avi'
        codec = 'png'

    audio_file = app_name + "_temp" + os.sep + "audio.mp3"
    upscaled_video_path = prepare_output_video_filename(input_video_path, 
                                                        fluidification_factor, 
                                                        slowmotion, 
                                                        resize_factor, 
                                                        selected_video_extension)

    clip = ImageSequenceClip.ImageSequenceClip(all_files_list, fps = frame_rate)
    if os.path.exists(audio_file) and slowmotion != True:
        clip.write_videofile(upscaled_video_path,
                            fps     = frame_rate,
                            audio   = audio_file,
                            codec   = codec,
                            verbose = False,
                            logger  = None,
                            threads = cpu_number)
    else:
        clip.write_videofile(upscaled_video_path,
                             fps     = frame_rate,
                             codec   = codec,
                             verbose = False,
                             logger  = None,
                             threads = cpu_number)      



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
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        #self.w = torch.tensor(self.w).float().to(device)

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

def prepare_model(backend, half_precision):
    def convert(param):
        return {
            k.replace("module.", ""): v
            for k, v in param.items()
            if "module." in k
        }

    model_path = find_by_relative_path("AI" + os.sep + "RIFE_HDv3.pkl")
    model = RIFEv3(backend)

    with torch.no_grad():
        pretrained_model = torch.load(model_path, map_location = torch.device('cpu'))
        model.flownet.load_state_dict(convert(pretrained_model))

    model.eval()

    if half_precision: model.flownet = model.flownet.half()

    model.flownet.to(backend, non_blocking=True)

    return model



# Core functions ------------------------

def remove_temp_files():
    remove_dir(app_name + "_temp")
    remove_file(app_name + ".log")

def stop_thread():
    # to stop a thread execution
    stop = 1 + "x"

def stop_fluidify_process():
    global process_fluidify_orchestrator
    process_fluidify_orchestrator.terminate()
    process_fluidify_orchestrator.join()

def check_fluidify_steps():
    time.sleep(3)
    try:
        while True:
            step = read_log_file()
            if "All files completed" in step:
                info_message.set(step)
                stop_fluidify_process()
                remove_temp_files()
                stop_thread()
            elif "Error while fluidifying" in step:
                info_message.set("Error while fluidifying :(")
                remove_temp_files()
                stop_thread()
            elif "Stopped fluidifying" in step:
                info_message.set("Stopped fluidifying")
                stop_fluidify_process()
                remove_temp_files()
                stop_thread()
            else:
                info_message.set(step)
            time.sleep(1)
    except:
        place_fluidify_button()

def update_process_status(actual_process_phase):
    print("> " + actual_process_phase)
    write_in_log_file(actual_process_phase) 

def stop_button_command():
    stop_fluidify_process()
    # this will stop thread that check fluidifying steps
    write_in_log_file("Stopped fluidifying") 

def frames_to_tensors(frame_1, 
                      frame_2, 
                      backend, 
                      half_precision):

    img_1 = (torch.tensor(frame_1.transpose(2, 0, 1)).to(backend, non_blocking = True) / 255.).unsqueeze(0)
    img_2 = (torch.tensor(frame_2.transpose(2, 0, 1)).to(backend, non_blocking = True) / 255.).unsqueeze(0)

    _ , _ , h, w = img_1.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)

    img_1 = F.pad(img_1, padding)
    img_2 = F.pad(img_2, padding)

    if half_precision:
        img_1 = img_1.half()
        img_2 = img_2.half()

    return img_1, img_2, h, w

def tensor_to_frame(result, h, w):
    return (result[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

def AI_generate_frames(frame1, 
                        frame2, 
                        frame_1_name,
                        frame_2_name,
                        frame_base_name,
                        all_files_list,
                        AI_model, 
                        selected_output_file_extension, 
                        backend, 
                        half_precision,
                        fluidification_factor):

    frames_to_generate = fluidification_factor - 1

    with torch.no_grad():
        first_frame_tensor, last_frame_tensor, h, w = frames_to_tensors(frame1, frame2, backend, half_precision)

        if frames_to_generate == 1: 
            # fluidification x2
            middle_frame_name = frame_base_name + '_middle' + selected_output_file_extension

            middle_frame_tensor = AI_model.inference(first_frame_tensor, last_frame_tensor)
            middle_frame = tensor_to_frame(middle_frame_tensor, h, w)
            
            image_write(frame_1_name, frame1)
            image_write(middle_frame_name, middle_frame)
            image_write(frame_2_name, frame2)

            all_files_list.append(frame_1_name)
            all_files_list.append(middle_frame_name)
            all_files_list.append(frame_2_name)

        elif frames_to_generate == 3: 
            # fluidification x4
            middle_frame_name      = frame_base_name + '_middle' + selected_output_file_extension
            after_first_frame_name = frame_base_name + '_afterfirst' + selected_output_file_extension
            prelast_frame_name     = frame_base_name + '_prelast' + selected_output_file_extension

            middle_frame_tensor       = AI_model.inference(first_frame_tensor, last_frame_tensor)
            after_first_frame_tensor  = AI_model.inference(first_frame_tensor, middle_frame_tensor)
            prelast_frame_tensor      = AI_model.inference(middle_frame_tensor, last_frame_tensor)
            
            after_first_frame = tensor_to_frame(after_first_frame_tensor, h, w)
            middle_frame      = tensor_to_frame(middle_frame_tensor, h, w)
            prelast_frame     = tensor_to_frame(prelast_frame_tensor, h, w)

            image_write(frame_1_name, frame1)
            image_write(after_first_frame_name, after_first_frame)
            image_write(middle_frame_name, middle_frame)
            image_write(prelast_frame_name, prelast_frame)
            image_write(frame_2_name, frame2)

            all_files_list.append(frame_1_name)
            all_files_list.append(after_first_frame_name)
            all_files_list.append(middle_frame_name)
            all_files_list.append(prelast_frame_name)
            all_files_list.append(frame_2_name)

    return all_files_list

def fluidify_video(video_path, 
                   AI_model, 
                   fluidification_factor, 
                   slowmotion, 
                   resize_factor, 
                   backend, 
                   selected_image_extension, 
                   selected_video_extension,
                   cpu_number, 
                   half_precision):
    
    create_temp_dir(app_name + "_temp")
    update_process_status('Extracting video frames')
    frame_list = extract_frames_from_video(video_path)

    temp_frame    = image_read(frame_list[0])
    target_height = int(temp_frame.shape[0] * resize_factor)
    target_width  = int(temp_frame.shape[1] * resize_factor)   

    update_process_status('Starting')
    how_many_frames = len(frame_list)
    all_files_list  = []
    done_frames     = 0

    for index, _ in enumerate(frame_list[:-1]):
        frame_1_name    = frame_list[index]
        frame_2_name    = frame_list[index + 1]
        frame_base_name = os.path.splitext(frame_1_name)[0]

        frame_1 = image_read(frame_list[index])
        frame_2 = image_read(frame_list[index + 1])

        if resize_factor != 1:
            frame_1, frame_2 = resize_frames(frame_1, frame_2, target_width, target_height)

        all_files_list = AI_generate_frames(frame_1, 
                                            frame_2, 
                                            frame_1_name,
                                            frame_2_name,
                                            frame_base_name,
                                            all_files_list, 
                                            AI_model, 
                                            selected_image_extension, 
                                            backend, 
                                            half_precision, 
                                            fluidification_factor)
        done_frames += 1
        if index % 8 == 0: update_process_status("Fluidifying frame " 
                                                    + str(done_frames) 
                                                    + "/" 
                                                    + str(how_many_frames))
        

    write_in_log_file("Processing video")

    # Remove duplicated frames from list
    all_files_list = list(dict.fromkeys(all_files_list))

    update_process_status("Processing fluidified video")
    video_reconstruction_by_frames(video_path, 
                                   all_files_list, 
                                   fluidification_factor, 
                                   slowmotion, 
                                   resize_factor, 
                                   cpu_number,
                                   selected_video_extension)

def check_fluidification_option(selected_fluidity_option):
    slowmotion = False
    fluidification_factor = 0

    if 'slowmotion' in selected_fluidity_option: slowmotion = True
    if '2' in selected_fluidity_option: 
        fluidification_factor = 2
    elif '4' in selected_fluidity_option:
        fluidification_factor = 4

    return fluidification_factor, slowmotion

def fluidify_orchestrator(selected_file_list,
                        selected_fluidity_option,
                        selected_AI_device, 
                        selected_image_extension,
                        selected_video_extension,
                        resize_factor,
                        cpu_number,
                        half_precision):
    
    start = timer()

    update_process_status("Preparing AI model")
    backend = torch.device(torch_directml.device(selected_AI_device))
    torch.set_num_threads(cpu_number)
    
    fluidification_factor, slowmotion = check_fluidification_option(selected_fluidity_option)

    try:
        create_temp_dir(app_name + "_temp")
        AI_model = prepare_model(backend, half_precision)

        how_many_files = len(selected_file_list)
        for index in range(how_many_files):
            update_process_status("Fluidifying " + str(index + 1) + "/" + str(how_many_files))
            fluidify_video(selected_file_list[index], 
                            AI_model,
                            fluidification_factor, 
                            slowmotion,
                            resize_factor, 
                            backend,
                            selected_image_extension, 
                            selected_video_extension,
                            cpu_number,
                            half_precision)

        update_process_status("All files completed (" + str(round(timer() - start)) + " sec.)")
        remove_dir(app_name + "_temp")

    except Exception as exception:
        update_process_status('Error while fluidifying') 
        show_error(exception)

def fludify_button_command(): 
    global selected_file_list
    global selected_fluidity_option
    global selected_AI_device 
    global selected_image_extension
    global selected_video_extension
    global resize_factor
    global cpu_number

    global process_fluidify_orchestrator

    remove_file(app_name + ".log")
    
    if user_input_checks():
        info_message.set("Loading")
        write_in_log_file("Loading")

        print("=================================================")
        print("> Starting fluidify:")
        print("   Files to fluidify: "        + str(len(selected_file_list)))
        print("   Selected fluidify option: " + str(selected_fluidity_option))
        print("   AI half precision: "        + str(half_precision))
        print("   Selected GPU: "             + str(torch_directml.device_name(selected_AI_device)))
        print("   Selected image output extension: "          + str(selected_image_extension))
        print("   Selected video output extension: "          + str(selected_video_extension))
        print("   Resize factor: "                   + str(int(resize_factor*100)) + "%")
        print("   Cpu number: "                      + str(cpu_number))
        print("=================================================")

        place_stop_button()

        process_fluidify_orchestrator = multiprocessing.Process(
                                            target = fluidify_orchestrator,
                                            args   = (selected_file_list,
                                                    selected_fluidity_option,
                                                    selected_AI_device, 
                                                    selected_image_extension,
                                                    selected_video_extension,
                                                    resize_factor,
                                                    cpu_number,
                                                    half_precision))
        process_fluidify_orchestrator.start()

        thread_wait = threading.Thread(
                                target = check_fluidify_steps,
                                daemon = True)
        thread_wait.start()



# GUI utils function ---------------------------

def user_input_checks():
    global selected_file_list
    global selected_fluidity_option
    global selected_AI_device 
    global selected_image_extension
    global resize_factor
    global cpu_number

    is_ready = True

    # files -------------------------------------------------
    try: selected_file_list = scrollable_frame_file_list.get_selected_file_list()
    except:
        info_message.set("No file selected. Please select a file")
        is_ready = False

    if len(selected_file_list) <= 0:
        info_message.set("No file selected. Please select a file")
        is_ready = False

    # resize factor -------------------------------------------------
    try: resize_factor = int(float(str(selected_resize_factor.get())))
    except:
        info_message.set("Resize % must be a numeric value")
        is_ready = False

    if resize_factor > 0: resize_factor = resize_factor/100
    else:
        info_message.set("Resize % must be a value > 0")
        is_ready = False


    # cpu number -------------------------------------------------
    try: cpu_number = int(float(str(selected_cpu_number.get())))
    except:
        info_message.set("Cpu number must be a numeric value")
        is_ready = False 

    if cpu_number <= 0:         
        info_message.set("Cpu number value must be > 0")
        is_ready = False
    else: cpu_number = int(cpu_number)


    return is_ready

def extract_video_info(video_file):
    cap          = cv2.VideoCapture(video_file)
    width        = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate   = cap.get(cv2.CAP_PROP_FPS)
    duration     = num_frames/frame_rate
    minutes      = int(duration/60)
    seconds      = duration % 60
    video_name   = str(video_file.split("/")[-1])
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False: break
        image_write("temp.jpg", frame)
        break
    cap.release()

    video_label = ( "VIDEO" + " | " + video_name + " | " + str(width) + "x" 
                   + str(height) + " | " + str(minutes) + 'm:' 
                   + str(round(seconds)) + "s | " + str(num_frames) 
                   + "frames | " + str(round(frame_rate)) + "fps" )

    ctkimage = CTkImage(Image.open("temp.jpg"), size = (25, 25))
    
    return video_label, ctkimage

def check_if_file_is_video(file):
    for video_extension in supported_video_extensions:
        if video_extension in file:
            return True

def check_supported_selected_files(uploaded_file_list):
    supported_files_list = []

    for file in uploaded_file_list:
        for supported_extension in supported_file_extensions:
            if supported_extension in file:
                supported_files_list.append(file)

    return supported_files_list

def open_files_action():
    info_message.set("Selecting files")

    uploaded_files_list = list(filedialog.askopenfilenames())
    uploaded_files_counter = len(uploaded_files_list)

    supported_files_list = check_supported_selected_files(uploaded_files_list)
    supported_files_counter = len(supported_files_list)
    
    print("> Uploaded files: " + str(uploaded_files_counter) + " => Supported files: " + str(supported_files_counter))

    if supported_files_counter > 0:
        place_up_background()

        global scrollable_frame_file_list
        scrollable_frame_file_list = ScrollableImagesTextFrame(master = window, 
                                                               fg_color = select_files_widget_color, 
                                                               bg_color = select_files_widget_color)
        scrollable_frame_file_list.place(relx = 0.5, 
                                         rely = 0.25, 
                                         relwidth = 1.0, 
                                         relheight = 0.475, 
                                         anchor = tk.CENTER)
        
        scrollable_frame_file_list.add_clean_button()

        for index in range(supported_files_counter):
            actual_file = supported_files_list[index]
            if check_if_file_is_video(actual_file):
                # video
                video_label, ctkimage = extract_video_info(actual_file)
                scrollable_frame_file_list.add_item(text_to_show = video_label, 
                                                    image = ctkimage,
                                                    file_element = actual_file)
                remove_file("temp.jpg")

    
        info_message.set("Ready")

    else: 
        info_message.set("Not supported files :(")



# GUI select from menus functions ---------------------------

def select_fluidity_option_from_menu(new_value: str):
    global selected_fluidity_option    
    selected_fluidity_option = new_value

def select_AI_device_from_menu(new_value: str):
    global selected_AI_device    

    for device in device_list:
        if device.name == new_value:
            selected_AI_device = device.index

def select_output_file_extension_from_menu(new_value: str):
    global selected_image_extension    
    selected_image_extension = new_value

def select_video_extension_from_menu(new_value: str):
    global selected_video_extension   
    selected_video_extension = new_value



# GUI info functions ---------------------------

def open_info_fluidity_option():
    info = """This widget allows you to choose between different AI fluidity option: \n
- x2 | doubles video framerate | 30fps => 60fps
- x4 | quadruples video framerate | 30fps => 120fps
- x2-slowmotion | slowmotion effect by a factor of 2 | no audio
- x4-slowmotion | slowmotion effect by a factor of 4 | no audio"""
    
    tk.messagebox.showinfo(title = 'AI fluidity', message = info)
    
def open_info_device():
    info = """This widget allows to choose the gpu to run AI with. \n 
Keep in mind that the more powerful your gpu is, 
the faster the process will be. \n
If the list is empty it means the app couldn't find 
a compatible gpu, try updating your video card driver :)"""

    tk.messagebox.showinfo(title = 'AI device', message = info)

def open_info_file_extension():
    info = """This widget allows to choose the extension of generated frames.\n
- png | very good quality | supports transparent images
- jpg | good quality | very fast
- bmp | highest quality | slow
- tiff | highest quality | very slow"""

    tk.messagebox.showinfo(title = 'AI output extension', message = info)

def open_info_resize():
    info = """This widget allows to choose the resolution input to the AI.\n
For example for a 100x100px video:
- Input resolution 50% => input to AI 50x50px
- Input resolution 100% => input to AI 100x100px
- Input resolution 200% => input to AI 200x200px """

    tk.messagebox.showinfo(title = 'Input resolution %', message = info)
    
def open_info_cpu():
    info = """This widget allows to choose how many cpus to devote to the app.\n
Where possible the app will use the number of processors you select, for example:
- Extracting frames from videos
- Resizing frames from videos
- Recostructing final video
- AI processing"""

    tk.messagebox.showinfo(title = 'Cpu number', message = info)

def open_info_video_extension():
    info = """This widget allows you to choose the video output:

- .mp4  | produces good quality and well compressed video
- .avi  | produces the highest quality video"""

    tk.messagebox.showinfo(title = 'Video output', message = info)  



# GUI place functions ---------------------------
        
def place_up_background():
    up_background = CTkLabel(master  = window, 
                            text    = "",
                            fg_color = select_files_widget_color,
                            font     = bold12,
                            anchor   = "w")
    
    up_background.place(relx = 0.5, 
                        rely = 0.0, 
                        relwidth = 1.0,  
                        relheight = 1.0,  
                        anchor = tk.CENTER)

def place_app_name():
    app_name_label = CTkLabel(master     = window, 
                              text       = app_name + " " + version,
                              text_color = "#F08080",
                              font       = bold19,
                              anchor     = "w")
    
    app_name_label.place(relx = 0.5, rely = 0.545, anchor = tk.CENTER)

    subtitle_app_name_label = CTkLabel(master  = window, 
                                    text       = second_title,
                                    text_color = "#0096FF",
                                    font       = bold18,
                                    anchor     = "w")
    
    subtitle_app_name_label.place(relx = 0.5, rely = 0.585, anchor = tk.CENTER)

def place_itch_button(): 
    itch_button = CTkButton(master     = window, 
                            width      = 30,
                            height     = 30,
                            fg_color   = "black",
                            text       = "", 
                            font       = bold11,
                            image      = logo_itch,
                            command    = openitch)
    itch_button.place(relx = 0.045, rely = 0.55, anchor = tk.CENTER)

def place_github_button():
    git_button = CTkButton(master      = window, 
                            width      = 30,
                            height     = 30,
                            fg_color   = "black",
                            text       = "", 
                            font       = bold11,
                            image      = logo_git,
                            command    = opengithub)
    git_button.place(relx = 0.045, rely = 0.61, anchor = tk.CENTER)

def place_telegram_button():
    telegram_button = CTkButton(master = window, 
                                width      = 30,
                                height     = 30,
                                fg_color   = "black",
                                text       = "", 
                                font       = bold11,
                                image      = logo_telegram,
                                command    = opentelegram)
    telegram_button.place(relx = 0.045, rely = 0.67, anchor = tk.CENTER)

def place_fluidify_button(): 
    upscale_button = CTkButton(master    = window, 
                                width      = 140,
                                height     = 30,
                                fg_color   = "#282828",
                                text_color = "#E0E0E0",
                                text       = "FLUIDIFY", 
                                font       = bold11,
                                image      = play_icon,
                                command    = fludify_button_command)
    upscale_button.place(relx = 0.8, rely = row3_y, anchor = tk.CENTER)
    
def place_stop_button(): 
    stop_button = CTkButton(master   = window, 
                            width      = 140,
                            height     = 30,
                            fg_color   = "#282828",
                            text_color = "#E0E0E0",
                            text       = "STOP", 
                            font       = bold11,
                            image      = stop_icon,
                            command    = stop_button_command)
    stop_button.place(relx = 0.8, rely = row3_y, anchor = tk.CENTER)

def place_fluidity_option_menu():
    fluidity_option_button = CTkButton(master  = window, 
                                    fg_color   = "black",
                                    text_color = "#ffbf00",
                                    text    = "AI fluidity",
                                    height   = 23,
                                    width    = 130,
                                    font     = bold11,
                                    corner_radius = 25,
                                    anchor  = "center",
                                    command = open_info_fluidity_option)

    fluidity_option_menu = CTkOptionMenu(master  = window, 
                                values     = fluidity_options_list,
                                width      = 140,
                                font       = bold11,
                                height     = 30,
                                fg_color   = "#000000",
                                anchor     = "center",
                                command    = select_fluidity_option_from_menu,
                                dropdown_font = bold11,
                                dropdown_fg_color = "#000000")

    fluidity_option_button.place(relx = 0.20, rely = row1_y - 0.05, anchor = tk.CENTER)
    fluidity_option_menu.place(relx = 0.20, rely = row1_y, anchor = tk.CENTER)

def place_AI_device_menu():
    AI_device_button = CTkButton(master  = window, 
                              fg_color   = "black",
                              text_color = "#ffbf00",
                              text     = "AI device",
                              height   = 23,
                              width    = 130,
                              font     = bold11,
                              corner_radius = 25,
                              anchor  = "center",
                              command = open_info_device)

    AI_device_menu = CTkOptionMenu(master  = window, 
                                    values   = device_list_names,
                                    width      = 140,
                                    font       = bold9,
                                    height     = 30,
                                    fg_color   = "#000000",
                                    anchor     = "center",
                                    dynamic_resizing = False,
                                    command    = select_AI_device_from_menu,
                                    dropdown_font = bold11,
                                    dropdown_fg_color = "#000000")
    
    AI_device_button.place(relx = 0.20, rely = row2_y - 0.05, anchor = tk.CENTER)
    AI_device_menu.place(relx = 0.20, rely = row2_y, anchor = tk.CENTER)

def place_file_extension_menu():
    file_extension_button = CTkButton(master  = window, 
                              fg_color   = "black",
                              text_color = "#ffbf00",
                              text     = "Frames output",
                              height   = 23,
                              width    = 130,
                              font     = bold11,
                              corner_radius = 25,
                              anchor  = "center",
                              command = open_info_file_extension)

    file_extension_menu = CTkOptionMenu(master  = window, 
                                        values     = image_extension_list,
                                        width      = 140,
                                        font       = bold11,
                                        height     = 30,
                                        fg_color   = "#000000",
                                        anchor     = "center",
                                        command    = select_output_file_extension_from_menu,
                                        dropdown_font = bold11,
                                        dropdown_fg_color = "#000000")
    
    file_extension_button.place(relx = 0.20, rely = row3_y - 0.05, anchor = tk.CENTER)
    file_extension_menu.place(relx = 0.20, rely = row3_y, anchor = tk.CENTER)

def place_video_extension_menu():
    video_extension_button = CTkButton(master  = window, 
                              fg_color   = "black",
                              text_color = "#ffbf00",
                              text     = "Video output",
                              height   = 23,
                              width    = 130,
                              font     = bold11,
                              corner_radius = 25,
                              anchor  = "center",
                              command = open_info_video_extension)

    video_extension_menu = CTkOptionMenu(master  = window, 
                                    values     = video_extension_list,
                                    width      = 140,
                                    font       = bold11,
                                    height     = 30,
                                    fg_color   = "#000000",
                                    anchor     = "center",
                                    dynamic_resizing = False,
                                    command    = select_video_extension_from_menu,
                                    dropdown_font = bold11,
                                    dropdown_fg_color = "#000000")
    
    video_extension_button.place(relx = 0.5, rely = row1_y - 0.05, anchor = tk.CENTER)
    video_extension_menu.place(relx = 0.5, rely = row1_y, anchor = tk.CENTER)

def place_resize_factor_textbox():
    resize_factor_button = CTkButton(master  = window, 
                              fg_color   = "black",
                              text_color = "#ffbf00",
                              text     = "Input resolution (%)",
                              height   = 23,
                              width    = 130,
                              font     = bold11,
                              corner_radius = 25,
                              anchor  = "center",
                              command = open_info_resize)

    resize_factor_textbox = CTkEntry(master    = window, 
                                    width      = 140,
                                    font       = bold11,
                                    height     = 30,
                                    fg_color   = "#000000",
                                    textvariable = selected_resize_factor)
    
    resize_factor_button.place(relx = 0.5, rely = row2_y - 0.05, anchor = tk.CENTER)
    resize_factor_textbox.place(relx = 0.5, rely  = row2_y, anchor = tk.CENTER)

def place_cpu_textbox():
    cpu_button = CTkButton(master  = window, 
                              fg_color   = "black",
                              text_color = "#ffbf00",
                              text     = "CPU number",
                              height   = 23,
                              width    = 130,
                              font     = bold11,
                              corner_radius = 25,
                              anchor  = "center",
                              command = open_info_cpu)

    cpu_textbox = CTkEntry(master    = window, 
                            width      = 140,
                            font       = bold11,
                            height     = 30,
                            fg_color   = "#000000",
                            textvariable = selected_cpu_number)

    cpu_button.place(relx = 0.5, rely = row3_y - 0.05, anchor = tk.CENTER)
    cpu_textbox.place(relx = 0.5, rely  = row3_y, anchor = tk.CENTER)

def place_loadFile_section():

    text_drop = """SUPPORTED FILES

VIDEO - mp4 webm mkv flv gif avi mov mpg qt 3gp"""

    input_file_text = CTkLabel(master    = window, 
                                text     = text_drop,
                                fg_color = select_files_widget_color,
                                bg_color = select_files_widget_color,
                                width   = 300,
                                height  = 150,
                                font    = bold12,
                                anchor  = "center")
    
    input_file_button = CTkButton(master = window, 
                                width    = 140,
                                height   = 30,
                                text     = "SELECT FILES", 
                                font     = bold11,
                                border_spacing = 0,
                                command        = open_files_action)

    input_file_text.place(relx = 0.5, rely = 0.22,  anchor = tk.CENTER)
    input_file_button.place(relx = 0.5, rely = 0.4, anchor = tk.CENTER)

def place_message_label():
    message_label = CTkLabel(master  = window, 
                            textvariable = info_message,
                            height       = 25,
                            font         = bold10,
                            fg_color     = "#ffbf00",
                            text_color   = "#000000",
                            anchor       = "center",
                            corner_radius = 25)
    message_label.place(relx = 0.8, rely = 0.56, anchor = tk.CENTER)



class App():
    def __init__(self, window):
        window.title('')
        width        = 650
        height       = 600
        window.geometry("650x600")
        window.minsize(width, height)
        window.iconbitmap(find_by_relative_path("Assets" + os.sep + "logo.ico"))

        place_up_background()

        place_app_name()
        place_itch_button()
        place_github_button()
        place_telegram_button()

        place_fluidity_option_menu()
        place_AI_device_menu()
        place_file_extension_menu()

        place_video_extension_menu()
        place_resize_factor_textbox()
        place_cpu_textbox()

        place_message_label()
        place_fluidify_button()

        place_loadFile_section()

if __name__ == "__main__":
    multiprocessing.freeze_support()

    set_appearance_mode("Dark")
    set_default_color_theme("dark-blue")

    window = CTk() 

    global selected_file_list
    global selected_fluidity_option
    global selected_AI_device 
    global resize_factor
    global cpu_number

    global selected_image_extension
    global selected_video_extension

    selected_file_list = []
    selected_AI_device = 0

    selected_fluidity_option = fluidity_options_list[0]
    selected_image_extension = image_extension_list[0]
    selected_video_extension = video_extension_list[0]

    info_message            = tk.StringVar()
    selected_resize_factor  = tk.StringVar()
    selected_cpu_number     = tk.StringVar()

    info_message.set("Hi :)")

    selected_resize_factor.set("70")
    cpu_count = str(int(os.cpu_count()/2))
    selected_cpu_number.set(cpu_count)

    bold8  = CTkFont(family = "Segoe UI", size = 8, weight = "bold")
    bold9  = CTkFont(family = "Segoe UI", size = 9, weight = "bold")
    bold10 = CTkFont(family = "Segoe UI", size = 10, weight = "bold")
    bold11 = CTkFont(family = "Segoe UI", size = 11, weight = "bold")
    bold12 = CTkFont(family = "Segoe UI", size = 12, weight = "bold")
    bold18 = CTkFont(family = "Segoe UI", size = 19, weight = "bold")
    bold19 = CTkFont(family = "Segoe UI", size = 19, weight = "bold")
    bold20 = CTkFont(family = "Segoe UI", size = 20, weight = "bold")
    bold21 = CTkFont(family = "Segoe UI", size = 21, weight = "bold")

    logo_git      = CTkImage(Image.open(find_by_relative_path("Assets" + os.sep + "github_logo.png")), size=(15, 15))
    logo_itch     = CTkImage(Image.open(find_by_relative_path("Assets" + os.sep + "itch_logo.png")),  size=(13, 13))
    logo_telegram = CTkImage(Image.open(find_by_relative_path("Assets" + os.sep + "telegram_logo.png")),  size=(15, 15))
    stop_icon     = CTkImage(Image.open(find_by_relative_path("Assets" + os.sep + "stop_icon.png")), size=(15, 15))
    play_icon     = CTkImage(Image.open(find_by_relative_path("Assets" + os.sep + "upscale_icon.png")), size=(15, 15))
    clear_icon    = CTkImage(Image.open(find_by_relative_path("Assets" + os.sep + "clear_icon.png")), size=(15, 15))

    app = App(window)
    window.update()
    window.mainloop()