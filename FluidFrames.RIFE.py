
# Standard library imports
import sys
from timeit import default_timer as timer
from time import sleep
from threading import Thread
from webbrowser import open as open_browser

from multiprocessing import ( 
    Process, 
    Queue          as multiprocessing_Queue,
    freeze_support as multiprocessing_freeze_support
)

from shutil import ( 
    rmtree as remove_directory,
)

from os import (
    sep         as os_separator,
    devnull     as os_devnull,
    cpu_count   as os_cpu_count,
    makedirs    as os_makedirs,
    remove      as os_remove,
)

from os.path import (
    dirname  as os_path_dirname,
    abspath  as os_path_abspath,
    join     as os_path_join,
    exists   as os_path_exists,
    splitext as os_path_splitext
)


# Third-party library imports

from PIL.Image import (
    open      as pillow_image_open,
    fromarray as pillow_image_fromarray
)

from moviepy.editor   import VideoFileClip
from moviepy.video.io import ImageSequenceClip

from torch import (
    device          as torch_device,
    inference_mode  as torch_inference_mode,
    linspace        as torch_linspace,
    cat             as torch_cat,
    load            as torch_load,
    ones            as torch_ones,
    set_num_threads as torch_set_num_threads,
    is_tensor       as torch_is_tensor,
    sigmoid         as torch_sigmoid,
    tensor          as torch_tensor,
)

from torch.nn.functional import (
    interpolate as torch_nn_interpolate,
    grid_sample as torch_nn_functional_grid_sample,
    pad         as torch_nn_functional_pad
)

from torch.nn import (
    ConvTranspose2d,
    Sequential,
    Conv2d,
    Module,
    Parameter,
    PixelShuffle,
    LeakyReLU,
)

from torch_directml import (
    device       as directml_device,
    device_count as directml_device_count,
    device_name  as directml_device_name,
    gpu_memory   as directml_gpu_memory,
    has_float64_support as directml_has_float64_support
)

from cv2 import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    COLOR_BGR2RGB,
    IMREAD_UNCHANGED,
    INTER_LINEAR,
    INTER_CUBIC,
    VideoCapture as opencv_VideoCapture,
    cvtColor     as opencv_cvtColor,
    imdecode     as opencv_imdecode,
    imencode     as opencv_imencode,
    cvtColor     as opencv_cvtColor,
    resize       as opencv_resize,
)

from numpy import (
    ndarray     as numpy_ndarray,
    frombuffer  as numpy_frombuffer,
    uint8
)

# GUI imports
from tkinter import StringVar
from customtkinter import (
    CTk,
    CTkButton,
    CTkEntry,
    CTkFont,
    CTkImage,
    CTkLabel,
    CTkOptionMenu,
    CTkScrollableFrame,
    CTkToplevel,
    filedialog,
    set_appearance_mode,
    set_default_color_theme,
)


app_name     = "FluidFrames"
second_title = "RIFE"
version      = "2.13"

dark_color   = "#080808"

githubme   = "https://github.com/Djdefrag/FluidFrames.RIFE"
telegramme = "https://linktr.ee/j3ngystudio"

AI_models_list = [ 'RIFE_4.13', 'RIFE_4.13_Lite' ]
fluidity_options_list = [ 
    'x2', 'x4', 'x8',
    'x2-slowmotion', 'x4-slowmotion', 'x8-slowmotion' 
    ]

image_extension_list  = [ '.jpg', '.png', '.bmp', '.tiff' ]
video_extension_list  = [ '.mp4 (x264)', '.mp4 (x265)', '.avi' ]
save_frames_list      = [ 'Enabled', 'Disabled' ]

offset_y_options = 0.125
row0_y           = 0.56
row1_y           = row0_y + offset_y_options
row2_y           = row1_y + offset_y_options
row3_y           = row2_y + offset_y_options

offset_x_options = 0.28
column1_x        = 0.5
column0_x        = column1_x - offset_x_options
column2_x        = column1_x + offset_x_options

COMPLETED_STATUS = "Completed"
ERROR_STATUS = "Error"
STOP_STATUS = "Stop"

if sys.stdout is None: sys.stdout = open(os_devnull, "w")
if sys.stderr is None: sys.stderr = open(os_devnull, "w")

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
                                '.qt', '.3gp', '.mpg', '.mpeg']



# ------------------ AI ------------------

class Head(Module):
    def __init__(self):
        super(Head, self).__init__()
        self.cnn0 = Conv2d(3, 32, 3, 2, 1)
        self.cnn1 = Conv2d(32, 32, 3, 1, 1)
        self.cnn2 = Conv2d(32, 32, 3, 1, 1)
        self.cnn3 = ConvTranspose2d(32, 8, 4, 2, 1)
        self.relu = LeakyReLU(0.2, True)

    def forward(self, x, feat=False):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        if feat:
            return [x0, x1, x2, x3]
        return x3

class ResConv(Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = Parameter(torch_ones((1, c, 1, 1)), requires_grad=True)
        self.relu = LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)

class IFBlock(Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()

        self.conv0 = Sequential(
            Sequential(Conv2d(in_planes, c//2, 3, 2, 1, bias=True), LeakyReLU(0.2, True)),
            Sequential(Conv2d(c//2, c, 3, 2, 1, bias=True), LeakyReLU(0.2, True))
        )
        self.convblock = Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = Sequential(
            ConvTranspose2d(c, 4*6, 4, 2, 1),
            PixelShuffle(2)
        )

    def forward(self, x, flow=None, scale=1):
        x = torch_nn_interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = torch_nn_interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x    = torch_cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp  = self.lastconv(feat)
        tmp  = torch_nn_interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask

class RIFE_413(Module):
    def __init__(self, backend):
        super(RIFE_413, self).__init__()
        self.block0 = IFBlock(7+16, c=192)
        self.block1 = IFBlock(8+4+16, c=128)
        self.block2 = IFBlock(8+4+16, c=96)
        self.block3 = IFBlock(8+4+16, c=64)
        self.encode = Head()   

        self.backend = backend     

    def forward(self, x, timestep=0.5, scale_list=[8, 4, 2, 1]):
        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]

        if not torch_is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])

        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])
        flow_list = []
        merged    = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        block = [self.block0, self.block1, self.block2, self.block3]
        for i in range(4):
            if flow is None:
                flow, mask = block[i](torch_cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1), None, scale=scale_list[i])
            else:
                wf0 = self.warp(f0, flow[:, :2])
                wf1 = self.warp(f1, flow[:, 2:4])
                fd, m0 = block[i](torch_cat((warped_img0[:, :3], warped_img1[:, :3], wf0, wf1, timestep, mask), 1), flow, scale=scale_list[i])
                mask = m0
                flow = flow + fd

            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = self.warp(img0, flow[:, :2])
            warped_img1 = self.warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))

        mask = torch_sigmoid(mask)
        merged[3] = (warped_img0 * mask + warped_img1 * (1 - mask))

        return flow_list, mask_list[3], merged
    
    def warp(
        self,
        tenInput: any, 
        tenFlow: any, 
        ) -> torch_tensor:
    
        backwarp_tenGrid = {}

        k = (str(tenFlow.device), str(tenFlow.size()))

        if k not in backwarp_tenGrid:
            tenHorizontal       = torch_linspace(-1.0, 1.0, tenFlow.shape[3], device = self.backend).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1,  tenFlow.shape[2], -1)
            tenVertical         = torch_linspace(-1.0, 1.0, tenFlow.shape[2], device = self.backend).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
            backwarp_tenGrid[k] = torch_cat([tenHorizontal, tenVertical], 1).to(self.backend, non_blocking = True)

        tenFlow = torch_cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

        g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)

        return torch_nn_functional_grid_sample(input = tenInput, grid = g, mode = 'bilinear', padding_mode = 'border', align_corners = True)

class RIFE_413_Lite(Module):

    def __init__(self, backend):
        super(RIFE_413_Lite, self).__init__()
        self.block0 = IFBlock(7+8, c=128)
        self.block1 = IFBlock(8+4+8, c=96)
        self.block2 = IFBlock(8+4+8, c=64)
        self.block3 = IFBlock(8+4+8, c=48)
        self.encode = Sequential(
            Conv2d(3, 32, 3, 2, 1),
            LeakyReLU(0.2, True),
            Conv2d(32, 32, 3, 1, 1),
            LeakyReLU(0.2, True),
            Conv2d(32, 32, 3, 1, 1),
            LeakyReLU(0.2, True),
            ConvTranspose2d(32, 4, 4, 2, 1)
        )

        self.backend = backend

    def forward(self, x, timestep=0.5, scale_list=[8, 4, 2, 1]):
        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]

        if not torch_is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])

        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        block = [self.block0, self.block1, self.block2, self.block3]

        for i in range(4):
            if flow is None:
                flow, mask = block[i](torch_cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1), None, scale=scale_list[i])
            else:
                wf0 = self.warp(f0, flow[:, :2])
                wf1 = self.warp(f1, flow[:, 2:4])
                fd, m0 = block[i](torch_cat((warped_img0[:, :3], warped_img1[:, :3], wf0, wf1, timestep, mask), 1), flow, scale=scale_list[i])
                mask = m0
                flow = flow + fd

            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = self.warp(img0, flow[:, :2])
            warped_img1 = self.warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))

        mask = torch_sigmoid(mask)
        merged[3] = (warped_img0 * mask + warped_img1 * (1 - mask))

        return flow_list, mask_list[3], merged
    
    def warp(
        self,
        tenInput: any, 
        tenFlow: any, 
        ) -> torch_tensor:
    
        backwarp_tenGrid = {}

        k = (str(tenFlow.device), str(tenFlow.size()))

        if k not in backwarp_tenGrid:
            tenHorizontal       = torch_linspace(-1.0, 1.0, tenFlow.shape[3], device = self.backend).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1,  tenFlow.shape[2], -1)
            tenVertical         = torch_linspace(-1.0, 1.0, tenFlow.shape[2], device = self.backend).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
            backwarp_tenGrid[k] = torch_cat([tenHorizontal, tenVertical], 1).to(self.backend, non_blocking = True)

        tenFlow = torch_cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

        g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)

        return torch_nn_functional_grid_sample(input = tenInput, grid = g, mode = 'bilinear', padding_mode = 'border', align_corners = True)

@torch_inference_mode(True)
def load_AI_model(
        selected_AI_model: str,
        backend: directml_device
        ) -> any:

    def convert(param):
        return {
            k.replace("module.", ""): v
            for k, v in param.items()
            if "module." in k
        }

    model_path = find_by_relative_path(f"AI{os_separator}{selected_AI_model}.pkl")

    if selected_AI_model == "RIFE_4.13":
        model = RIFE_413(backend)
    elif selected_AI_model == "RIFE_4.13_Lite":
        model = RIFE_413_Lite(backend)

    pretrained_model = torch_load(
        model_path, 
        map_location = torch_device('cpu')
        )
    
    model.load_state_dict(convert(pretrained_model))
    model.eval()

    model.to(backend, non_blocking = True)

    return model

@torch_inference_mode(True)
def AI_generate_frames(
        AI_model: any,
        backend: directml_device,
        frame1: numpy_ndarray, 
        frame2: numpy_ndarray, 
        frame_1_name: str,
        frame_2_name: str,
        frame_base_name: str,
        all_video_frames_path_list: list,
        selected_output_file_extension: str, 
        fluidification_factor: int
        ) -> list:
    
    frames_to_generate = fluidification_factor - 1

    frame_1_tensor, frame_2_tensor, h, w = frames_to_tensors(frame1, frame2, backend)

    if frames_to_generate == 1: 
        # fluidification x2
        frame_1_1_name = f"{frame_base_name}_.1{selected_output_file_extension}"

        frame_1_1 = tensor_to_frame(AI_interpolation(AI_model, frame_1_tensor, frame_2_tensor), h, w)
        
        image_write(frame_1_name,   frame1)
        image_write(frame_1_1_name, frame_1_1)
        image_write(frame_2_name,   frame2)

        all_video_frames_path_list.append(frame_1_name)
        all_video_frames_path_list.append(frame_1_1_name)
        all_video_frames_path_list.append(frame_2_name)

    elif frames_to_generate == 3: 
        # fluidification x4
        frame_1_1_name = f"{frame_base_name}_.1{selected_output_file_extension}"
        frame_1_2_name = f"{frame_base_name}_.2{selected_output_file_extension}"
        frame_1_3_name = f"{frame_base_name}_.3{selected_output_file_extension}"

        frame_1_2_tensor = AI_interpolation(AI_model, frame_1_tensor, frame_2_tensor)
        frame_1_1_tensor = AI_interpolation(AI_model, frame_1_tensor, frame_1_2_tensor)
        frame_1_3_tensor = AI_interpolation(AI_model, frame_1_2_tensor, frame_2_tensor)
        
        frame_1_1 = tensor_to_frame(frame_1_1_tensor, h, w)
        frame_1_2 = tensor_to_frame(frame_1_2_tensor, h, w)
        frame_1_3 = tensor_to_frame(frame_1_3_tensor, h, w)

        image_write(frame_1_name,   frame1)
        image_write(frame_1_1_name, frame_1_1)
        image_write(frame_1_2_name, frame_1_2)
        image_write(frame_1_3_name, frame_1_3)
        image_write(frame_2_name,   frame2)

        all_video_frames_path_list.append(frame_1_name)
        all_video_frames_path_list.append(frame_1_1_name)
        all_video_frames_path_list.append(frame_1_2_name)
        all_video_frames_path_list.append(frame_1_3_name)
        all_video_frames_path_list.append(frame_2_name)

    elif frames_to_generate == 7: 
        # fluidification x8
        frame_1_1_name = f"{frame_base_name}_.1{selected_output_file_extension}"
        frame_1_2_name = f"{frame_base_name}_.2{selected_output_file_extension}"
        frame_1_3_name = f"{frame_base_name}_.3{selected_output_file_extension}"
        frame_1_4_name = f"{frame_base_name}_.4{selected_output_file_extension}"
        frame_1_5_name = f"{frame_base_name}_.5{selected_output_file_extension}"
        frame_1_6_name = f"{frame_base_name}_.6{selected_output_file_extension}"
        frame_1_7_name = f"{frame_base_name}_.7{selected_output_file_extension}"

        frame_1_4_tensor   = AI_interpolation(AI_model, frame_1_tensor, frame_2_tensor)
        frame_1_2_tensor   = AI_interpolation(AI_model, frame_1_tensor, frame_1_4_tensor)
        frame_1_1_tensor   = AI_interpolation(AI_model, frame_1_tensor, frame_1_2_tensor)
        frame_1_3_tensor   = AI_interpolation(AI_model, frame_1_2_tensor, frame_1_4_tensor)

        frame_1_6_tensor   = AI_interpolation(AI_model, frame_1_4_tensor, frame_2_tensor)
        frame_1_5_tensor   = AI_interpolation(AI_model, frame_1_4_tensor, frame_1_6_tensor)
        frame_1_7_tensor   = AI_interpolation(AI_model, frame_1_6_tensor, frame_2_tensor)

        frame_1_1 = tensor_to_frame(frame_1_1_tensor, h, w)
        frame_1_2 = tensor_to_frame(frame_1_2_tensor, h, w)
        frame_1_3 = tensor_to_frame(frame_1_3_tensor, h, w)
        frame_1_4 = tensor_to_frame(frame_1_4_tensor, h, w)
        frame_1_5 = tensor_to_frame(frame_1_5_tensor, h, w)
        frame_1_6 = tensor_to_frame(frame_1_6_tensor, h, w)
        frame_1_7 = tensor_to_frame(frame_1_7_tensor, h, w)

        image_write(frame_1_name, frame1)
        image_write(frame_1_1_name, frame_1_1)
        image_write(frame_1_2_name, frame_1_2)
        image_write(frame_1_3_name, frame_1_3)
        image_write(frame_1_4_name, frame_1_4)
        image_write(frame_1_5_name, frame_1_5)
        image_write(frame_1_6_name, frame_1_6)
        image_write(frame_1_7_name, frame_1_7)
        image_write(frame_2_name, frame2)

        all_video_frames_path_list.append(frame_1_name)
        all_video_frames_path_list.append(frame_1_1_name)
        all_video_frames_path_list.append(frame_1_2_name)
        all_video_frames_path_list.append(frame_1_3_name)
        all_video_frames_path_list.append(frame_1_4_name)
        all_video_frames_path_list.append(frame_1_5_name)
        all_video_frames_path_list.append(frame_1_6_name)
        all_video_frames_path_list.append(frame_1_7_name)
        all_video_frames_path_list.append(frame_2_name)

    return all_video_frames_path_list

def AI_interpolation(
        AI_model: any, 
        image1: numpy_ndarray, 
        image2: numpy_ndarray, 
        timestep = 0.5, 
        scale = 1.0) -> numpy_ndarray:
    
    imgs = torch_cat((image1, image2), 1)
    scale_list = [8/scale, 4/scale, 2/scale, 1/scale]
    _, _, merged = AI_model(imgs, timestep, scale_list)
    return merged[3]

def frames_to_tensors(
        frame_1: numpy_ndarray, 
        frame_2: numpy_ndarray, 
        backend: directml_device,
    ) -> tuple:

    tensor_1 = (torch_tensor(frame_1.transpose(2, 0, 1)).to(backend, non_blocking=True) / 255.0).unsqueeze(0)
    tensor_2 = (torch_tensor(frame_2.transpose(2, 0, 1)).to(backend, non_blocking=True) / 255.0).unsqueeze(0)

    _, _, original_height, original_width = tensor_1.shape

    padded_height = ((original_height - 1) // 32 + 1) * 32
    padded_width  = ((original_width - 1) // 32 + 1) * 32

    padding = (0, padded_width - original_width, 0, padded_height - original_height)

    padded_tensor_1 = torch_nn_functional_pad(tensor_1, padding)
    padded_tensor_2 = torch_nn_functional_pad(tensor_2, padding)

    return padded_tensor_1, padded_tensor_2, original_height, original_width

def tensor_to_frame(
        result: torch_tensor, 
        height: int, 
        width: int
        ) -> any:
    
    return (result[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:height, :width]



# GUI utils ---------------------------

class ScrollableImagesTextFrame(CTkScrollableFrame):

    def __init__(
            self, 
            master,
            selected_file_list, 
            **kwargs
            ) -> None:
        
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight = 1)

        self.file_list = selected_file_list
        self._create_widgets()

    def _create_widgets(
            self,
            ) -> None:
        
        self.add_clean_button()

        index_row = 1

        for file_path in self.file_list:

            if check_if_file_is_video(file_path):
                infos, icon = self.extract_video_info(file_path)
        
            label = CTkLabel(
                self, 
                text       = infos,
                image      = icon, 
                font       = bold11,
                text_color = "#E0E0E0",
                compound   = "left", 
                padx       = 10,
                pady       = 5,
                anchor     = "center"
            )
                            
            label.grid(
                row = index_row, 
                column = 0, 
                pady = (3, 3), 
                padx = (3, 3), 
                sticky = "w"
            )
            
            index_row +=1

    def add_clean_button(
            self: any
            ) -> None:
        
        button = CTkButton(
            self, 
            image        = clear_icon,
            font         = bold11,
            text         = "CLEAN", 
            compound     = "left",
            width        = 100, 
            height       = 28,
            border_width = 1,
            fg_color     = "#282828",
            text_color   = "#E0E0E0",
            border_color = "#0096FF"
            )

        button.configure(command=lambda: self.clean_all_items())
        button.grid(row = 0, column=2, pady=(7, 7), padx = (0, 7))
        
    def get_selected_file_list(
        self: any
        ) -> list: 
    
        return self.file_list  

    def clean_all_items(
            self: any
            ) -> None:
        
        self.file_list = []
        self.destroy()
        place_loadFile_section()

    def extract_video_info(
        self: any,
        video_file: str
        ) -> tuple:
        
        cap          = opencv_VideoCapture(video_file)
        width        = round(cap.get(CAP_PROP_FRAME_WIDTH))
        height       = round(cap.get(CAP_PROP_FRAME_HEIGHT))
        num_frames   = int(cap.get(CAP_PROP_FRAME_COUNT))
        frame_rate   = cap.get(CAP_PROP_FPS)
        duration     = num_frames/frame_rate
        minutes      = int(duration/60)
        seconds      = duration % 60
        video_name   = str(video_file.split("/")[-1])
        
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False: break
            frame = opencv_cvtColor(frame, COLOR_BGR2RGB)
            video_icon = CTkImage(
                pillow_image_fromarray(frame, mode="RGB"), 
                size = (25, 25)
                )
            break
        cap.release()

        video_infos = f"{video_name} • {width}x{height} • {minutes}m:{round(seconds)}s • {num_frames}frames • {round(frame_rate, 2)}fps"
        
        return video_infos, video_icon
    
class CTkMessageBox(CTkToplevel):

    def __init__(
            self,
            messageType: str,
            title: str,
            subtitle: str,
            default_value: str,
            option_list: list,
            ) -> None:

        super().__init__()

        self._running: bool = False

        self._messageType = messageType
        self._title = title        
        self._subtitle = subtitle
        self._default_value = default_value
        self._option_list = option_list
        self._ctkwidgets_index = 0

        self.title('')
        self.lift()                          # lift window on top
        self.attributes("-topmost", True)    # stay on top
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.after(10, self._create_widgets)  # create widgets with slight delay, to avoid white flickering of background
        self.resizable(False, False)
        self.grab_set()                       # make other windows not clickable

    def _ok_event(
            self, 
            event = None
            ) -> None:
        self.grab_release()
        self.destroy()

    def _on_closing(
            self
            ) -> None:
        self.grab_release()
        self.destroy()

    def createEmptyLabel(
            self
            ) -> CTkLabel:
        
        return CTkLabel(master = self, 
                        fg_color = "transparent",
                        width    = 500,
                        height   = 17,
                        text     = '')

    def placeInfoMessageTitleSubtitle(
            self,
            ) -> None:

        spacingLabel1 = self.createEmptyLabel()
        spacingLabel2 = self.createEmptyLabel()

        if self._messageType == "info":
            title_subtitle_text_color = "#3399FF"
        elif self._messageType == "error":
            title_subtitle_text_color = "#FF3131"

        titleLabel = CTkLabel(
            master     = self,
            width      = 500,
            anchor     = 'w',
            justify    = "left",
            fg_color   = "transparent",
            text_color = title_subtitle_text_color,
            font       = bold22,
            text       = self._title
            )
        
        if self._default_value != None:
            defaultLabel = CTkLabel(
                master     = self,
                width      = 500,
                anchor     = 'w',
                justify    = "left",
                fg_color   = "transparent",
                text_color = "#3399FF",
                font       = bold17,
                text       = f"Default: {self._default_value}"
                )
        
        subtitleLabel = CTkLabel(
            master     = self,
            width      = 500,
            anchor     = 'w',
            justify    = "left",
            fg_color   = "transparent",
            text_color = title_subtitle_text_color,
            font       = bold14,
            text       = self._subtitle
            )
        
        spacingLabel1.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 0, pady = 0, sticky = "ew")
        
        self._ctkwidgets_index += 1
        titleLabel.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 25, pady = 0, sticky = "ew")
        
        if self._default_value != None:
            self._ctkwidgets_index += 1
            defaultLabel.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 25, pady = 0, sticky = "ew")
        
        self._ctkwidgets_index += 1
        subtitleLabel.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 25, pady = 0, sticky = "ew")
        
        self._ctkwidgets_index += 1
        spacingLabel2.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 0, pady = 0, sticky = "ew")

    def placeInfoMessageOptionsText(
            self,
            ) -> None:
        
        for option_text in self._option_list:
            optionLabel = CTkLabel(master = self,
                                    width  = 600,
                                    height = 45,
                                    corner_radius = 6,
                                    anchor     = 'w',
                                    justify    = "left",
                                    text_color = "#C0C0C0",
                                    fg_color   = "#282828",
                                    bg_color   = "transparent",
                                    font       = bold12,
                                    text       = option_text)
            
            self._ctkwidgets_index += 1
            optionLabel.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 25, pady = 4, sticky = "ew")

        spacingLabel3 = self.createEmptyLabel()

        self._ctkwidgets_index += 1
        spacingLabel3.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 0, pady = 0, sticky = "ew")

    def placeInfoMessageOkButton(
            self
            ) -> None:
        
        ok_button = CTkButton(
            master  = self,
            command = self._ok_event,
            text    = 'OK',
            width   = 125,
            font         = bold11,
            border_width = 1,
            fg_color     = "#282828",
            text_color   = "#E0E0E0",
            border_color = "#0096FF"
            )
        
        self._ctkwidgets_index += 1
        ok_button.grid(row = self._ctkwidgets_index, column = 1, columnspan = 1, padx = (10, 20), pady = (10, 20), sticky = "e")

    def _create_widgets(
            self
            ) -> None:

        self.grid_columnconfigure((0, 1), weight=1)
        self.rowconfigure(0, weight=1)

        self.placeInfoMessageTitleSubtitle()
        self.placeInfoMessageOptionsText()
        self.placeInfoMessageOkButton()

def create_info_button(
        command: any, 
        text: str
        ) -> CTkButton:
    
    return CTkButton(
        master  = window, 
        command = command,
        text          = text,
        fg_color      = "transparent",
        hover_color   = "#181818",
        text_color    = "#C0C0C0",
        anchor        = "w",
        height        = 23,
        width         = 150,
        corner_radius = 12,
        font          = bold12,
        image         = info_icon
        )

def create_option_menu(
        command: any, 
        values: list
        ) -> CTkOptionMenu:
    
    return CTkOptionMenu(
        master  = window, 
        command = command,
        values  = values,
        width              = 150,
        height             = 31,
        corner_radius      = 6,
        dropdown_font      = bold11,
        font               = bold11,
        anchor             = "center",
        text_color         = "#C0C0C0",
        fg_color           = "#000000",
        button_color       = "#000000",
        button_hover_color = "#000000",
        dropdown_fg_color  = "#000000"
        )

def create_text_box(
        textvariable: StringVar,
        ) -> CTkEntry:
    
    return CTkEntry(
        master        = window, 
        textvariable  = textvariable,
        border_width  = 1,
        corner_radius = 6,
        width         = 150,
        height        = 30,
        font          = bold11,
        justify       = "center",
        fg_color      = "#000000",
        border_color  = "#404040",
        )

def test_callback(a, b, c):
    print("Pippo")



# File Utils functions ------------------------

def find_by_relative_path(
        relative_path: str
        ) -> str:
    
    base_path = getattr(sys, '_MEIPASS', os_path_dirname(os_path_abspath(__file__)))
    return os_path_join(base_path, relative_path)

def remove_file(
        file_name: str
        ) -> None: 

    if os_path_exists(file_name): 
        os_remove(file_name)

def remove_dir(
        name_dir: str
        ) -> None:
    
    if os_path_exists(name_dir): 
        remove_directory(name_dir)

def create_dir(
        name_dir: str
        ) -> None:
    
    if os_path_exists(name_dir): 
        remove_directory(name_dir)
    if not os_path_exists(name_dir): 
        os_makedirs(name_dir, mode=0o777)

def remove_temp_files() -> None:
    remove_dir(app_name + "_temp")



# Image/video Utils functions ------------------------

def image_write(
        file_path: str, 
        file_data: any
        ) -> None: 
    
    _, file_extension = os_path_splitext(file_path)
    opencv_imencode(file_extension, file_data)[1].tofile(file_path)

def image_read(
        file_path: str, 
        flags: int = IMREAD_UNCHANGED
        ) -> numpy_ndarray: 
    
    with open(file_path, 'rb') as file:
        image_data    = file.read()
        image_buffer  = numpy_frombuffer(image_data, uint8)
        image_decoded = opencv_imdecode(image_buffer, flags)
        return image_decoded

def resize_frames(
        frame_1: numpy_ndarray, 
        frame_2: numpy_ndarray, 
        resize_factor: int, 
        target_width: int, 
        target_height: int
        ) -> tuple:

    if resize_factor == 1: 
        return frame_1, frame_2
    else: 
        frame_1_resized = opencv_resize(frame_1, (target_width, target_height), interpolation = INTER_LINEAR)    
        frame_2_resized = opencv_resize(frame_2, (target_width, target_height), interpolation = INTER_CUBIC)
        return frame_1_resized, frame_2_resized

def extract_video_fps(
        video_path: str
        ) -> float:
    
    video_capture = opencv_VideoCapture(video_path)
    frame_rate    = video_capture.get(CAP_PROP_FPS)
    video_capture.release()
    return frame_rate

def extract_video_frames_and_audio(
        target_directory: str,
        video_path: str, 
    ) -> list[str]:

    create_dir(target_directory)

    with VideoFileClip(video_path) as video_file_clip:
        try: 
            audio_path = f"{target_directory}{os_separator}audio.mp3"
            video_file_clip.audio.write_audiofile(audio_path, verbose = False, logger = None)
        except:
            pass
        
        video_frame_rate = extract_video_fps(video_path)
        frames_sequence_path = f"{target_directory}{os_separator}frame_%01d.jpg"
        video_frames_list = video_file_clip.write_images_sequence(
            nameformat = frames_sequence_path, 
            fps        = video_frame_rate,
            verbose    = False, 
            withmask   = True,
            logger     = None, 
            )
        
    return video_frames_list, audio_path

def video_reconstruction_by_frames(
        video_path: str, 
        audio_path: str,
        all_video_frames_paths: list, 
        selected_AI_model: str,
        fluidification_factor: int,
        slowmotion: bool,
        resize_factor: int,
        cpu_number: int,
        selected_video_extension: str
        ) -> None:
    
    frame_rate = extract_video_fps(video_path)

    if not slowmotion: 
        frame_rate = frame_rate * fluidification_factor

    # Choose the appropriate codec
    if selected_video_extension == '.mp4 (x264)':  
        selected_video_extension = '.mp4'
        codec = 'libx264'
    elif selected_video_extension == '.mp4 (x265)':  
        selected_video_extension = '.mp4'
        codec = 'libx265'
    elif selected_video_extension == '.avi': 
        selected_video_extension = '.avi' 
        codec = 'png'

    upscaled_video_path = prepare_output_video_filename(
        video_path, 
        selected_AI_model,
        fluidification_factor, 
        slowmotion, 
        resize_factor, 
        selected_video_extension
    )

    clip = ImageSequenceClip.ImageSequenceClip(all_video_frames_paths, fps = frame_rate)
    if os_path_exists(audio_path):
        clip.write_videofile(
            upscaled_video_path,
            fps     = frame_rate,
            audio   = audio_path,
            codec   = codec,
            bitrate = '16M',
            #ffmpeg_params = [ '-vf', 'scale=out_range=full' ],
            verbose = False,
            logger  = None,
            threads = cpu_number
        )
    else:
        clip.write_videofile(
            upscaled_video_path,
            fps     = frame_rate,
            codec   = codec,
            bitrate = '16M',
            #ffmpeg_params = [ '-vf', 'scale=out_range=full' ],
            verbose = False,
            logger  = None,
            threads = cpu_number
            )      

def calculate_time_to_complete_video(
        start_timer: float, 
        end_timer: float, 
        how_many_frames: int, 
        index_frame: int
        ) -> str:
    
    seconds_for_frame = round(end_timer - start_timer, 2)
    frames_left       = how_many_frames - (index_frame + 1)
    seconds_left      = seconds_for_frame * frames_left

    hours_left   = seconds_left // 3600
    minutes_left = (seconds_left % 3600) // 60
    seconds_left = round((seconds_left % 3600) % 60)

    time_left = ""

    if int(hours_left) > 0: 
        time_left = f"{int(hours_left):02d}h"
    
    if int(minutes_left) > 0: 
        time_left = f"{time_left}{int(minutes_left):02d}m"

    if seconds_left > 0: 
        time_left = f"{time_left}{seconds_left:02d}s"

    return time_left   

def update_process_status_videos(
        processing_queue: multiprocessing_Queue, 
        file_number: int, 
        start_timer: float, 
        index_frame: int, 
        how_many_frames: int
        ) -> None:
    
    if index_frame != 0 and (index_frame + 1) % 4 == 0:    
        percent_complete = (index_frame + 1) / how_many_frames * 100 
        end_timer        = timer()
        time_left        = calculate_time_to_complete_video(start_timer, end_timer, how_many_frames, index_frame)
    
        write_process_status(processing_queue, f"{file_number}. Fluidifying video {percent_complete:.2f}% ({time_left})")



# Core functions ------------------------

def stop_thread() -> None:
    stop = 1 + "x"

def check_fluidify_steps() -> None:
    sleep(1)

    try:
        while True:
            actual_step = read_process_status()

            if actual_step == COMPLETED_STATUS:
                info_message.set(f"All files completed! :)")
                stop_fluidify_process()
                remove_temp_files()
                stop_thread()

            elif actual_step == STOP_STATUS:
                info_message.set(f"Fluidify stopped")
                stop_fluidify_process()
                remove_temp_files()
                stop_thread()

            elif ERROR_STATUS in actual_step:
                error_message = f"Error during fluidify process :("
                error = actual_step.replace(ERROR_STATUS, "")
                info_message.set(error_message)
                show_error_message(error)
                remove_temp_files()
                stop_thread()

            else:
                info_message.set(actual_step)

            sleep(1)
    except:
        place_fluidify_button()

def read_process_status() -> None:
    return processing_queue.get()

def write_process_status(
        processing_queue: multiprocessing_Queue,
        step: str
        ) -> None:
    
    print(f"{step}")
    while not processing_queue.empty(): processing_queue.get()
    processing_queue.put(f"{step}")

def stop_fluidify_process() -> None:
    global process_fluidify_orchestrator

    try:
        process_fluidify_orchestrator
    except:
        pass
    else:
        process_fluidify_orchestrator.terminate()
        process_fluidify_orchestrator.join()

def stop_button_command() -> None:
    stop_fluidify_process()
    write_process_status(processing_queue, f"{STOP_STATUS}")

def prepare_output_video_filename(
        video_path: str, 
        selected_AI_model: str,
        fluidification_factor: int, 
        slowmotion: bool, 
        resize_factor: int, 
        selected_video_extension: str
        ) -> str:
    
    result_path, _ = os_path_splitext(video_path)

    # Selected AI model
    to_append = f"_{selected_AI_model}x{str(fluidification_factor)}"

    # Slowmotion?
    if slowmotion: to_append += f"_slowmo_"

    # Selected resize
    to_append += f"_Resize-{str(int(resize_factor * 100))}"

    # Video output
    to_append += f"{selected_video_extension}"

    result_path += to_append

    return result_path

def prepare_output_video_frames_directory_name(
        video_path: str, 
        selected_AI_model: str,
        fluidification_factor: int, 
        slowmotion: bool, 
        resize_factor: int, 
        ) -> str:
    
    result_path, _ = os_path_splitext(video_path)

    # Selected AI model
    to_append = f"_{selected_AI_model}x{str(fluidification_factor)}"

    # Slowmotion?
    if slowmotion: to_append += f"_slowmo_"

    # Selected resize
    to_append += f"_Resize-{str(int(resize_factor * 100))}"

    result_path += to_append

    return result_path

def get_video_target_resolution(
        first_video_frame: numpy_ndarray, 
        resize_factor: int
        ) -> tuple:
    
    temp_frame    = image_read(first_video_frame)
    target_height = int(temp_frame.shape[0] * resize_factor)
    target_width  = int(temp_frame.shape[1] * resize_factor) 

    return target_height, target_width

def check_fluidification_option(
        selected_fluidity_option: str
        ) -> tuple:
    
    slowmotion = False
    fluidification_factor = 0

    if 'slowmotion' in selected_fluidity_option: slowmotion = True

    if '2' in selected_fluidity_option:   fluidification_factor = 2
    elif '4' in selected_fluidity_option: fluidification_factor = 4
    elif '8' in selected_fluidity_option: fluidification_factor = 8

    return fluidification_factor, slowmotion

def fludify_button_command() -> None: 
    global selected_file_list
    global selected_AI_model
    global selected_fluidity_option
    global selected_AI_device 
    global selected_image_extension
    global selected_video_extension
    global resize_factor
    global cpu_number
    global selected_save_frames

    global process_fluidify_orchestrator
    
    if user_input_checks():
        info_message.set("Loading")

        print("=" * 50)
        print(f"> Starting fluidify:")
        print(f"   Files to fluidify: {len(selected_file_list)}")
        print(f"   Selected AI model: {selected_AI_model}")
        print(f"   Selected fluidify option: {selected_fluidity_option}")
        print(f"   Selected GPU: {directml_device_name(selected_AI_device)}")
        print(f"   Selected image output extension: {selected_image_extension}")
        print(f"   Selected video output extension: {selected_video_extension}")
        print(f"   Resize factor: {int(resize_factor * 100)}%")
        print(f"   Cpu number: {cpu_number}")
        print(f"   Save frames: {selected_save_frames}")
        print("=" * 50)

        backend = torch_device(directml_device(selected_AI_device))

        place_stop_button()

        process_fluidify_orchestrator = Process(
            target = fluidify_orchestrator,
            args = (
                processing_queue, 
                selected_file_list, 
                selected_AI_model,
                selected_fluidity_option, 
                backend, 
                selected_image_extension, 
                selected_video_extension, 
                resize_factor, 
                cpu_number, 
                selected_save_frames
            )
        )
        process_fluidify_orchestrator.start()

        thread_wait = Thread(
            target = check_fluidify_steps
        )
        thread_wait.start()

def fluidify_orchestrator(
        processing_queue: multiprocessing_Queue,
        selected_file_list: list,
        selected_AI_model: str,
        selected_fluidity_option: str,
        backend: directml_device, 
        selected_image_extension: str,
        selected_video_extension: str,
        resize_factor: int,
        cpu_number: int,
        selected_save_frames: bool
        ) -> None:
        
    torch_set_num_threads(2)
    
    fluidification_factor, slowmotion = check_fluidification_option(selected_fluidity_option)

    try:
        write_process_status(processing_queue, f"Loading AI model")
        
        AI_model = load_AI_model(selected_AI_model, backend)

        how_many_files = len(selected_file_list)
        for file_number in range(how_many_files):
            file_path   = selected_file_list[file_number]
            file_number = file_number + 1

            fluidify_video(
                processing_queue,
                file_path, 
                file_number,
                AI_model,
                backend,
                selected_AI_model,
                fluidification_factor, 
                slowmotion,
                resize_factor, 
                selected_image_extension, 
                selected_video_extension,
                cpu_number,
                selected_save_frames
            )

        write_process_status(processing_queue, f"{COMPLETED_STATUS}")

    except Exception as exception:
        write_process_status(processing_queue, f"{ERROR_STATUS}{str(exception)}")

def fluidify_video(
        processing_queue: multiprocessing_Queue,
        video_path: str, 
        file_number: int,
        AI_model: any,
        backend: directml_device, 
        selected_AI_model: str,
        fluidification_factor: int, 
        slowmotion: bool, 
        resize_factor: int, 
        selected_image_extension: str, 
        selected_video_extension: str,
        cpu_number: int, 
        selected_save_frames: bool
        ) -> None:
        
    # Directory for video frames and audio
    target_directory = prepare_output_video_frames_directory_name(
        video_path, 
        selected_AI_model, 
        fluidification_factor, 
        slowmotion,
        resize_factor
        )
    
    # Extract video frames and audio
    write_process_status(processing_queue, f"{file_number}. Extracting video frames")
    frame_list_paths, audio_path = extract_video_frames_and_audio(target_directory, video_path)
    target_height, target_width = get_video_target_resolution(frame_list_paths[0], resize_factor)  

    write_process_status(processing_queue, f"{file_number}. Video frame generation")
    all_video_frames_paths = []
    how_many_frames = len(frame_list_paths)
    for index_frame in range(how_many_frames-1):

        start_timer = timer()

        frame_base_name = os_path_splitext(frame_list_paths[index_frame])[0]
        frame_1_name = frame_list_paths[index_frame]
        frame_2_name = frame_list_paths[index_frame + 1]

        # Read frames and resize if needed
        frame_1, frame_2 = resize_frames(
            frame_1 = image_read(frame_list_paths[index_frame]), 
            frame_2 = image_read(frame_list_paths[index_frame + 1]), 
            resize_factor = resize_factor, 
            target_width  = target_width, 
            target_height = target_height
            )

        # Generate frame/s beetween frame 1 and 2
        all_video_frames_paths = AI_generate_frames(
            AI_model,
            backend,
            frame_1, 
            frame_2, 
            frame_1_name,
            frame_2_name,
            frame_base_name,
            all_video_frames_paths, 
            selected_image_extension, 
            fluidification_factor
        )

        # Update process status every 4 frames
        update_process_status_videos(processing_queue, file_number, start_timer, index_frame, how_many_frames)
            
    # Delete duplicates frames in list
    all_video_frames_paths = list(dict.fromkeys(all_video_frames_paths))

    # Video reconstruction
    write_process_status(processing_queue, f"{file_number}. Processing fluidified video")  
    video_reconstruction_by_frames(
        video_path,
        audio_path,
        all_video_frames_paths, 
        selected_AI_model,
        fluidification_factor, 
        slowmotion, 
        resize_factor, 
        cpu_number, 
        selected_video_extension
        )

    if not selected_save_frames:
        remove_dir(target_directory)



# GUI utils function ---------------------------

def opengithub() -> None:   
    open_browser(githubme, new=1)

def opentelegram() -> None: 
    open_browser(telegramme, new=1)

def user_input_checks() -> None:
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

def check_if_file_is_video(
        file: str
        ) -> bool:
    
    return any(video_extension in file for video_extension in supported_video_extensions)

def check_supported_selected_files(
        uploaded_file_list: list
        ) -> list:
    
    return [file for file in uploaded_file_list if any(supported_extension in file for supported_extension in supported_file_extensions)]

def extract_video_info(
        video_file: str
        ) -> tuple:
    cap          = opencv_VideoCapture(video_file)
    width        = round(cap.get(CAP_PROP_FRAME_WIDTH))
    height       = round(cap.get(CAP_PROP_FRAME_HEIGHT))
    num_frames   = int(cap.get(CAP_PROP_FRAME_COUNT))
    frame_rate   = cap.get(CAP_PROP_FPS)
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

    video_label = f"VIDEO • {video_name} • {width}x{height} • {minutes}m:{round(seconds)}s • {num_frames}frames • {round(frame_rate, 2)}fps"

    ctkimage = CTkImage(pillow_image_open("temp.jpg"), size = (25, 25))
    
    return video_label, ctkimage

def show_error_message(
        exception: str
        ) -> None:
    
    messageBox_title = "Upscale error"
    messageBox_text  = str(exception) + "\n\n" + "Please report the error on Github/Telegram"
    CTkMessageBox(text = messageBox_text, title = messageBox_title, type = "error")



# GUI select from menus functions ---------------------------

def select_AI_from_menu(
        selected_option: str
        ) -> None:
    
    global selected_AI_model    
    selected_AI_model = selected_option

def select_framegeneration_option_from_menu(new_value: str):
    global selected_fluidity_option    
    selected_fluidity_option = new_value

def select_AI_device_from_menu(new_value: str):
    global selected_AI_device    

    for device in gpu_list:
        if device.name == new_value:
            selected_AI_device = device.index

def select_image_extension_from_menu(new_value: str):
    global selected_image_extension    
    selected_image_extension = new_value

def select_video_extension_from_menu(new_value: str):
    global selected_video_extension   
    selected_video_extension = new_value

def select_save_frame_from_menu(new_value: str):
    global selected_save_frames
    if new_value == 'Enabled':
        selected_save_frames = True
    elif new_value == 'Disabled':
        selected_save_frames = False



# GUI info functions ---------------------------

def open_info_AI_model():
    option_list = [
        "\n RIFE_4.13\n" + 
        "   • The complete RIFE AI model\n" + 
        "   • Excellent frame generation quality\n" + 
        "   • Recommended GPUs with VRAM >= 4GB\n",

        "\n RIFE_4.13_Lite\n" + 
        "   • Lightweight version of RIFE AI model\n" +
        "   • High frame generation quality\n" +
        "   • 25% faster than full model\n" + 
        "   • Use less GPU VRAM memory\n" +
        "   • Recommended for GPUs with VRAM < 4GB \n",
    ]
    
    CTkMessageBox(messageType = "info",
                    title = "AI model", 
                    subtitle = " This widget allows to choose between different RIFE models",
                    default_value = "RIFE_4.13",
                    option_list = option_list)

def open_info_fluidity_option():
    option_list = [
        "\n FLUIDIFY\n" + 
        "   • x2 ( doubles video framerate • 30fps => 60fps )\n" + 
        "   • x4 ( quadruples video framerate • 30fps => 120fps )\n" + 
        "   • x8 ( octuplicate video framerate • 30fps => 240fps )\n",

        "\n SLOWMOTION (no audio)\n" + 
        "   • x2-slowmotion ( slowmotion effect by a factor of 2 )\n" +
        "   • x4-slowmotion ( slowmotion effect by a factor of 4 )\n" +
        "   • x8-slowmotion ( slowmotion effect by a factor of 8 )\n"
    ]
    
    CTkMessageBox(messageType = "info",
                    title = "AI frame generation", 
                    subtitle = " This widget allows to choose between different AI frame generation option",
                    default_value = "x2",
                    option_list = option_list)

def open_info_gpu():
    option_list = [
        " Keep in mind that the more powerful the chosen gpu is, the faster the upscaling will be",
        " For optimal results, it is essential to regularly update your GPU drivers"
    ]

    CTkMessageBox(messageType = "info",
                    title = "GPU", 
                    subtitle = "This widget allows to select the GPU for AI processing",
                    default_value = None,
                    option_list = option_list)

def open_info_AI_output():
    option_list = [
        "\n PNG\n" + 
        "   • very good quality\n" + 
        "   • slow and heavy file\n" + 
        "   • supports transparent images\n",

        "\n JPG\n" + 
        "   • good quality\n" +
        "   • fast and lightweight file\n",

        "\n BMP\n" + 
        "   • highest quality\n" +
        "   • slow and heavy file\n",

        "\n TIFF\n" + 
        "   • highest quality\n" +
        "   • very slow and heavy file\n",
    ]

    CTkMessageBox(messageType = "info",
                    title = "AI output", 
                    subtitle = "This widget allows to choose the extension of generated frames",
                    default_value = ".png",
                    option_list = option_list)

def open_info_input_resolution():
    option_list = [
        " A high value (>70%) will create high quality video but will be slower",
        " While a low value (<40%) will create good quality video but will much faster",

        " \n For example, for a 1080p (1920x1080) video\n" + 
        " • Input resolution 25% => input to AI 270p (480x270)\n" +
        " • Input resolution 50% => input to AI 540p (960x540)\n" + 
        " • Input resolution 75% => input to AI 810p (1440x810)\n" + 
        " • Input resolution 100% => input to AI 1080p (1920x1080) \n",
    ]

    CTkMessageBox(messageType = "info",
                    title = "Input resolution %", 
                    subtitle = "This widget allows to choose the resolution input to the AI",
                    default_value = "60%",
                    option_list = option_list)

def open_info_video_extension():
    option_list = [
        "\n MP4 (x264)\n" + 
        "   • produces well compressed video using x264 codec\n",

        "\n MP4 (x265)\n" + 
        "   • produces well compressed video using x265 codec\n",

        "\n AVI\n" + 
        "   • produces the highest quality video\n" +
        "   • the video produced can also be of large size\n",

    ]


    CTkMessageBox(messageType = "info",
                title = "Video output", 
                subtitle = "This widget allows to choose the extension of the upscaled video",
                default_value = ".mp4 (x264)",
                option_list = option_list)

def open_info_save_frames():
    option_list = [
        "\n ENABLED \n FluidFrames.RIFE will create \n   • the fluidified video \n   • a folder containing all original and interpolated frames \n",
        "\n DISABLED \n FluidFrames.RIFE will create only the fluidified video \n"
    ]

    CTkMessageBox(messageType = "info",
                    title = "Save frames",
                    subtitle = "This widget allows to choose to save frames generated by the AI",
                    default_value = "Enabled",
                    option_list = option_list)

def open_info_cpu():
    option_list = [
        " When possible the app will use the number of cpus selected",
        " Currently this value is only used for the video encoding phase",
    ]

    default_cpus = str(int(os_cpu_count()/2))

    CTkMessageBox(messageType = "info",
                    title = "Cpu number",
                    subtitle = "This widget allows to choose how many cpus to devote to the app",
                    default_value = default_cpus,
                    option_list = option_list)



# GUI place functions ---------------------------

def open_files_action():
    info_message.set("Selecting files")

    uploaded_files_list    = list(filedialog.askopenfilenames())
    uploaded_files_counter = len(uploaded_files_list)

    supported_files_list    = check_supported_selected_files(uploaded_files_list)
    supported_files_counter = len(supported_files_list)
    
    print("> Uploaded files: " + str(uploaded_files_counter) + " => Supported files: " + str(supported_files_counter))

    if supported_files_counter > 0:

        global scrollable_frame_file_list
        scrollable_frame_file_list = ScrollableImagesTextFrame(
            master = window, 
            selected_file_list = supported_files_list,
            fg_color = dark_color, 
            bg_color = dark_color
        )
        
        scrollable_frame_file_list.place(
            relx = 0.0, 
            rely = 0.0, 
            relwidth  = 1.0, 
            relheight = 0.45
        )
        
        info_message.set("Ready")

    else: 
        info_message.set("Not supported files :(")

def place_github_button():
    git_button = CTkButton(master      = window, 
                            command    = opengithub,
                            image      = logo_git,
                            width         = 30,
                            height        = 30,
                            border_width  = 1,
                            fg_color      = "transparent",
                            text_color    = "#C0C0C0",
                            border_color  = "#404040",
                            anchor        = "center",                           
                            text          = "", 
                            font          = bold11)
    
    git_button.place(relx = 0.045, rely = 0.87, anchor = "center")

def place_telegram_button():
    telegram_button = CTkButton(master     = window, 
                                image      = logo_telegram,
                                command    = opentelegram,
                                width         = 30,
                                height        = 30,
                                border_width  = 1,
                                fg_color      = "transparent",
                                text_color    = "#C0C0C0",
                                border_color  = "#404040",
                                anchor        = "center",                           
                                text          = "", 
                                font          = bold11)
    telegram_button.place(relx = 0.045, rely = 0.93, anchor = "center")
 
def place_loadFile_section():
    up_background = CTkLabel(master  = window, 
                            text     = "",
                            fg_color = dark_color,
                            font     = bold12,
                            anchor   = "w")
    
    up_background.place(relx = 0.0, 
                        rely = 0.0, 
                        relwidth  = 1.0,  
                        relheight = 0.45)

    text_drop = """ •  SUPPORTED FILES  •

VIDEOS • mp4 webm mkv flv gif avi mov mpg qt 3gp"""

    input_file_text = CTkLabel(master      = window, 
                                text       = text_drop,
                                fg_color   = dark_color,
                                bg_color   = dark_color,
                                text_color = "#C0C0C0",
                                width      = 300,
                                height     = 150,
                                font       = bold12,
                                anchor     = "center")
    
    input_file_button = CTkButton(master = window,
                                command  = open_files_action, 
                                text     = "SELECT FILES",
                                width      = 140,
                                height     = 30,
                                font       = bold11,
                                border_width = 1,
                                fg_color     = "#282828",
                                text_color   = "#E0E0E0",
                                border_color = "#0096FF")

    input_file_text.place(relx = 0.5, rely = 0.20,  anchor = "center")
    input_file_button.place(relx = 0.5, rely = 0.35, anchor = "center")

def place_app_name():
    app_name_label = CTkLabel(master     = window, 
                              text       = app_name + " " + version,
                              text_color = "#F08080",
                              font       = bold19,
                              anchor     = "w")
    
    app_name_label.place(relx = column0_x, rely = row0_y - 0.03, anchor = "center")

    subtitle_app_name_label = CTkLabel(master  = window, 
                                    text       = second_title,
                                    text_color = "#0096FF",
                                    font       = bold18,
                                    anchor     = "w")
    
    subtitle_app_name_label.place(relx = column0_x, rely = row0_y + 0.01, anchor = "center")

def place_AI_menu():

    AI_menu_button = create_info_button(open_info_AI_model, "AI model")
    AI_menu        = create_option_menu(select_AI_from_menu, AI_models_list)

    AI_menu_button.place(relx = column0_x, rely = row1_y - 0.053, anchor = "center")
    AI_menu.place(relx = column0_x, rely = row1_y, anchor = "center")

def place_framegeneration_option_menu():

    fluidity_option_button = create_info_button(open_info_fluidity_option, "AI frame generation")
    fluidity_option_menu   = create_option_menu(select_framegeneration_option_from_menu, fluidity_options_list)

    fluidity_option_button.place(relx = column0_x, rely = row2_y - 0.05, anchor = "center")
    fluidity_option_menu.place(relx = column0_x, rely = row2_y, anchor = "center")

def place_gpu_menu():

    gpu_button = create_info_button(open_info_gpu, "GPU")
    gpu_menu   = create_option_menu(select_AI_device_from_menu, gpu_list_names)
    
    gpu_button.place(relx = column0_x, rely = row3_y - 0.053, anchor = "center")
    gpu_menu.place(relx = column0_x, rely  = row3_y, anchor = "center")

def place_AI_output_menu():

    file_extension_button = create_info_button(open_info_AI_output, "AI output")
    file_extension_menu   = create_option_menu(select_image_extension_from_menu, image_extension_list)
    
    file_extension_button.place(relx = column1_x, rely = row0_y - 0.053, anchor = "center")
    file_extension_menu.place(relx = column1_x, rely = row0_y, anchor = "center")

def place_video_extension_menu():

    video_extension_button = create_info_button(open_info_video_extension, "Video output")
    video_extension_menu   = create_option_menu(select_video_extension_from_menu, video_extension_list)
    
    video_extension_button.place(relx = column1_x, rely = row1_y - 0.053, anchor = "center")
    video_extension_menu.place(relx = column1_x, rely = row1_y, anchor = "center")

def place_save_frames_menu():

    save_frames_button = create_info_button(open_info_save_frames, "Save frames")
    save_frames_menu   = create_option_menu(select_save_frame_from_menu, save_frames_list)
    
    save_frames_button.place(relx = column1_x, rely = row2_y - 0.053, anchor = "center")
    save_frames_menu.place(relx = column1_x, rely = row2_y, anchor = "center")

def place_input_resolution_textbox():

    resize_factor_button  = create_info_button(open_info_input_resolution, "Input resolution %")
    resize_factor_textbox = create_text_box(selected_resize_factor) 

    resize_factor_button.place(relx = column1_x, rely = row3_y - 0.053, anchor = "center")
    resize_factor_textbox.place(relx = column1_x, rely = row3_y, anchor = "center")

def place_cpu_textbox():

    cpu_button  = create_info_button(open_info_cpu, "CPU number")
    cpu_textbox = create_text_box(selected_cpu_number)

    cpu_button.place(relx = column2_x, rely = row0_y - 0.053, anchor = "center")
    cpu_textbox.place(relx = column2_x, rely = row0_y, anchor = "center")

def place_message_label():

    message_label = CTkLabel(master  = window, 
                            textvariable = info_message,
                            height       = 25,
                            font         = bold11,
                            fg_color     = "#ffbf00",
                            text_color   = "#000000",
                            anchor       = "center",
                            corner_radius = 12)
    message_label.place(relx = column2_x, rely = row2_y, anchor = "center")

def place_fluidify_button(): 

    fluidify_button = CTkButton(master     = window,
                                command    = fludify_button_command, 
                                image      = play_icon,
                                text       = "FLUIDIFY",
                                width      = 140,
                                height     = 30,
                                font       = bold11,
                                border_width = 1,
                                fg_color     = "#282828",
                                text_color   = "#E0E0E0",
                                border_color = "#0096FF")
    fluidify_button.place(relx = column2_x, rely = row3_y, anchor = "center")

def place_stop_button(): 

    stop_button = CTkButton(master     = window,
                            command    = stop_button_command, 
                            image      = stop_icon,
                            text       = "STOP",
                            width      = 140,
                            height     = 30,
                            font       = bold11,
                            border_width = 1,
                            fg_color     = "#282828",
                            text_color   = "#E0E0E0",
                            border_color = "#EC1D1D")
    stop_button.place(relx = column2_x, rely = row3_y, anchor = "center")

   

# Main functions ---------------------------

class Gpu:
    def __init__(
            self, 
            index: int, 
            name: str, 
            memory: any, 
            float64: bool
            ) -> None:
        
        self.index: int    = index
        self.name: str     = name
        self.memory: any   = memory
        self.float64: bool = float64

def on_app_close():
    window.grab_release()
    window.destroy()
    stop_fluidify_process()

class App():
    def __init__(self, window):

        self.toplevel_window = None

        window.title('')
        width        = 675
        height       = 675
        window.geometry("675x675")
        window.minsize(width, height)
        window.iconbitmap(find_by_relative_path("Assets" + os_separator + "logo.ico"))

        window.protocol("WM_DELETE_WINDOW", on_app_close)

        place_app_name()
        place_github_button()
        place_telegram_button()

        place_AI_menu()
        place_framegeneration_option_menu()
        place_gpu_menu()

        place_AI_output_menu()
        place_video_extension_menu()
        place_save_frames_menu()
        place_input_resolution_textbox()

        place_cpu_textbox()
        place_message_label()
        place_fluidify_button()

        place_loadFile_section()

if __name__ == "__main__":
    multiprocessing_freeze_support()

    processing_queue = multiprocessing_Queue(maxsize=1)

    set_appearance_mode("Dark")
    set_default_color_theme("dark-blue")

    gpu_list_names = []
    gpu_list = []
    how_many_gpus = directml_device_count()
    for index in range(how_many_gpus): 
        gpu_index   = index
        gpu_name    = directml_device_name(index)
        gpu_memory  = directml_gpu_memory(index)
        gpu_float64 = directml_has_float64_support(index)
        gpu = Gpu(gpu_index, gpu_name, gpu_memory, gpu_float64)

        gpu_list.append(gpu)
        gpu_list_names.append(gpu_name)

    window = CTk() 

    global selected_file_list
    global selected_AI_model
    global selected_fluidity_option
    global selected_AI_device 
    global selected_save_frames
    global resize_factor
    global cpu_number

    global selected_image_extension
    global selected_video_extension

    selected_file_list = []

    selected_AI_device = 0
    selected_AI_model = AI_models_list[0]
    selected_fluidity_option = fluidity_options_list[0]
    selected_image_extension = image_extension_list[0]
    selected_video_extension = video_extension_list[0]

    if save_frames_list[0] == "Disabled": 
        selected_save_frames = False
    elif save_frames_list[0] == "Enabled":  
        selected_save_frames = True

    info_message            = StringVar()
    selected_resize_factor  = StringVar()
    selected_cpu_number     = StringVar()

    info_message.set("Hi :)")

    selected_resize_factor.set("60")
    cpu_count = str(int(os_cpu_count()/2))
    selected_cpu_number.set(cpu_count)

    font   = "Segoe UI"    
    bold8  = CTkFont(family = font, size = 8, weight = "bold")
    bold9  = CTkFont(family = font, size = 9, weight = "bold")
    bold10 = CTkFont(family = font, size = 10, weight = "bold")
    bold11 = CTkFont(family = font, size = 11, weight = "bold")
    bold12 = CTkFont(family = font, size = 12, weight = "bold")
    bold13 = CTkFont(family = font, size = 13, weight = "bold")
    bold14 = CTkFont(family = font, size = 14, weight = "bold")
    bold16 = CTkFont(family = font, size = 16, weight = "bold")
    bold17 = CTkFont(family = font, size = 17, weight = "bold")
    bold18 = CTkFont(family = font, size = 18, weight = "bold")
    bold19 = CTkFont(family = font, size = 19, weight = "bold")
    bold20 = CTkFont(family = font, size = 20, weight = "bold")
    bold21 = CTkFont(family = font, size = 21, weight = "bold")
    bold22 = CTkFont(family = font, size = 22, weight = "bold")
    bold23 = CTkFont(family = font, size = 23, weight = "bold")
    bold24 = CTkFont(family = font, size = 24, weight = "bold")


    # Images
    play_icon      = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}upscale_icon.png")),   size=(15, 15))
    logo_git       = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}github_logo.png")),    size=(15, 15))
    logo_telegram  = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}telegram_logo.png")),  size=(15, 15))
    stop_icon      = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}stop_icon.png")),      size=(15, 15))
    upscale_icon   = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}upscale_icon.png")),   size=(15, 15))
    clear_icon     = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}clear_icon.png")),     size=(15, 15))
    info_icon      = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}info_icon.png")),      size=(16, 16))

    app = App(window)
    window.update()
    window.mainloop()