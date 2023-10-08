
# Standard library imports
import sys
from shutil          import rmtree, copytree
from time            import sleep
from threading       import Thread
from timeit          import default_timer as timer
from subprocess      import run  as subprocess_run
from webbrowser      import open as open_browser

from multiprocessing import (
    Process, 
    freeze_support
)

from os import (
    sep         as os_separator,
    devnull     as os_devnull,
    chmod       as os_chmod,
    cpu_count   as os_cpu_count,
    makedirs    as os_makedirs,
    remove      as os_remove,
    listdir     as os_listdir
)

from os.path import (
    basename as os_path_basename,
    dirname  as os_path_dirname,
    abspath  as os_path_abspath,
    join     as os_path_join,
    exists   as os_path_exists,
    splitext as os_path_splitext
)


# Third-party library imports
from PIL.Image        import open as pillow_image_open
from moviepy.editor   import VideoFileClip
from moviepy.video.io import ImageSequenceClip

from torch.nn.functional import interpolate as torch_nn_interpolate

from torch import (
    device          as torch_device,
    no_grad         as torch_no_grad,
    linspace        as torch_linspace,
    cat             as torch_cat,
    load            as torch_load,
    ones            as torch_ones,
    set_num_threads as torch_set_num_threads,
    is_tensor       as torch_is_tensor,
    sigmoid         as torch_sigmoid,
    tensor          as torch_tensor,
)

from torch_directml import (
     device       as directml_device,
     device_count as directml_device_count,
     device_name  as directml_device_name
)

from torch.nn.functional import (
    grid_sample as torch_nn_functional_grid_sample,
    pad         as torch_nn_functional_pad
)

from torch.nn import (
    ConvTranspose2d,
    Sequential,
    BatchNorm2d,
    Conv2d,
    Module,
    Parameter,
    PixelShuffle,
    LeakyReLU,
)

from cv2 import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    IMREAD_UNCHANGED,
    INTER_AREA,
    INTER_LINEAR,
    VideoCapture as opencv_VideoCapture,
    imread       as opencv_imread,
    imwrite      as opencv_imwrite,
    resize       as opencv_resize,
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
version      = "2.7"

githubme   = "https://github.com/Djdefrag/FluidFrames.RIFE"
telegramme = "https://linktr.ee/j3ngystudio"

half_precision           = False 
fluidity_options_list    = [
                            'x2', 
                            'x4', 
                            'x8',
                            'x2-slowmotion', 
                            'x4-slowmotion',
                            'x8-slowmotion'
                            ]

image_extension_list  = [ '.jpg', '.png', '.bmp', '.tiff' ]
video_extension_list  = [ '.mp4', '.avi' ]
save_frames_list      = [ 'Enabled', 'Disabled' ]

device_list_names    = []
device_list          = []
resize_algorithm     = INTER_AREA

offset_y_options = 0.125
row0_y           = 0.56
row1_y           = row0_y + offset_y_options
row2_y           = row1_y + offset_y_options
row3_y           = row2_y + offset_y_options

offset_x_options = 0.28
column1_x        = 0.5
column0_x        = column1_x - offset_x_options
column2_x        = column1_x + offset_x_options

dark_color     = "#080808"

log_file_path  = f"{app_name}.log"
temp_dir       = f"{app_name}_temp"
audio_path     = f"{app_name}_temp{os_separator}audio.mp3"
frame_sequence = f"{app_name}_temp{os_separator}frame_%01d.jpg"

if sys.stdout is None: sys.stdout = open(os_devnull, "w")
if sys.stderr is None: sys.stderr = open(os_devnull, "w")



# ------------------ AI ------------------

def load_AI_model(backend, half_precision):
    def convert(param):
        return {
            k.replace("module.", ""): v
            for k, v in param.items()
            if "module." in k
        }

    update_process_status(f"Loading AI model")

    model_path = find_by_relative_path(f"AI{os_separator}RIFE_v4.6.pkl")

    with torch_no_grad():
        model            = IFNet_46(backend)
        pretrained_model = torch_load(model_path, map_location = torch_device('cpu'))
        model.load_state_dict(convert(pretrained_model))
        model.eval()

        if half_precision: model = model.half()
        model.to(backend, non_blocking = True)

    return model

def warp(tenInput, tenFlow, backend):
    backwarp_tenGrid = {}

    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch_linspace(-1.0, 1.0, tenFlow.shape[3], device = backend).view(1, 1, 1, 
                                                              tenFlow.shape[3]).expand(tenFlow.shape[0], -1,  tenFlow.shape[2], -1)
        tenVertical   = torch_linspace(-1.0, 1.0, tenFlow.shape[2], device = backend).view(1, 1, 
                                                            tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch_cat([tenHorizontal, tenVertical], 1).to(backend, non_blocking = True)

    tenFlow = torch_cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch_nn_functional_grid_sample(input = tenInput, grid = g, mode = 'bilinear', padding_mode = 'border', align_corners = True)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return Sequential(
        Conv2d(in_planes, 
                  out_planes, 
                  kernel_size=kernel_size, 
                  stride=stride,
                  padding=padding, 
                  dilation=dilation, bias=True),        
        LeakyReLU(0.2, True)
    )

def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return Sequential(
        Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        BatchNorm2d(out_planes),
        LeakyReLU(0.2, True)
    )
    
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
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
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
            x = torch_cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp  = self.lastconv(feat)
        tmp  = torch_nn_interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask
     
class IFNet_46(Module):
    def __init__(self, backend):
        super(IFNet_46, self).__init__()
        self.block0 = IFBlock(7,   c=192)
        self.block1 = IFBlock(8+4, c=128)
        self.block2 = IFBlock(8+4, c=96)
        self.block3 = IFBlock(8+4, c=64)

        self.backend = backend

    def forward(self, x, timestep=0.5, scale_list=[8, 4, 2, 1]):
        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]

        if not torch_is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])

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
                flow, mask = block[i](torch_cat((img0[:, :3], img1[:, :3], timestep), 1), None, scale=scale_list[i])
            else:
                f0, m0 = block[i](torch_cat((warped_img0[:, :3], warped_img1[:, :3], timestep, mask), 1), flow, scale=scale_list[i])                
                flow = flow + f0
                mask = mask + m0

            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2], self.backend)
            warped_img1 = warp(img1, flow[:, 2:4], self.backend)
            merged.append((warped_img0, warped_img1))
        mask_list[3] = torch_sigmoid(mask_list[3])
        merged[3] = merged[3][0] * mask_list[3] + merged[3][1] * (1 - mask_list[3])

        return flow_list, mask_list[3], merged
    
    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        imgs       = torch_cat((img0, img1), 1)
        scale_list = [8/scale, 4/scale, 2/scale, 1/scale]
        flow, mask, merged = self(imgs, timestep, scale_list)
        return merged[3]

def frames_to_tensors(frame_1, 
                      frame_2, 
                      backend, 
                      half_precision):

    if half_precision:
        img_1 = (torch_tensor(frame_1.transpose(2, 0, 1)).half().to(backend, non_blocking = True) / 255.).unsqueeze(0)
        img_2 = (torch_tensor(frame_2.transpose(2, 0, 1)).half().to(backend, non_blocking = True) / 255.).unsqueeze(0)
    else:
        img_1 = (torch_tensor(frame_1.transpose(2, 0, 1)).to(backend, non_blocking = True) / 255.).unsqueeze(0)
        img_2 = (torch_tensor(frame_2.transpose(2, 0, 1)).to(backend, non_blocking = True) / 255.).unsqueeze(0)

    _ , _ , h, w = img_1.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)

    img_1 = torch_nn_functional_pad(img_1, padding)
    img_2 = torch_nn_functional_pad(img_2, padding)

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

    with torch_no_grad():

        first_frame_tensor, last_frame_tensor, h, w = frames_to_tensors(frame1, frame2, backend, half_precision)

        if frames_to_generate == 1: 
            # fluidification x2
            middle_frame_name = frame_base_name + '_middle' + selected_output_file_extension

            middle_frame_tensor = AI_model.inference(first_frame_tensor, last_frame_tensor)
            middle_frame        = tensor_to_frame(middle_frame_tensor, h, w)
            
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

        elif frames_to_generate == 7: 
            # fluidification x8
            frame_1_1_name      = frame_base_name + '_.1' + selected_output_file_extension
            frame_1_2_name      = frame_base_name + '_.2' + selected_output_file_extension
            frame_1_3_name      = frame_base_name + '_.3' + selected_output_file_extension
            frame_1_4_name      = frame_base_name + '_.4' + selected_output_file_extension
            frame_1_5_name      = frame_base_name + '_.5' + selected_output_file_extension
            frame_1_6_name      = frame_base_name + '_.6' + selected_output_file_extension
            frame_1_7_name      = frame_base_name + '_.7' + selected_output_file_extension

            frame_1_4_tensor   = AI_model.inference(first_frame_tensor, last_frame_tensor)
            frame_1_2_tensor   = AI_model.inference(first_frame_tensor, frame_1_4_tensor)
            frame_1_1_tensor   = AI_model.inference(first_frame_tensor, frame_1_2_tensor)
            frame_1_3_tensor   = AI_model.inference(frame_1_2_tensor, frame_1_4_tensor)

            frame_1_6_tensor   = AI_model.inference(frame_1_4_tensor, last_frame_tensor)
            frame_1_5_tensor   = AI_model.inference(frame_1_4_tensor, frame_1_6_tensor)
            frame_1_7_tensor   = AI_model.inference(frame_1_6_tensor, last_frame_tensor)


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

            all_files_list.append(frame_1_name)

            all_files_list.append(frame_1_1_name)
            all_files_list.append(frame_1_2_name)
            all_files_list.append(frame_1_3_name)
            all_files_list.append(frame_1_4_name)
            all_files_list.append(frame_1_5_name)
            all_files_list.append(frame_1_6_name)
            all_files_list.append(frame_1_7_name)

            all_files_list.append(frame_2_name)

        return all_files_list



# Classes and utils -------------------

class Gpu:
    def __init__(self, index, name):
        self.name   = name
        self.index  = index

for index in range(directml_device_count()): 
    gpu = Gpu(index = index, name = directml_device_name(index))
    device_list.append(gpu)
    device_list_names.append(gpu.name)

class ScrollableImagesTextFrame(CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.label_list  = []
        self.button_list = []
        self.file_list   = []

    def get_selected_file_list(self): return self.file_list

    def add_clean_button(self):
        label  = CTkLabel(self, text = "")
        button = CTkButton(self, 
                            image        = clear_icon,
                            font         = bold11,
                            text         = "CLEAN", 
                            compound     = "left",
                            width        = 100, 
                            height       = 28,
                            border_width = 1,
                            fg_color     = "#282828",
                            text_color   = "#E0E0E0",
                            border_color = "#0096FF")

        button.configure(command=lambda: self.clean_all_items())
        button.grid(row = len(self.button_list), column=1, pady=(0, 10), padx = 5)
        self.label_list.append(label)
        self.button_list.append(button)

    def add_item(self, text_to_show, file_element, image = None):
        label = CTkLabel(self, 
                        text          = text_to_show,
                        font          = bold11,
                        image         = image, 
                        text_color    = "#E0E0E0",
                        compound      = "left", 
                        padx          = 10,
                        pady          = 5,
                        anchor        = "center")
                        
        label.grid(row = len(self.label_list), column = 0, pady = (3, 3), padx = (3, 3), sticky = "w")
        self.label_list.append(label)
        self.file_list.append(file_element)    

    def clean_all_items(self):
        self.label_list  = []
        self.button_list = []
        self.file_list   = []
        self.destroy()
        place_loadFile_section()

class CTkMessageBox(CTkToplevel):
    def __init__(self,
                 title: str = "CTkDialog",
                 text: str = "CTkDialog",
                 type: str = "info"):

        super().__init__()

        self._running: bool = False
        self._title = title
        self._text = text
        self.type = type

        self.title('')
        self.lift()                          # lift window on top
        self.attributes("-topmost", True)    # stay on top
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.after(10, self._create_widgets)  # create widgets with slight delay, to avoid white flickering of background
        self.resizable(False, False)
        self.grab_set()                       # make other windows not clickable

    def _create_widgets(self):

        self.grid_columnconfigure((0, 1), weight=1)
        self.rowconfigure(0, weight=1)

        self._text = '\n' + self._text +'\n'

        if self.type == "info":
            color_for_messagebox_title = "#0096FF"
        elif self.type == "error":
            color_for_messagebox_title = "#ff1a1a"


        self._titleLabel = CTkLabel(master  = self,
                                    width      = 500,
                                    anchor     = 'w',
                                    justify    = "left",
                                    fg_color   = "transparent",
                                    text_color = color_for_messagebox_title,
                                    font       = bold24,
                                    text       = self._title)
        
        self._titleLabel.grid(row=0, column=0, columnspan=2, padx=30, pady=20, sticky="ew")

        self._label = CTkLabel(master = self,
                                width      = 550,
                                wraplength = 550,
                                corner_radius = 10,
                                anchor     = 'w',
                                justify    = "left",
                                text_color = "#C0C0C0",
                                bg_color   = "transparent",
                                fg_color   = "#303030",
                                font       = bold12,
                                text       = self._text)
        
        self._label.grid(row=1, column=0, columnspan=2, padx=30, pady=5, sticky="ew")

        self._ok_button = CTkButton(master  = self,
                                    command = self._ok_event,
                                    text    = 'OK',
                                    width   = 125,
                                    font         = bold11,
                                    border_width = 1,
                                    fg_color     = "#282828",
                                    text_color   = "#E0E0E0",
                                    border_color = "#0096FF")
        
        self._ok_button.grid(row=2, column=1, columnspan=1, padx=(10, 20), pady=(10, 20), sticky="e")

    def _ok_event(self, event = None):
        self.grab_release()
        self.destroy()

    def _on_closing(self):
        self.grab_release()
        self.destroy()

def create_info_button(command, text):
    return CTkButton(master  = window, 
                    command  = command,
                    text          = text,
                    fg_color      = "transparent",
                    text_color    = "#C0C0C0",
                    anchor        = "w",
                    height        = 23,
                    width         = 150,
                    corner_radius = 12,
                    font          = bold12,
                    image         = info_icon)

def create_option_menu(command, values):
    return CTkOptionMenu(master = window, 
                        command = command,
                        values  = values,
                        width              = 150,
                        height             = 31,
                        anchor             = "center",
                        dropdown_font      = bold10,
                        font               = bold11,
                        text_color         = "#C0C0C0",
                        fg_color           = "#000000",
                        button_color       = "#000000",
                        button_hover_color = "#000000",
                        dropdown_fg_color  = "#000000")

def create_text_box(textvariable):
    return CTkEntry(master        = window, 
                    textvariable  = textvariable,
                    border_width  = 1,
                    width         = 150,
                    height        = 30,
                    font          = bold10,
                    justify       = "center",
                    fg_color      = "#000000",
                    border_color  = "#404040")

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



# Utils functions ------------------------

def opengithub():   
    open_browser(githubme, new=1)

def opentelegram(): 
    open_browser(telegramme, new=1)

def find_by_relative_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os_path_dirname(os_path_abspath(__file__)))
    return os_path_join(base_path, relative_path)

def write_in_log_file(text_to_insert):
    with open(log_file_path,'w') as log_file: 
        os_chmod(log_file_path, 0o777)
        log_file.write(text_to_insert) 
    log_file.close()

def read_log_file():
    with open(log_file_path,'r') as log_file: 
        os_chmod(log_file_path, 0o777)
        step = log_file.readline()
    log_file.close()
    return step

def image_write(file_path, file_data): 
    opencv_imwrite(file_path, file_data)

def image_read(file_path, flags = IMREAD_UNCHANGED): 
    return opencv_imread(file_path, flags)

def remove_file(name_file): 
    if os_path_exists(name_file): 
        os_remove(name_file)

def remove_dir(name_dir):
    if os_path_exists(name_dir): 
        rmtree(name_dir)

def create_dir(name_dir):
    if os_path_exists(name_dir):     
        rmtree(name_dir)
    if not os_path_exists(name_dir): 
        os_makedirs(name_dir, mode=0o777)

def prepare_output_video_filename(video_path, 
                                  fluidification_factor, 
                                  slowmotion, 
                                  resize_factor, 
                                  selected_video_output):
    
    result_video_path = os_path_splitext(video_path)[0]

    resize_percentage = str(int(resize_factor * 100)) + "%"
    
    if slowmotion: to_append = f"_RIFEx{str(fluidification_factor)}_slowmo_{resize_percentage}{selected_video_output}"
    else:          to_append = f"_RIFEx{str(fluidification_factor)}_{resize_percentage}{selected_video_output}"

    result_video_path = result_video_path + to_append

    return result_video_path

def prepare_output_directory_name(video_path, 
                                  fluidification_factor, 
                                  resize_factor):
    
    result_video_path = os_path_splitext(video_path)[0]

    resize_percentage = str(int(resize_factor * 100)) + "%"
    
    to_append = f"_RIFEx{str(fluidification_factor)}_{resize_percentage}"

    result_video_path = result_video_path + to_append

    return result_video_path

def resize_frames(frame_1, frame_2, target_width, target_height):
    frame_1_resized = opencv_resize(frame_1, (target_width, target_height), interpolation = resize_algorithm)    
    frame_2_resized = opencv_resize(frame_2, (target_width, target_height), interpolation = resize_algorithm)    

    return frame_1_resized, frame_2_resized

def extract_video_frames_and_audio(video_path, file_number):
    video_frames_list = []
    cap          = opencv_VideoCapture(video_path)
    frame_rate   = cap.get(CAP_PROP_FPS)
    cap.release()

    video_file_clip = VideoFileClip(video_path)
    
    # Extract audio
    try: 
        update_process_status(f"{file_number}. Extracting video audio")
        video_file_clip.audio.write_audiofile(audio_path, verbose = False, logger = None)
    except:
        pass

    # Extract frames
    update_process_status(f"{file_number}. Extracting video frames")
    video_frames_list = video_file_clip.write_images_sequence(frame_sequence, verbose = False, logger = None, fps = frame_rate)
    
    return video_frames_list

def video_reconstruction_by_frames(input_video_path, 
                                    all_files_list, 
                                    fluidification_factor,
                                    slowmotion,
                                    resize_factor,
                                    cpu_number,
                                    selected_video_extension):
    
    # Find original video FPS
    cap = opencv_VideoCapture(input_video_path)
    if slowmotion: frame_rate = cap.get(CAP_PROP_FPS)
    else:          frame_rate = cap.get(CAP_PROP_FPS) * fluidification_factor
    cap.release()

    # Choose the appropriate codec
    if selected_video_extension == '.mp4':   
        codec = 'libx264'
    elif selected_video_extension == '.avi': 
        codec = 'png'

    audio_file = app_name + "_temp" + os_separator + "audio.mp3"
    upscaled_video_path = prepare_output_video_filename(input_video_path, 
                                                        fluidification_factor, 
                                                        slowmotion, 
                                                        resize_factor, 
                                                        selected_video_extension)

    clip = ImageSequenceClip.ImageSequenceClip(all_files_list, fps = frame_rate)
    if os_path_exists(audio_file) and slowmotion != True:
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



# Core functions ------------------------

def remove_temp_files():
    remove_dir(app_name + "_temp")
    remove_file(app_name + ".log")

def stop_thread():
    stop = 1 + "x"

def stop_fluidify_process():
    global process_fluidify_orchestrator

    try:
        process_fluidify_orchestrator
    except:
        pass
    else:
        process_fluidify_orchestrator.terminate()
        process_fluidify_orchestrator.join()

def check_fluidify_steps():
    sleep(3)
    try:
        while True:
            actual_step = read_log_file()

            info_message.set(actual_step)

            if "All files completed! :)" in actual_step or "Fluidify stopped" in actual_step:
                stop_fluidify_process()
                remove_temp_files()
                stop_thread()
            elif "Error during fluidify process" in actual_step:
                info_message.set('Error during fluidify process :(')
                show_error_message(actual_step.replace("Error during fluidify process", ""))
                remove_temp_files()
                stop_thread()

            sleep(2)
    except:
        place_fluidify_button()

def update_process_status(actual_process_phase):
    print(f"{actual_process_phase}")
    write_in_log_file(actual_process_phase) 

def stop_button_command():
    stop_fluidify_process()
    write_in_log_file("Fluidify stopped") 

def fludify_button_command(): 
    global selected_file_list
    global selected_fluidity_option
    global selected_AI_device 
    global selected_image_extension
    global selected_video_extension
    global resize_factor
    global cpu_number
    global selected_save_frames

    global process_fluidify_orchestrator

    remove_file(app_name + ".log")
    
    if user_input_checks():
        info_message.set("Loading")
        write_in_log_file("Loading")

        print("=================================================")
        print(f"> Starting fluidify:")
        print(f"   Files to fluidify: {len(selected_file_list)}")
        print(f"   Selected fluidify option: {selected_fluidity_option}")
        print(f"   AI half precision: {half_precision}")
        print(f"   Selected GPU: {directml_device_name(selected_AI_device)}")
        print(f"   Selected image output extension: {selected_image_extension}")
        print(f"   Selected video output extension: {selected_video_extension}")
        print(f"   Resize factor: {int(resize_factor * 100)}%")
        print(f"   Cpu number: {cpu_number}")
        print(f"   Save frames: {selected_save_frames}")
        print("=================================================")

        place_stop_button()

        backend = torch_device(directml_device(selected_AI_device))

        process_fluidify_orchestrator = Process(
                                            target = fluidify_orchestrator,
                                            args   = (selected_file_list,
                                                    selected_fluidity_option,
                                                    backend, 
                                                    selected_image_extension,
                                                    selected_video_extension,
                                                    resize_factor,
                                                    cpu_number,
                                                    half_precision,
                                                    selected_save_frames))
        process_fluidify_orchestrator.start()

        thread_wait = Thread(target = check_fluidify_steps, daemon = True)
        thread_wait.start()

def check_fluidification_option(selected_fluidity_option):
    slowmotion = False
    fluidification_factor = 0

    if 'slowmotion' in selected_fluidity_option: slowmotion = True

    if   '2' in selected_fluidity_option: fluidification_factor = 2
    elif '4' in selected_fluidity_option: fluidification_factor = 4
    elif '8' in selected_fluidity_option: fluidification_factor = 8

    return fluidification_factor, slowmotion

def calculate_time_to_complete_video(start_timer, 
                                     end_timer, 
                                     how_many_frames, 
                                     index_frame):
    
    seconds_for_frame = round(end_timer - start_timer, 2)
    frames_left       = how_many_frames - (index_frame + 1)
    seconds_left      = seconds_for_frame * frames_left

    hours_left   = seconds_left // 3600
    minutes_left = (seconds_left % 3600) // 60
    seconds_left = round((seconds_left % 3600) % 60)

    time_left = ""

    if int(hours_left)   > 0: time_left = f"{int(hours_left):02d}h"
    if int(minutes_left) > 0: time_left = f"{time_left}{int(minutes_left):02d}m"
    if seconds_left      > 0: time_left = f"{time_left}{seconds_left:02d}s"

    return time_left     

def copy_files_from_temp_directory(file_number, destination_dir):
    update_process_status(f"{file_number}. Saving video frames")
    copytree(f"{app_name}_temp", destination_dir)

def fluidify_video(video_path, 
                   file_number,
                   AI_model, 
                   fluidification_factor, 
                   slowmotion, 
                   resize_factor, 
                   backend, 
                   selected_image_extension, 
                   selected_video_extension,
                   cpu_number, 
                   half_precision,
                   selected_save_frames):
        
    create_dir(f"{app_name}_temp")
    
    frame_list = extract_video_frames_and_audio(video_path, file_number)

    temp_frame    = image_read(frame_list[0])
    target_height = int(temp_frame.shape[0] * resize_factor)
    target_width  = int(temp_frame.shape[1] * resize_factor)   

    done_frames     = 0
    all_files_list  = []
    how_many_frames = len(frame_list)

    update_process_status(f"{file_number}. Fluidifying video")

    for index_frame, _ in enumerate(frame_list[:-1]):
        
        start_timer = timer()

        frame_1_name    = frame_list[index_frame]
        frame_2_name    = frame_list[index_frame + 1]
        frame_base_name = os_path_splitext(frame_1_name)[0]

        frame_1 = image_read(frame_list[index_frame])
        frame_2 = image_read(frame_list[index_frame + 1])

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

        if index_frame != 0 and (index_frame + 1) % 6 == 0: 
            end_timer        = timer()    
            percent_complete = (index_frame + 1) / how_many_frames * 100 
            time_left        = calculate_time_to_complete_video(start_timer, end_timer, how_many_frames, index_frame)
        
            update_process_status(f"{file_number}. Fluidifying video {percent_complete:.2f}% ({time_left})")
            
    all_files_list = list(dict.fromkeys(all_files_list)) # Remove duplicated frames from list

    if selected_save_frames:
        save_directory = prepare_output_directory_name(video_path, fluidification_factor, resize_factor)
        copy_files_from_temp_directory(file_number, save_directory)


    update_process_status(f"{file_number}. Processing fluidified video")
    video_reconstruction_by_frames(video_path, 
                                   all_files_list, 
                                   fluidification_factor, 
                                   slowmotion, 
                                   resize_factor, 
                                   cpu_number,
                                   selected_video_extension)

def fluidify_orchestrator(selected_file_list,
                        selected_fluidity_option,
                        backend, 
                        selected_image_extension,
                        selected_video_extension,
                        resize_factor,
                        cpu_number,
                        half_precision,
                        selected_save_frames):
    
    torch_set_num_threads(1)
    
    fluidification_factor, slowmotion = check_fluidification_option(selected_fluidity_option)

    try:
        AI_model = load_AI_model(backend, half_precision)

        for file_number, file_path in enumerate(selected_file_list, 0):
            file_number = file_number + 1
            fluidify_video(file_path, 
                           file_number,
                            AI_model,
                            fluidification_factor, 
                            slowmotion,
                            resize_factor, 
                            backend,
                            selected_image_extension, 
                            selected_video_extension,
                            cpu_number,
                            half_precision,
                            selected_save_frames)

        update_process_status(f"All files completed! :)")

    except Exception as exception:
        update_process_status(f"Error during fluidify process {str(exception)}")



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

    video_label = ( "VIDEO" + " • " + video_name + " • " + str(width) + "x" 
                   + str(height) + " • " + str(minutes) + 'm:' 
                   + str(round(seconds)) + "s • " + str(num_frames) 
                   + "frames • " + str(round(frame_rate)) + "fps" )

    ctkimage = CTkImage(pillow_image_open("temp.jpg"), size = (25, 25))
    
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

def show_error_message(exception):
    messageBox_title = "Upscale error"

    messageBox_text  = str(exception) + "\n\n" + "Please report the error on Github/Telegram"

    CTkMessageBox(text = messageBox_text, title = messageBox_title, type = "error")



# GUI select from menus functions ---------------------------

def select_fluidity_option_from_menu(new_value: str):
    global selected_fluidity_option    
    selected_fluidity_option = new_value

def select_AI_device_from_menu(new_value: str):
    global selected_AI_device    

    for device in device_list:
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

def open_info_fluidity_option():
    messageBox_title = "AI fluidity"
    messageBox_text = """This widget allows you to choose between different AI fluidity option.

 • x2 • doubles video framerate • 30fps => 60fps
 • x4 • quadruples video framerate • 30fps => 120fps
 • x8 • octuplicate video framerate • 30fps => 240fps
 • x2-slowmotion • slowmotion effect by a factor of 2 (no audio)
 • x4-slowmotion • slowmotion effect by a factor of 4 (no audio)
 • x8-slowmotion • slowmotion effect by a factor of 8 (no audio)
"""

    CTkMessageBox(text = messageBox_text, title = messageBox_title)
    
def open_info_device():
    messageBox_title = "GPU"

    messageBox_text = """This widget allows you to select the GPU for AI processing.

 • Keep in mind that the more powerful your GPU is, faster the upscaling will be
 • For optimal results, it's essential to regularly update your GPU drivers"""

    CTkMessageBox(text = messageBox_text, title = messageBox_title)

def open_info_AI_output():
    messageBox_title = "AI output"

    messageBox_text = """This widget allows to choose the extension of upscaled frames.

 • .jpg • good quality • very fast
 • .png • very good quality • slow • supports transparent images
 • .bmp • highest quality • slow
 • .tiff • highest quality • very slow"""

    CTkMessageBox(text = messageBox_text, title = messageBox_title)

def open_info_input_resolution():
    messageBox_title = "Input resolution %"

    messageBox_text = """This widget allows to choose the resolution input to the AI.

For example for a 100x100px video:
 • Input resolution 50% => input to AI 50x50px
 • Input resolution 100% => input to AI 100x100px
 • Input resolution 200% => input to AI 200x200px """

    CTkMessageBox(text = messageBox_text, title = messageBox_title)

def open_info_cpu():
    messageBox_title = "Cpu number"

    messageBox_text = """This widget allows you to choose how many cpus to devote to the app.
    
Where possible the app will use the number of cpus selected."""

    CTkMessageBox(text = messageBox_text, title = messageBox_title)

def open_info_video_extension():
    messageBox_title = "Video output"

    messageBox_text = """This widget allows you to choose the video output.

 • .mp4  • produces good quality and well compressed video
 • .avi  • produces the highest quality video"""

    CTkMessageBox(text = messageBox_text, title = messageBox_title)  

def open_info_save_frames():
    messageBox_title = "Save frames"

    messageBox_text = """This widget allows you to choose to save frames generated by the AI.

 • Enabled
    A video called eating_pizza.mp4 will create
       • the fluidified video eating_pizza_RIFE.mp4
       • the folder eating_pizza_RIFE containing all frames 

• Disabled
    FluidFrames will generate only the fluidified video"""

    CTkMessageBox(text = messageBox_text, title = messageBox_title)  



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
        scrollable_frame_file_list = ScrollableImagesTextFrame(master = window, fg_color = dark_color, bg_color = dark_color)
        scrollable_frame_file_list.place(relx = 0.0, 
                                        rely = 0.0, 
                                        relwidth  = 1.0,  
                                        relheight = 0.45)
        
        scrollable_frame_file_list.add_clean_button()

        for index in range(supported_files_counter):
            actual_file = supported_files_list[index]
            if check_if_file_is_video(actual_file):
                video_label, ctkimage = extract_video_info(actual_file)
                scrollable_frame_file_list.add_item(text_to_show = video_label, image = ctkimage, file_element = actual_file)
                remove_file("temp.jpg")

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

def place_fluidity_option_menu():
    fluidity_option_button = create_info_button(open_info_fluidity_option, "AI fluidity")
    fluidity_option_menu   = create_option_menu(select_fluidity_option_from_menu, fluidity_options_list)

    fluidity_option_button.place(relx = column0_x, rely = row1_y - 0.05, anchor = "center")
    fluidity_option_menu.place(relx = column0_x, rely = row1_y, anchor = "center")

def place_gpu_menu():
    gpu_button = create_info_button(open_info_device, "GPU")
    gpu_menu   = create_option_menu(select_AI_device_from_menu, device_list_names)
    
    gpu_button.place(relx = column0_x, rely = row2_y - 0.053, anchor = "center")
    gpu_menu.place(relx = column0_x, rely  = row2_y, anchor = "center")

def place_AI_output_menu():
    file_extension_button = create_info_button(open_info_AI_output, "AI output")
    file_extension_menu   = create_option_menu(select_image_extension_from_menu, image_extension_list)
    
    file_extension_button.place(relx = column0_x, rely = row3_y - 0.053, anchor = "center")
    file_extension_menu.place(relx = column0_x, rely = row3_y, anchor = "center")

def place_video_extension_menu():
    video_extension_button = create_info_button(open_info_video_extension, "Video output")
    video_extension_menu   = create_option_menu(select_video_extension_from_menu, video_extension_list)
    
    video_extension_button.place(relx = column1_x, rely = row0_y - 0.053, anchor = "center")
    video_extension_menu.place(relx = column1_x, rely = row0_y, anchor = "center")

def place_save_frames_menu():
    save_frames_button = create_info_button(open_info_save_frames, "Save frames")
    save_frames_menu   = create_option_menu(select_save_frame_from_menu, save_frames_list)
    
    save_frames_button.place(relx = column1_x, rely = row1_y - 0.053, anchor = "center")
    save_frames_menu.place(relx = column1_x, rely = row1_y, anchor = "center")

def place_input_resolution_textbox():
    resize_factor_button  = create_info_button(open_info_input_resolution, "Input resolution %")
    resize_factor_textbox = create_text_box(selected_resize_factor) 

    resize_factor_button.place(relx = column1_x, rely = row2_y - 0.053, anchor = "center")
    resize_factor_textbox.place(relx = column1_x, rely = row2_y, anchor = "center")

def place_cpu_textbox():
    cpu_button  = create_info_button(open_info_cpu, "CPU number")
    cpu_textbox = create_text_box(selected_cpu_number)

    cpu_button.place(relx = column1_x, rely = row3_y - 0.053, anchor = "center")
    cpu_textbox.place(relx = column1_x, rely = row3_y, anchor = "center")

def place_message_label():
    message_label = CTkLabel(master  = window, 
                            textvariable = info_message,
                            height       = 25,
                            font         = bold10,
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

        place_fluidity_option_menu()
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
    freeze_support()

    set_appearance_mode("Dark")
    set_default_color_theme("dark-blue")

    window = CTk() 

    global selected_file_list
    global selected_fluidity_option
    global selected_AI_device 
    global selected_save_frames
    global resize_factor
    global cpu_number

    global selected_image_extension
    global selected_video_extension

    selected_file_list = []
    selected_AI_device = 0

    selected_fluidity_option = fluidity_options_list[0]
    selected_image_extension = image_extension_list[0]
    selected_video_extension = video_extension_list[0]

    if   save_frames_list[0] == "Disabled": selected_save_frames = False
    elif save_frames_list[0] == "Enabled":  selected_save_frames = True

    info_message            = StringVar()
    selected_resize_factor  = StringVar()
    selected_cpu_number     = StringVar()

    info_message.set("Hi :)")

    selected_resize_factor.set("50")
    cpu_count = str(int(os_cpu_count()/2))
    selected_cpu_number.set(cpu_count)

    font   = "Segoe UI"    
    bold8  = CTkFont(family = font, size = 8, weight = "bold")
    bold9  = CTkFont(family = font, size = 9, weight = "bold")
    bold10 = CTkFont(family = font, size = 10, weight = "bold")
    bold11 = CTkFont(family = font, size = 11, weight = "bold")
    bold12 = CTkFont(family = font, size = 12, weight = "bold")
    bold18 = CTkFont(family = font, size = 18, weight = "bold")
    bold19 = CTkFont(family = font, size = 19, weight = "bold")
    bold20 = CTkFont(family = font, size = 20, weight = "bold")
    bold21 = CTkFont(family = font, size = 21, weight = "bold")
    bold24 = CTkFont(family = font, size = 24, weight = "bold")


    # Images
    play_icon      = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}upscale_icon.png")),   size=(15, 15))
    logo_git       = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}github_logo.png")),    size=(15, 15))
    logo_telegram  = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}telegram_logo.png")),  size=(15, 15))
    stop_icon      = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}stop_icon.png")),      size=(15, 15))
    upscale_icon   = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}upscale_icon.png")),   size=(15, 15))
    clear_icon     = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}clear_icon.png")),     size=(15, 15))
    info_icon      = CTkImage(pillow_image_open(find_by_relative_path(f"Assets{os_separator}info_icon.png")),      size=(14, 14))

    app = App(window)
    window.update()
    window.mainloop()