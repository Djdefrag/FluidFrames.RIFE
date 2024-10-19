
# Standard library imports
import sys
from functools  import cache
from time       import sleep
from webbrowser import open as open_browser
from subprocess import run  as subprocess_run
from shutil     import rmtree as remove_directory
from timeit     import default_timer as timer

from typing    import Callable
from threading import Thread
from multiprocessing.pool import ThreadPool
from multiprocessing import ( 
    Process, 
    Queue          as multiprocessing_Queue,
    freeze_support as multiprocessing_freeze_support
)

from json import (
    load  as json_load, 
    dumps as json_dumps
)

from os import (
    sep        as os_separator,
    devnull    as os_devnull,
    environ    as os_environ,
    makedirs   as os_makedirs,
    listdir    as os_listdir,
    remove     as os_remove
)

from os.path import (
    basename   as os_path_basename,
    dirname    as os_path_dirname,
    abspath    as os_path_abspath,
    join       as os_path_join,
    exists     as os_path_exists,
    splitext   as os_path_splitext,
    expanduser as os_path_expanduser
)

# Third-party library imports
from natsort          import natsorted
from moviepy.video.io import ImageSequenceClip 
from onnxruntime      import InferenceSession as onnxruntime_inferenceSession

from PIL.Image import (
    open      as pillow_image_open,
    fromarray as pillow_image_fromarray
)

from cv2 import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    COLOR_BGR2RGB,
    IMREAD_UNCHANGED,
    INTER_LINEAR,
    INTER_AREA,
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
    concatenate as numpy_concatenate, 
    transpose   as numpy_transpose,
    expand_dims as numpy_expand_dims,
    squeeze     as numpy_squeeze,
    clip        as numpy_clip,
    mean        as numpy_mean,
    float32,
    uint8
)

# GUI imports
from tkinter import StringVar
from tkinter import DISABLED
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
    set_default_color_theme
)

if sys.stdout is None: sys.stdout = open(os_devnull, "w")
if sys.stderr is None: sys.stderr = open(os_devnull, "w")

def find_by_relative_path(relative_path: str) -> str:
    base_path = getattr(sys, '_MEIPASS', os_path_dirname(os_path_abspath(__file__)))
    return os_path_join(base_path, relative_path)



app_name   = "FluidFrames"
version    = "3.10"
dark_color = "#080808"

githubme   = "https://github.com/Djdefrag/FluidFrames.RIFE"
telegramme = "https://linktr.ee/j3ngystudio"

AI_models_list          = [ "RIFE_4.17", "RIFE_4.15_Lite" ]
generation_options_list = [ "x2", "x4", "x8", "Slowmotion x2", "Slowmotion x4", "Slowmotion x8" ]
gpus_list               = [ "GPU 1", "GPU 2", "GPU 3", "GPU 4" ]
image_extension_list    = [ ".jpg", ".png" ]
video_extension_list    = [ ".mp4 (x264)", ".mp4 (x265)", ".avi" ]
keep_frames_list        = [ "Enabled", "Disabled" ]


OUTPUT_PATH_CODED    = "Same path as input files"
DOCUMENT_PATH        = os_path_join(os_path_expanduser('~'), 'Documents')
USER_PREFERENCE_PATH = find_by_relative_path(f"{DOCUMENT_PATH}{os_separator}{app_name}_UserPreference.json")
FFMPEG_EXE_PATH      = find_by_relative_path(f"Assets{os_separator}ffmpeg.exe")
EXIFTOOL_EXE_PATH    = find_by_relative_path(f"Assets{os_separator}exiftool.exe")
FRAMES_FOR_CPU       = 30

if os_path_exists(FFMPEG_EXE_PATH): 
    print(f"[{app_name}] External ffmpeg.exe file found")
    os_environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_EXE_PATH

if os_path_exists(USER_PREFERENCE_PATH):
    print(f"[{app_name}] Preference file exist")
    with open(USER_PREFERENCE_PATH, "r") as json_file:
        json_data = json_load(json_file)
        default_AI_model          = json_data["default_AI_model"]
        default_generation_option = json_data["default_generation_option"]
        default_gpu               = json_data["default_gpu"]
        default_image_extension   = json_data["default_image_extension"]
        default_video_extension   = json_data["default_video_extension"]
        default_keep_frames       = json_data["default_keep_frames"]
        default_output_path       = json_data["default_output_path"]
        default_resize_factor     = json_data["default_resize_factor"]
        default_cpu_number        = json_data["default_cpu_number"]
else:
    print(f"[{app_name}] Preference file does not exist, using default coded value")
    default_AI_model          = AI_models_list[0]
    default_generation_option = generation_options_list[0]
    default_gpu               = gpus_list[0]
    default_image_extension   = image_extension_list[0]
    default_video_extension   = video_extension_list[0]
    default_keep_frames       = keep_frames_list[0]
    default_output_path       = OUTPUT_PATH_CODED
    default_resize_factor     = str(50)
    default_cpu_number        = str(4)


COMPLETED_STATUS = "Completed"
ERROR_STATUS     = "Error"
STOP_STATUS      = "Stop"

offset_y_options = 0.105
row0_y = 0.52
row1_y = row0_y + offset_y_options
row2_y = row1_y + offset_y_options
row3_y = row2_y + offset_y_options
row4_y = row3_y + offset_y_options

offset_x_options = 0.28
column1_x = 0.5
column0_x = column1_x - offset_x_options
column2_x = column1_x + offset_x_options
column1_5_x = column1_x + offset_x_options/2

if sys.stdout is None: sys.stdout = open(os_devnull, "w")
if sys.stderr is None: sys.stderr = open(os_devnull, "w")

supported_file_extensions = [
    ".mp4", ".MP4", ".webm", ".WEBM", ".mkv", ".MKV",
    ".flv", ".FLV", ".gif", ".GIF", ".m4v", ".M4V",
    ".avi", ".AVI", ".mov", ".MOV", ".qt", ".3gp",
    ".mpg", ".mpeg", ".vob", ".VOB"
]

supported_video_extensions = [
    ".mp4", ".MP4", ".webm", ".WEBM", ".mkv", ".MKV",
    ".flv", ".FLV", ".gif", ".GIF", ".m4v", ".M4V",
    ".avi", ".AVI", ".mov", ".MOV", ".qt", ".3gp",
    ".mpg", ".mpeg", ".vob", ".VOB"
]



# AI -------------------

class AI:

    # CLASS INIT FUNCTIONS

    def __init__(
            self, 
            AI_model_name: str, 
            directml_gpu: str, 
            resize_factor: int,
            frame_gen_factor: int
            ):
        
        # Passed variables
        self.AI_model_name    = AI_model_name
        self.directml_gpu     = directml_gpu
        self.resize_factor    = resize_factor
        self.frame_gen_factor = frame_gen_factor

        # Calculated variables
        self.AI_model_path    = find_by_relative_path(f"AI-onnx{os_separator}{self.AI_model_name}.onnx")
        self.inferenceSession = self._load_inferenceSession()

    def _load_inferenceSession(self) -> onnxruntime_inferenceSession:        
        match self.directml_gpu:
            case 'GPU 1': directml_backend = [('DmlExecutionProvider', {"device_id": "0"})]
            case 'GPU 2': directml_backend = [('DmlExecutionProvider', {"device_id": "1"})]
            case 'GPU 3': directml_backend = [('DmlExecutionProvider', {"device_id": "2"})]
            case 'GPU 4': directml_backend = [('DmlExecutionProvider', {"device_id": "3"})]
            case 'CPU':   directml_backend = ['CPUExecutionProvider']

        inference_session = onnxruntime_inferenceSession(path_or_bytes = self.AI_model_path, providers = directml_backend)

        return inference_session



    # INTERNAL CLASS FUNCTIONS

    def get_image_mode(self, image: numpy_ndarray) -> str:
        match image.shape:
            case (rows, cols):
                return "Grayscale"
            case (rows, cols, channels) if channels == 3:
                return "RGB"
            case (rows, cols, channels) if channels == 4:
                return "RGBA"

    def get_image_resolution(self, image: numpy_ndarray) -> tuple:
        height = image.shape[0]
        width  = image.shape[1]

        return height, width 

    def resize_image(self, image: numpy_ndarray) -> numpy_ndarray:
        old_height, old_width = self.get_image_resolution(image)

        new_width  = int(old_width * self.resize_factor)
        new_height = int(old_height * self.resize_factor)

        match self.resize_factor:
            case factor if factor > 1:
                return opencv_resize(image, (new_width, new_height), interpolation = INTER_LINEAR)
            case factor if factor < 1:
                return opencv_resize(image, (new_width, new_height), interpolation = INTER_AREA)
            case _:
                return image



    # AI CLASS FUNCTIONS

    def concatenate_images(self, image1: numpy_ndarray, image2: numpy_ndarray) -> numpy_ndarray:
        image1 = image1 / 255
        image2 = image2 / 255
        concateneted_image = numpy_concatenate((image1, image2), axis=2)
        return concateneted_image

    def preprocess_image(self, image: numpy_ndarray) -> numpy_ndarray:
        image = numpy_transpose(image, (2, 0, 1))
        image = numpy_expand_dims(image, axis=0)
        return image

    def onnxruntime_inference(self, image: numpy_ndarray) -> numpy_ndarray:

        # IO BINDING
        
        # io_binding = self.inferenceSession.io_binding()
        # io_binding.bind_cpu_input(self.inferenceSession.get_inputs()[0].name, image)
        # io_binding.bind_output(self.inferenceSession.get_outputs()[0].name, element_type = float32)
        # self.inferenceSession.run_with_iobinding(io_binding)
        # onnx_output = io_binding.copy_outputs_to_cpu()[0]

        onnx_input  = {self.inferenceSession.get_inputs()[0].name: image}
        onnx_output = self.inferenceSession.run(None, onnx_input)[0]

        return onnx_output

    def postprocess_output(self, onnx_output: numpy_ndarray) -> numpy_ndarray:
        onnx_output = numpy_squeeze(onnx_output, axis=0)
        onnx_output = numpy_clip(onnx_output, 0, 1)
        onnx_output = numpy_transpose(onnx_output, (1, 2, 0))

        return onnx_output.astype(float32)

    def de_normalize_image(self, onnx_output: numpy_ndarray, max_range: int) -> numpy_ndarray:    
        match max_range:
            case 255:   return (onnx_output * max_range).astype(uint8)
            case 65535: return (onnx_output * max_range).round().astype(float32)

    def AI_interpolation(self, image1: numpy_ndarray, image2: numpy_ndarray) -> numpy_ndarray:
        image        = self.concatenate_images(image1, image2).astype(float32)
        image        = self.preprocess_image(image)
        onnx_output  = self.onnxruntime_inference(image)
        onnx_output  = self.postprocess_output(onnx_output)     
        output_image = self.de_normalize_image(onnx_output, 255) 

        return output_image  



    # EXTERNAL FUNCTION

    def AI_orchestration(self, image1: numpy_ndarray, image2: numpy_ndarray) -> tuple[list[numpy_ndarray], list[numpy_ndarray]]:

        generated_images = []
        
        if self.frame_gen_factor == 2:   # Generate 1 image [image1 / image_A / image2]
            image_A = self.AI_interpolation(image1, image2)
            generated_images.append(image_A)

        elif self.frame_gen_factor == 4: # Generate 3 images [image1 / image_A / image_B / image_C / image2]
            image_B = self.AI_interpolation(image1, image2)
            image_A = self.AI_interpolation(image1, image_B)
            image_C = self.AI_interpolation(image_B, image2)

            generated_images.append(image_A)
            generated_images.append(image_B)
            generated_images.append(image_C)

        elif self.frame_gen_factor == 8: # Generate 7 images [image1 / image_A / image_B / image_C / image_D / image_E / image_F / image_G / image2]
            image_D = self.AI_interpolation(image1, image2)
            image_B = self.AI_interpolation(image1, image_D)
            image_A = self.AI_interpolation(image1, image_B)
            image_C = self.AI_interpolation(image_B, image_D)
            image_F = self.AI_interpolation(image_D, image2)
            image_E = self.AI_interpolation(image_D, image_F)
            image_G = self.AI_interpolation(image_F, image2)

            generated_images.append(image_A)
            generated_images.append(image_B)
            generated_images.append(image_C)
            generated_images.append(image_D)
            generated_images.append(image_E)
            generated_images.append(image_F)
            generated_images.append(image_G)

        return generated_images



# GUI utils ---------------------------

class MessageBox(CTkToplevel):

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
        self.lift()                         
        self.attributes("-topmost", True)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.after(10, self._create_widgets)
        self.resizable(False, False)
        self.grab_set()                       

    def _ok_event(self) -> None:
        self.grab_release()
        self.destroy()

    def _on_closing(self) -> None:
        self.grab_release()
        self.destroy()

    def createEmptyLabel(self) -> CTkLabel:
        return CTkLabel(
            master = self,
            fg_color = "transparent",
            width    = 500,
            height   = 17,
            text     = ''
        )

    def placeInfoMessageTitleSubtitle(self) -> None:
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

    def placeInfoMessageOptionsText(self) -> None:
        for option_text in self._option_list:
            optionLabel = CTkLabel(
                master = self,
                width  = 600,
                height = 45,
                corner_radius = 6,
                anchor     = 'w',
                justify    = "left",
                text_color = "#C0C0C0",
                fg_color   = "#282828",
                bg_color   = "transparent",
                font       = bold12,
                text       = option_text
            )
            
            self._ctkwidgets_index += 1
            optionLabel.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 25, pady = 4, sticky = "ew")

        spacingLabel3 = self.createEmptyLabel()

        self._ctkwidgets_index += 1
        spacingLabel3.grid(row = self._ctkwidgets_index, column = 0, columnspan = 2, padx = 0, pady = 0, sticky = "ew")

    def placeInfoMessageOkButton(self) -> None:
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

    def _create_widgets(self) -> None:
        self.grid_columnconfigure((0, 1), weight=1)
        self.rowconfigure(0, weight=1)

        self.placeInfoMessageTitleSubtitle()
        self.placeInfoMessageOptionsText()
        self.placeInfoMessageOkButton()

class FileWidget(CTkScrollableFrame):

    def __init__(
            self, 
            master,
            selected_file_list, 
            resize_factor = 0,
            frame_generation_factor = 1,
            **kwargs
            ) -> None:
        
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight = 1)

        self.file_list               = selected_file_list
        self.resize_factor           = resize_factor
        self.frame_generation_factor = frame_generation_factor

        self.label_list = []
        self._create_widgets()

    def _destroy_(self) -> None:
        self.file_list = []
        self.destroy()
        place_loadFile_section()

    def _create_widgets(self) -> None:
        self.add_clean_button()
        index_row = 1
        for file_path in self.file_list:
            label = self.add_file_information(file_path, index_row)
            self.label_list.append(label)
            index_row +=1



    def add_file_information(self, file_path, index_row) -> CTkLabel:
        infos, icon = self.extract_file_info(file_path)
        label = CTkLabel(
            self, 
            text       = infos,
            image      = icon, 
            font       = bold12,
            text_color = "#C0C0C0",
            compound   = "left", 
            anchor     = "w",
            padx       = 10,
            pady       = 5,
            justify    = "left",
        )      
        label.grid(
            row    = index_row, 
            column = 0,
            pady   = (3, 3), 
            padx   = (3, 3),
            sticky = "w")
        
        return label

    def add_clean_button(self) -> None:
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

        button.configure(command=lambda: self._destroy_())
        button.grid(row = 0, column=2, pady=(7, 7), padx = (0, 7))
        
    @cache
    def extract_file_icon(self, file_path) -> CTkImage:
        max_size = 50

        if check_if_file_is_video(file_path):
            video_cap   = opencv_VideoCapture(file_path)
            _, frame    = video_cap.read()
            source_icon = opencv_cvtColor(frame, COLOR_BGR2RGB)
            video_cap.release()
        else:
            source_icon = opencv_cvtColor(image_read(file_path), COLOR_BGR2RGB)

        ratio       = min(max_size / source_icon.shape[0], max_size / source_icon.shape[1])
        new_width   = int(source_icon.shape[1] * ratio)
        new_height  = int(source_icon.shape[0] * ratio)
        source_icon = opencv_resize(source_icon,(new_width, new_height))
        ctk_icon    = CTkImage(pillow_image_fromarray(source_icon, mode="RGB"), size = (new_width, new_height))

        return ctk_icon

    def extract_file_info(self, file_path) -> tuple:
        
        if check_if_file_is_video(file_path):
            cap          = opencv_VideoCapture(file_path)
            width        = round(cap.get(CAP_PROP_FRAME_WIDTH))
            height       = round(cap.get(CAP_PROP_FRAME_HEIGHT))
            num_frames   = int(cap.get(CAP_PROP_FRAME_COUNT))
            frame_rate   = cap.get(CAP_PROP_FPS)
            duration     = num_frames/frame_rate
            minutes      = int(duration/60)
            seconds      = duration % 60
            cap.release()

            video_name = str(file_path.split("/")[-1])
            file_icon  = self.extract_file_icon(file_path)

            file_infos = (f"{video_name}\n"
                          f"{width}x{height} • {round(frame_rate, 2)} fps • {minutes}m:{round(seconds)}s • {num_frames} frames\n")
            
            if self.resize_factor != 0:
                resized_height  = int(height * (self.resize_factor/100))
                resized_width   = int(width * (self.resize_factor/100))

                if   "x2" in self.frame_generation_factor: generation_factor = 2
                elif "x4" in self.frame_generation_factor: generation_factor = 4
                elif "x8" in self.frame_generation_factor: generation_factor = 8

                if "Slowmotion" in self.frame_generation_factor: slowmotion = True
                else: slowmotion = False

                if slowmotion:
                    duration_slowmotion = (num_frames/frame_rate) * generation_factor
                    minutes_slowmotion  = int(duration_slowmotion/60)
                    seconds_slowmotion  = duration_slowmotion % 60
                    
                    file_infos += (f"AI input {self.resize_factor}% ➜ {resized_width}x{resized_height} • {round(frame_rate, 2)} fps \n"
                                   f"AI output x{generation_factor}-slowmo ➜ {resized_width}x{resized_height} • {round(frame_rate, 2)} fps • {minutes_slowmotion}m:{round(seconds_slowmotion)}s")
                else:
                    fps_frame_generated = frame_rate * generation_factor

                    file_infos += (f"AI input {self.resize_factor}% ➜ {resized_width}x{resized_height} • {round(frame_rate, 2)} fps \n"
                                   f"AI output x{generation_factor} ➜ {resized_width}x{resized_height} • {round(fps_frame_generated, 2)} fps")


            return file_infos, file_icon



    # EXTERNAL FUNCTIONS

    def clean_file_list(self) -> None:
        for label in self.label_list:
            label.grid_forget()

    def get_selected_file_list(self) -> list: 
        return self.file_list  

    def set_frame_generation_factor(self, frame_generation_factor) -> None:
        self.frame_generation_factor = frame_generation_factor

    def set_resize_factor(self, resize_factor) -> None:
        self.resize_factor = resize_factor


def update_file_widget(a, b, c) -> None:
    try:
        global file_widget
        file_widget
    except:
        return
        
    try:
        resize_factor = int(float(str(selected_resize_factor.get())))
    except:
        resize_factor = 0
    
    file_widget.clean_file_list()
    file_widget.set_resize_factor(resize_factor)
    file_widget.set_frame_generation_factor(selected_generation_option)
    file_widget._create_widgets()

def create_info_button(
        command: Callable, 
        text: str,
        width: int = 150
        ) -> CTkButton:
    
    return CTkButton(
        master  = window, 
        command = command,
        text          = text,
        fg_color      = "transparent",
        hover_color   = "#181818",
        text_color    = "#C0C0C0",
        anchor        = "w",
        height        = 22,
        width         = width,
        corner_radius = 10,
        font          = bold12,
        image         = info_icon
    )

def create_option_menu(
        command: Callable, 
        values: list,
        default_value: str) -> CTkOptionMenu:
    
    option_menu = CTkOptionMenu(
        master  = window, 
        command = command,
        values  = values,
        width              = 150,
        height             = 30,
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
    option_menu.set(default_value)
    return option_menu

def create_text_box(textvariable: StringVar) -> CTkEntry:
    return CTkEntry(
        master        = window, 
        textvariable  = textvariable,
        corner_radius = 6,
        width         = 150,
        height        = 30,
        font          = bold11,
        justify       = "center",
        text_color    = "#C0C0C0",
        fg_color      = "#000000",
        border_width  = 1,
        border_color  = "#404040",
    )

def create_text_box_output_path(textvariable: StringVar) -> CTkEntry:
    return CTkEntry(
        master        = window, 
        textvariable  = textvariable,
        border_width  = 1,
        corner_radius = 6,
        width         = 300,
        height        = 30,
        font          = bold11,
        justify       = "center",
        text_color    = "#C0C0C0",
        fg_color      = "#000000",
        border_color  = "#404040",
        state         = DISABLED
    )

def create_active_button(
        command: Callable,
        text: str,
        icon: CTkImage = None,
        width: int = 140,
        height: int = 30,
        border_color: str = "#0096FF"
        ) -> CTkButton:
    
    return CTkButton(
        master     = window, 
        command    = command,
        text       = text,
        image      = icon,
        width      = width,
        height     = height,
        font         = bold11,
        border_width = 1,
        fg_color     = "#282828",
        text_color   = "#E0E0E0",
        border_color = border_color
    )



# File Utils functions ------------------------

def image_write(file_path: str, file_data: numpy_ndarray) -> None: 
    _, file_extension = os_path_splitext(file_path)
    opencv_imencode(file_extension, file_data)[1].tofile(file_path)

def image_read(file_path: str) -> numpy_ndarray: 
    with open(file_path, 'rb') as file:
        return opencv_imdecode(numpy_frombuffer(file.read(), uint8), IMREAD_UNCHANGED)

def remove_dir(name_dir: str) -> None:
    if os_path_exists(name_dir): remove_directory(name_dir)

def create_dir(name_dir: str) -> None:
    if os_path_exists(name_dir):     remove_directory(name_dir)
    if not os_path_exists(name_dir): os_makedirs(name_dir, mode=0o777)



# Image/video Utils functions ------------------------

def get_video_fps(video_path: str) -> float:
    video_capture = opencv_VideoCapture(video_path)
    frame_rate    = video_capture.get(CAP_PROP_FPS)
    video_capture.release()
    return frame_rate

def save_extracted_frames(
        extracted_frames_paths: list[str], 
        extracted_frames: list[numpy_ndarray], 
        cpu_number: int
        ) -> None:
    
    pool = ThreadPool(cpu_number)
    pool.starmap(image_write, zip(extracted_frames_paths, extracted_frames))
    pool.close()
    pool.join()

def extract_video_frames(
        processing_queue: multiprocessing_Queue,
        file_number: int,
        target_directory: str,
        AI_instance: AI,
        video_path: str, 
        selected_image_extension: str,
        cpu_number: int
    ) -> tuple[list[str], str]:

    create_dir(target_directory)

    # Video frame extraction + resize
    frames_number_to_save = cpu_number * FRAMES_FOR_CPU
    video_capture         = opencv_VideoCapture(video_path)
    frame_count           = int(video_capture.get(CAP_PROP_FRAME_COUNT))

    extracted_frames       = []
    extracted_frames_paths = []
    video_frames_list      = []

    for frame_number in range(frame_count):
        success, frame = video_capture.read()
        if success:
            frame_path = f"{target_directory}{os_separator}frame_{frame_number:03d}{selected_image_extension}"
            frame      = AI_instance.resize_image(frame)
            
            extracted_frames.append(frame)
            extracted_frames_paths.append(frame_path)
            video_frames_list.append(frame_path)

            if len(extracted_frames) == frames_number_to_save:
                percentage_extraction = (frame_number / frame_count) * 100

                write_process_status(processing_queue, f"{file_number}. Extracting video frames ({round(percentage_extraction, 2)}%)")
                save_extracted_frames(extracted_frames_paths, extracted_frames, cpu_number)
                extracted_frames       = []
                extracted_frames_paths = []

    video_capture.release()

    if len(extracted_frames) > 0: save_extracted_frames(extracted_frames_paths, extracted_frames, cpu_number)
    
    return video_frames_list

def video_encoding(
        video_path: str, 
        video_output_path: str,
        total_frames_paths: list, 
        frame_gen_factor: int,
        slowmotion: bool,
        cpu_number: int,
        selected_video_extension: str
        ) -> None:
    
    match selected_video_extension:
        case ".mp4 (x264)": codec = "libx264"
        case ".mp4 (x265)": codec = "libx265"
        case ".avi":        codec = "png"


    no_audio_path = f"{os_path_splitext(video_output_path)[0]}_no_audio{os_path_splitext(video_output_path)[1]}"
    video_fps     = get_video_fps(video_path) if slowmotion else get_video_fps(video_path) * frame_gen_factor
    video_clip    = ImageSequenceClip.ImageSequenceClip(sequence = total_frames_paths, fps = video_fps)

    if slowmotion:
        video_clip.write_videofile(
            filename = video_output_path,
            fps      = video_fps,
            codec    = codec,
            threads  = cpu_number,
            verbose  = False,
            logger   = None,
            audio    = None,
            bitrate  = "12M",
            preset   = "ultrafast"
        )
    else:
        video_clip.write_videofile(
            filename = no_audio_path,
            fps      = video_fps,
            codec    = codec,
            threads  = cpu_number,
            verbose  = False,
            logger   = None,
            audio    = None,
            bitrate  = "12M",
            preset   = "ultrafast"
        )  

        # Copy the audio from original video
        audio_passthrough_command = [
            FFMPEG_EXE_PATH,
            "-y",
            "-i", video_path,
            "-i", no_audio_path,
            "-c:v", "copy",
            "-map", "1:v:0",
            "-map", "0:a?",
            "-c:a", "copy",
            video_output_path
        ]
        try: 
            subprocess_run(audio_passthrough_command, check = True, shell = "False")
            if os_path_exists(no_audio_path): os_remove(no_audio_path)
        except:
            pass

def check_video_frame_generation_resume(
        target_directory: str, 
        selected_AI_model: str,
        selected_image_extension: str
        ) -> bool:
    
    if os_path_exists(target_directory):
        directory_files        = os_listdir(target_directory)
        generated_frames_paths = [file for file in directory_files if selected_AI_model in file]
        generated_frames_paths = [file for file in generated_frames_paths if file.endswith(selected_image_extension)]

        if len(generated_frames_paths) > 1:
            return True
        else:
            return False
    else:
        return False

def get_video_frames_for_frame_generation_resume(
        target_directory: str,
        selected_AI_model: str,
        selected_image_extension: str
        ) -> list[str]:
    
    # Only file names
    directory_files      = os_listdir(target_directory)
    original_frames_path = [file for file in directory_files if file.endswith(selected_image_extension)]
    original_frames_path = [file for file in original_frames_path if selected_AI_model not in file]

    # Adding the complete path to files
    original_frames_path = natsorted([os_path_join(target_directory, file) for file in original_frames_path])

    return original_frames_path

def calculate_time_to_complete_video(
        time_for_frame: float, 
        remaining_frames: int
        ) -> str:
    
    remaining_time = time_for_frame * remaining_frames

    hours_left   = remaining_time // 3600
    minutes_left = (remaining_time % 3600) // 60
    seconds_left = round((remaining_time % 3600) % 60)

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
        frame_index: int, 
        how_many_frames: int,
        average_processing_time: float,
        ) -> None:
    
    if frame_index != 0 and (frame_index + 1) % 8 == 0:  

        remaining_frames = how_many_frames - frame_index
        remaining_time   = calculate_time_to_complete_video(average_processing_time, remaining_frames)
        if remaining_time != "":
            percent_complete = (frame_index + 1) / how_many_frames * 100 
            write_process_status(processing_queue, f"{file_number}. Generating frames {percent_complete:.2f}% ({remaining_time})")

def save_generated_video_frames(
        generated_images_paths: list[str],
        generated_images: list[numpy_ndarray],
        ) -> None:
    
    for i in range(len(generated_images)): image_write(generated_images_paths[i], generated_images[i])

def prepare_generated_frames_paths(
        base_path: str,
        selected_AI_model: str,
        selected_image_extension: str,
        frame_gen_factor: int
        ) -> list[str]:
    
    generated_frames_paths = [f"{base_path}_{selected_AI_model}_{i}{selected_image_extension}" for i in range(frame_gen_factor-1)]
    
    return generated_frames_paths

def prepare_output_video_frame_filenames(
        extracted_frames_paths: list[str],
        selected_AI_model: str,
        frame_gen_factor: int,
        selected_image_extension: str,
        ) -> list[str]:

    total_frames_paths = []
    how_many_frames    = len(extracted_frames_paths)

    for index in range(how_many_frames - 1):
        frame_path            = extracted_frames_paths[index]
        base_path             = os_path_splitext(frame_path)[0]
        generated_frames_paths = prepare_generated_frames_paths(base_path, selected_AI_model, selected_image_extension, frame_gen_factor)

        total_frames_paths.append(frame_path)
        total_frames_paths.extend(generated_frames_paths)

    total_frames_paths.append(extracted_frames_paths[-1])

    return total_frames_paths

def copy_file_metadata(
        original_file_path: str, 
        upscaled_file_path: str
        ) -> None:
    
    exiftool_cmd = [
        EXIFTOOL_EXE_PATH, 
        '-fast', 
        '-TagsFromFile', 
        original_file_path, 
        '-overwrite_original', 
        '-all:all',
        '-unsafe',
        '-largetags', 
        upscaled_file_path
    ]
    
    try: 
        subprocess_run(exiftool_cmd, check = True, shell = 'False')
    except:
        pass



# Core functions ------------------------

def stop_thread() -> None:
    stop = 1 + "x"

def check_frame_generation_steps() -> None:
    sleep(1)

    try:
        while True:
            actual_step = read_process_status()

            if actual_step == COMPLETED_STATUS:
                info_message.set(f"All files completed! :)")
                stop_generation_process()
                stop_thread()

            elif actual_step == STOP_STATUS:
                info_message.set(f"Generation stopped")
                stop_generation_process()
                stop_thread()

            elif ERROR_STATUS in actual_step:
                error_message = f"Error while generating :("
                error = actual_step.replace(ERROR_STATUS, "")
                info_message.set(error_message)
                show_error_message(error)
                stop_thread()

            else:
                info_message.set(actual_step)

            sleep(1)
    except:
        place_generation_button()

def read_process_status() -> None:
    return processing_queue.get()

def write_process_status(
        processing_queue: multiprocessing_Queue,
        step: str
        ) -> None:
    
    print(f"{step}")
    while not processing_queue.empty(): processing_queue.get()
    processing_queue.put(f"{step}")

def stop_generation_process() -> None:
    global process_frame_generation_orchestrator

    try:
        process_frame_generation_orchestrator
    except:
        pass
    else:
        process_frame_generation_orchestrator.terminate()
        process_frame_generation_orchestrator.join()

def stop_button_command() -> None:
    stop_generation_process()
    write_process_status(processing_queue, f"{STOP_STATUS}")

def generate_button_command() -> None: 
    global selected_file_list
    global selected_AI_model
    global selected_generation_option
    global selected_gpu
    global selected_image_extension
    global selected_video_extension
    global resize_factor
    global cpu_number
    global selected_keep_frames

    global process_frame_generation_orchestrator
    
    if user_input_checks():
        info_message.set("Loading")

        print("=" * 50)
        print(f"> Starting frame generation:")
        print(f"   Files to process: {len(selected_file_list)}")
        print(f"   Output path: {(selected_output_path.get())}")
        print(f"   Selected AI model: {selected_AI_model}")
        print(f"   Selected frame generation option: {selected_generation_option}")
        print(f"   Selected image output extension: {selected_image_extension}")
        print(f"   Selected video output extension: {selected_video_extension}")
        print(f"   Resize factor: {int(resize_factor * 100)}%")
        print(f"   Cpu number: {cpu_number}")
        print(f"   Save frames: {selected_keep_frames}")
        print("=" * 50)

        place_stop_button()

        process_frame_generation_orchestrator = Process(
            target = frame_generation_orchestrator,
            args = (
                processing_queue, 
                selected_file_list, 
                selected_output_path.get(),
                selected_AI_model,
                selected_gpu,
                selected_generation_option, 
                selected_image_extension, 
                selected_video_extension, 
                resize_factor, 
                cpu_number, 
                selected_keep_frames
            )
        )
        process_frame_generation_orchestrator.start()

        thread_wait = Thread(target = check_frame_generation_steps)
        thread_wait.start()



def prepare_output_video_filename(
        video_path: str, 
        selected_output_path: str,
        selected_AI_model: str,
        fluidification_factor: int, 
        slowmotion: bool, 
        resize_factor: int, 
        selected_video_extension: str
        ) -> str:
    
    match selected_video_extension:
        case '.mp4 (x264)':
            selected_video_extension = '.mp4'
        case '.mp4 (x265)':
            selected_video_extension = '.mp4'
        case '.avi':
            selected_video_extension = '.avi'

    if selected_output_path == OUTPUT_PATH_CODED:
        file_path_no_extension, _ = os_path_splitext(video_path)
        output_path = file_path_no_extension
    else:
        file_name   = os_path_basename(video_path)
        output_path = f"{selected_output_path}{os_separator}{file_name}"

    # Selected AI model
    to_append = f"_{selected_AI_model}x{str(fluidification_factor)}"

    # Slowmotion?
    if slowmotion: to_append += f"_slowmo"

    # Selected resize
    to_append += f"_Resize-{str(int(resize_factor * 100))}"

    # Video output
    to_append += f"{selected_video_extension}"

    output_path += to_append

    return output_path

def prepare_output_video_directory_name(
        video_path: str, 
        selected_output_path: str,
        selected_AI_model: str,
        frame_gen_factor: int, 
        slowmotion: bool, 
        resize_factor: int, 
        ) -> str:
    
    if selected_output_path == OUTPUT_PATH_CODED:
        file_path_no_extension, _ = os_path_splitext(video_path)
        output_path = file_path_no_extension
    else:
        file_name   = os_path_basename(video_path)
        output_path = f"{selected_output_path}{os_separator}{file_name}"

    # Selected AI model
    to_append = f"_{selected_AI_model}x{str(frame_gen_factor)}"

    # Slowmotion?
    if slowmotion: to_append += f"_slowmo"

    # Selected resize
    to_append += f"_Resize-{str(int(resize_factor * 100))}"

    output_path += to_append

    return output_path

def check_frame_generation_option(selected_generation_option: str) -> tuple:
    slowmotion = False
    frame_gen_factor = 0

    if "Slowmotion" in selected_generation_option: slowmotion = True

    if   "2" in selected_generation_option: frame_gen_factor = 2
    elif "4" in selected_generation_option: frame_gen_factor = 4
    elif "8" in selected_generation_option: frame_gen_factor = 8

    return frame_gen_factor, slowmotion



# ORCHESTRATOR

def frame_generation_orchestrator(
        processing_queue: multiprocessing_Queue,
        selected_file_list: list,
        selected_output_path: str,
        selected_AI_model: str,
        selected_gpu: str,
        selected_generation_option: str,
        selected_image_extension: str,
        selected_video_extension: str,
        resize_factor: int,
        cpu_number: int,
        selected_keep_frames: bool
        ) -> None:
            
    frame_gen_factor, slowmotion = check_frame_generation_option(selected_generation_option)
    how_many_files = len(selected_file_list)

    try:
        write_process_status(processing_queue, f"Loading AI model")
        AI_instance = AI(selected_AI_model, selected_gpu, resize_factor, frame_gen_factor)

        for file_number in range(how_many_files):
            file_path   = selected_file_list[file_number]
            file_number = file_number + 1

            video_frame_generation(
                processing_queue,
                file_path, 
                file_number,
                selected_output_path,
                AI_instance,
                selected_AI_model,
                frame_gen_factor, 
                slowmotion,
                resize_factor, 
                selected_image_extension, 
                selected_video_extension,
                cpu_number,
                selected_keep_frames
            )

        write_process_status(processing_queue, f"{COMPLETED_STATUS}")

    except Exception as exception:
        write_process_status(processing_queue, f"{ERROR_STATUS} {str(exception)}")

# FRAME GENERATION

def are_frames_already_generated(generated_images_paths: list[str]) -> bool:
    already_generated = all(os_path_exists(generated_image_path) for generated_image_path in generated_images_paths)
    return already_generated

def video_frame_generation(
        processing_queue: multiprocessing_Queue,
        video_path: str, 
        file_number: int,
        selected_output_path: str,
        AI_instance: AI,
        selected_AI_model: str,
        frame_gen_factor: int, 
        slowmotion: bool, 
        resize_factor: int, 
        selected_image_extension: str, 
        selected_video_extension: str,
        cpu_number: int, 
        selected_keep_frames: bool
        ) -> None:
    
    # 1. Preparation
    target_directory  = prepare_output_video_directory_name(video_path, selected_output_path, selected_AI_model, frame_gen_factor, slowmotion, resize_factor)
    video_output_path = prepare_output_video_filename(video_path, selected_output_path, selected_AI_model, frame_gen_factor, slowmotion, resize_factor, selected_video_extension)

    # 2. Resume frame generation OR Extract video frames and start frame generation
    frame_generation_resume = check_video_frame_generation_resume(target_directory, selected_AI_model, selected_image_extension)
    if frame_generation_resume:
        write_process_status(processing_queue, f"{file_number}. Resume frame generation")
        extracted_frames_paths = get_video_frames_for_frame_generation_resume(target_directory, selected_AI_model, selected_image_extension)
        total_frames_paths     = prepare_output_video_frame_filenames(extracted_frames_paths, selected_AI_model, frame_gen_factor, selected_image_extension)
    else:
        write_process_status(processing_queue, f"{file_number}. Extracting video frames")
        extracted_frames_paths = extract_video_frames(processing_queue, file_number, target_directory, AI_instance, video_path, selected_image_extension, cpu_number)
        total_frames_paths     = prepare_output_video_frame_filenames(extracted_frames_paths, selected_AI_model, frame_gen_factor, selected_image_extension)

    # 3. Frame generation
    write_process_status(processing_queue, f"{file_number}. Video frame generation")
    generate_video_frames(processing_queue, file_number, selected_AI_model, AI_instance, extracted_frames_paths, selected_image_extension)

    # 4. Video encoding
    write_process_status(processing_queue, f"{file_number}. Processing video")
    video_encoding(video_path, video_output_path, total_frames_paths, frame_gen_factor, slowmotion, cpu_number, selected_video_extension)
    copy_file_metadata(video_path, video_output_path)

    # 5. Video frames directory keep or delete
    if not selected_keep_frames: remove_dir(target_directory)

def generate_video_frames(
        processing_queue: multiprocessing_Queue,
        file_number: int,
        selected_AI_model: str,
        AI_instance: AI,
        frame_list_paths: list[str],
        selected_image_extension: str
        ) -> None:
    
    frame_processing_times = []

    for frame_index in range(len(frame_list_paths)-1):
        frame_1_path = frame_list_paths[frame_index]
        frame_2_path = frame_list_paths[frame_index + 1]
        base_path    = os_path_splitext(frame_1_path)[0]

        frame_gen_factor       = AI_instance.frame_gen_factor
        generated_frames_paths = prepare_generated_frames_paths(base_path, selected_AI_model, selected_image_extension, frame_gen_factor)
        already_generated      = are_frames_already_generated(generated_frames_paths)
        
        if already_generated == False:
            start_timer = timer()

            frame_1 = image_read(frame_1_path)
            frame_2 = image_read(frame_2_path)
            
            generated_images = AI_instance.AI_orchestration(frame_1, frame_2)
            thread = Thread(
                target = save_generated_video_frames,
                args   = (generated_frames_paths, generated_images)
            )
            thread.start()

            end_timer    = timer()
            elapsed_time = end_timer - start_timer
            frame_processing_times.append(elapsed_time)
            if (frame_index + 1) % 8 == 0:
                average_processing_time = numpy_mean(frame_processing_times)
                update_process_status_videos(processing_queue, file_number, frame_index, len(frame_list_paths), average_processing_time)

            if (frame_index + 1) % 100 == 0: frame_processing_times = []



# GUI utils function ---------------------------

def opengithub() -> None:   
    open_browser(githubme, new=1)

def opentelegram() -> None: 
    open_browser(telegramme, new=1)

def user_input_checks() -> None:
    global selected_file_list
    global selected_generation_option
    global selected_image_extension
    global resize_factor
    global cpu_number

    is_ready = True

    # files -------------------------------------------------
    try: selected_file_list = file_widget.get_selected_file_list()
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

def check_if_file_is_video(file: str) -> bool:
    return any(video_extension in file for video_extension in supported_video_extensions)

def check_supported_selected_files(uploaded_file_list: list) -> list:
    return [file for file in uploaded_file_list if any(supported_extension in file for supported_extension in supported_file_extensions)]

def show_error_message(exception: str) -> None:
    messageBox_title    = "Frame generation error"
    messageBox_subtitle = "Please report the error on Github or Telegram"
    messageBox_text     = f"\n {str(exception)} \n"

    MessageBox(
        messageType   = "error",
        title         = messageBox_title,
        subtitle      = messageBox_subtitle,
        default_value = None,
        option_list   = [messageBox_text]
    )

def open_files_action() -> None:
    info_message.set("Selecting files")

    uploaded_files_list    = list(filedialog.askopenfilenames())
    uploaded_files_counter = len(uploaded_files_list)

    supported_files_list    = check_supported_selected_files(uploaded_files_list)
    supported_files_counter = len(supported_files_list)
    
    print("> Uploaded files: " + str(uploaded_files_counter) + " => Supported files: " + str(supported_files_counter))

    if supported_files_counter > 0:
        global file_widget

        try:
            resize_factor = int(float(str(selected_resize_factor.get())))
        except:
            resize_factor = 0

        file_widget = FileWidget(
            master             = window, 
            selected_file_list = supported_files_list,
            resize_factor      = resize_factor,
            frame_generation_factor = selected_generation_option,
            fg_color = dark_color, 
            bg_color = dark_color
        )
        
        file_widget.place(
            relx = 0.0, 
            rely = 0.0, 
            relwidth  = 1.0, 
            relheight = 0.42
        )
        
        info_message.set("Ready")

    else: 
        info_message.set("Not supported files :(")

def open_output_path_action() -> None:
    asked_selected_output_path = filedialog.askdirectory()
    if asked_selected_output_path == "":
        selected_output_path.set(OUTPUT_PATH_CODED)
    else:
        selected_output_path.set(asked_selected_output_path)



# GUI select from menus functions ---------------------------

def select_AI_from_menu(selected_option: str) -> None:
    global selected_AI_model    
    selected_AI_model = selected_option

def select_framegeneration_option_from_menu(selected_option: str):
    global selected_generation_option    
    selected_generation_option = selected_option
    update_file_widget(1,2,3)

def select_gpu_from_menu(selected_option: str) -> None:
    global selected_gpu    
    selected_gpu = selected_option

def select_image_extension_from_menu(selected_option: str):
    global selected_image_extension    
    selected_image_extension = selected_option

def select_video_extension_from_menu(selected_option: str):
    global selected_video_extension   
    selected_video_extension = selected_option

def select_save_frame_from_menu(selected_option: str):
    global selected_keep_frames
    if   selected_option == 'Enabled': selected_keep_frames = True
    elif selected_option == 'Disabled': selected_keep_frames = False



# GUI info functions ---------------------------

def open_info_output_path():
    option_list = [
        "\n The default path is defined by the input files."
        + "\n For example uploading a file from the Download folder,"
        + "\n the app will save the generated files in the Download folder \n",

        " Otherwise it is possible to select the desired path using the SELECT button",
    ]

    MessageBox(
        messageType = "info",
        title       = "Output path",
        subtitle    = "This widget allows to choose upscaled files path",
        default_value = default_output_path,
        option_list   = option_list
    )

def open_info_AI_model():
    option_list = [
        "\n RIFE_4.17\n" + 
        "   • The complete RIFE AI model\n" + 
        "   • Excellent frame generation quality\n" + 
        "   • Recommended GPUs with VRAM >= 4GB\n",

        "\n RIFE_4.15_Lite\n" + 
        "   • Lightweight version of RIFE AI model\n" +
        "   • High frame generation quality\n" +
        "   • 10% faster than full model\n" + 
        "   • Use less GPU VRAM memory\n" +
        "   • Recommended for GPUs with VRAM < 4GB \n",
    ]
    
    MessageBox(
        messageType   = "info",
        title         = "AI model", 
        subtitle      = " This widget allows to choose between different RIFE models",
        default_value = default_AI_model,
        option_list   = option_list
    )

def open_info_frame_generation_option():
    option_list = [
        "\n FRAME GENERATION\n" + 
        "   • x2 - doubles video framerate • 30fps => 60fps\n" + 
        "   • x4 - quadruples video framerate • 30fps => 120fps\n" + 
        "   • x8 - octuplicate video framerate • 30fps => 240fps\n",

        "\n SLOWMOTION (no audio)\n" + 
        "   • Slowmotion x2 - slowmotion effect by a factor of 2\n" +
        "   • Slowmotion x4 - slowmotion effect by a factor of 4\n" +
        "   • Slowmotion x8 - slowmotion effect by a factor of 8\n"
    ]
    
    MessageBox(
        messageType   = "info",
        title         = "AI frame generation", 
        subtitle      = " This widget allows to choose between different AI frame generation option",
        default_value = default_generation_option,
        option_list   = option_list
    )

def open_info_gpu():
    option_list = [
        "\n It is possible to select up to 4 GPUs, via the index (also visible in the Task Manager):\n" +
        "  • GPU 1 (GPU 0 in Task manager)\n" + 
        "  • GPU 2 (GPU 1 in Task manager)\n" + 
        "  • GPU 3 (GPU 2 in Task manager)\n" + 
        "  • GPU 4 (GPU 3 in Task manager)\n",

        "\n NOTES\n" +
        "  • Keep in mind that the more powerful the chosen gpu is, the faster the upscaling will be\n" +
        "  • For optimal performance, it is essential to regularly update your GPUs drivers\n" +
        "  • Selecting the index of a GPU not present in the PC will cause the app to use the CPU for AI operations\n"+
        "  • In the case of a single GPU, select 'GPU 1'\n"
    ]

    MessageBox(
        messageType = "info",
        title       = "GPU",
        subtitle    = "This widget allows to select the GPU for AI upscale",
        default_value = default_gpu,
        option_list   = option_list
    )

def open_info_AI_output():
    option_list = [
        " \n PNG\n  • very good quality\n  • slow and heavy file\n  • supports transparent images\n",
        " \n JPG\n  • good quality\n  • fast and lightweight file\n",
    ]

    MessageBox(
        messageType   = "info",
        title         = "Image output",
        subtitle      = "This widget allows to choose the extension of generated frames",
        default_value = default_image_extension,
        option_list   = option_list
    )

def open_info_video_extension():
    option_list = [
        "\n MP4 (x264)\n" + 
        "   • produces well compressed video using x264 codec\n",

        "\n MP4 (x265)\n" + 
        "   • produces well compressed video using x265 codec\n",

        "\n AVI\n" + 
        "   • produces the highest quality video\n" +
        "   • the video produced can also be of large size\n"
    ]

    MessageBox(
        messageType   = "info",
        title         = "Video output",
        subtitle      = "This widget allows to choose the extension of the upscaled video",
        default_value = default_video_extension,
        option_list   = option_list
    )

def open_info_keep_frames():
    option_list = [
        "\n ENABLED \n FluidFrames.RIFE will create \n" +
        "   • the frame generated video \n" + 
        "   • a folder containing all original and generated frames \n",
        "\n DISABLED \n FluidFrames.RIFE will create only the frame generated video \n"
    ]

    MessageBox(
        messageType   = "info",
        title         = "Keep frames",
        subtitle      = "This widget allows to choose to save frames generated by the AI",
        default_value = "Enabled",
        option_list   = option_list
    )

def open_info_input_resolution():
    option_list = [
        " A high value (>70%) will create high quality videos but will be slower",
        " While a low value (<40%) will create good quality videos but will much faster",

        " \n For example, for a 1080p (1920x1080) video\n" + 
        " • Input resolution 25% => input to AI 270p (480x270)\n" +
        " • Input resolution 50% => input to AI 540p (960x540)\n" + 
        " • Input resolution 75% => input to AI 810p (1440x810)\n" + 
        " • Input resolution 100% => input to AI 1080p (1920x1080) \n",
    ]

    MessageBox(
        messageType   = "info",
        title         = "Input resolution %",
        subtitle      = "This widget allows to choose the resolution input to the AI",
        default_value = default_resize_factor,
        option_list   = option_list
    )

def open_info_cpu():
    option_list = [
        " When possible the app will use the number of cpus selected",

        "\n Currently this value is used for: \n" +
        "  • video frames extraction \n" +
        "  • video encoding \n",
    ]

    MessageBox(
        messageType   = "info",
        title         = "Cpu number",
        subtitle      = "This widget allows to choose how many cpus to devote to the app",
        default_value = default_cpu_number,
        option_list   = option_list
    )



# GUI place functions ---------------------------

def place_github_button():
    git_button = CTkButton(
        master     = window, 
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
        font          = bold11
    )  
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
    background = CTkLabel(
        master   = window,
        text     = "",
        fg_color = dark_color
    )

    text_drop = """ SUPPORTED FILES \n\n VIDEOS • mp4 webm mkv flv gif avi mov mpg qt 3gp """

    input_file_text = CTkLabel(
        master = window, 
        text       = text_drop,
        fg_color   = dark_color,
        bg_color   = dark_color,
        text_color = "#C0C0C0",
        width      = 300,
        height     = 150,
        font       = bold12,
        anchor     = "center"
    )
    
    input_file_button = CTkButton(
        master = window,
        command  = open_files_action, 
        text     = "SELECT FILES",
        width    = 140,
        height   = 30,
        font     = bold11,
        border_width = 1,
        fg_color     = "#282828",
        text_color   = "#E0E0E0",
        border_color = "#0096FF"
        )
    
    background.place(relx = 0.0, rely = 0.0, relwidth = 1.0, relheight = 0.42)
    input_file_text.place(relx = 0.5, rely = 0.20,  anchor = "center")
    input_file_button.place(relx = 0.5, rely = 0.35, anchor = "center")

def place_app_name():
    app_name_label = CTkLabel(
        master     = window,
        text       = app_name + " " + version,
        text_color = "#F08080",
        font       = bold19,
        anchor     = "w"
    )
    
    app_name_label.place(relx = column0_x, rely = row0_y - 0.03, anchor = "center")

def place_output_path_textbox():
    output_path_button  = create_info_button(open_info_output_path, "Output path", width = 300)
    output_path_textbox = create_text_box_output_path(selected_output_path) 
    select_output_path_button = create_active_button(
        command = open_output_path_action,
        text    = "SELECT",
        width   = 85,
        height  = 25
    )
  
    output_path_button.place(relx = column1_5_x, rely = row0_y - 0.05, anchor = "center")
    output_path_textbox.place(relx = column1_5_x, rely  = row0_y, anchor = "center")
    select_output_path_button.place(relx = column2_x, rely  = row0_y - 0.05, anchor = "center")

def place_AI_menu():
    AI_menu_button = create_info_button(open_info_AI_model, "AI model")
    AI_menu        = create_option_menu(select_AI_from_menu, AI_models_list, default_AI_model)

    AI_menu_button.place(relx = column0_x, rely = row1_y - 0.053, anchor = "center")
    AI_menu.place(relx = column0_x, rely = row1_y, anchor = "center")

def place_generation_option_menu():
    fluidity_option_button = create_info_button(open_info_frame_generation_option, "AI frame generation")
    fluidity_option_menu   = create_option_menu(select_framegeneration_option_from_menu, generation_options_list, default_generation_option)

    fluidity_option_button.place(relx = column0_x, rely = row2_y - 0.05, anchor = "center")
    fluidity_option_menu.place(relx = column0_x, rely = row2_y, anchor = "center")

def place_gpu_menu():
    gpu_button = create_info_button(open_info_gpu, "GPU")
    gpu_menu   = create_option_menu(select_gpu_from_menu, gpus_list, default_gpu)
    
    gpu_button.place(relx = column1_x, rely = row1_y - 0.053, anchor = "center")
    gpu_menu.place(relx = column1_x, rely  = row1_y, anchor = "center")

def place_image_output_menu():
    file_extension_button = create_info_button(open_info_AI_output, "Image output")
    file_extension_menu   = create_option_menu(select_image_extension_from_menu, image_extension_list, default_image_extension)
    
    file_extension_button.place(relx = column2_x, rely = row1_y - 0.05, anchor = "center")
    file_extension_menu.place(relx = column2_x, rely = row1_y, anchor = "center")

def place_video_extension_menu():
    video_extension_button = create_info_button(open_info_video_extension, "Video output")
    video_extension_menu   = create_option_menu(select_video_extension_from_menu, video_extension_list, default_video_extension)
    
    video_extension_button.place(relx = column2_x, rely = row2_y - 0.05, anchor = "center")
    video_extension_menu.place(relx = column2_x, rely = row2_y, anchor = "center")

def place_keep_frames_menu():
    keep_frames_button = create_info_button(open_info_keep_frames, "Keep frames")
    keep_frames_menu   = create_option_menu(select_save_frame_from_menu, keep_frames_list, default_keep_frames)
    
    keep_frames_button.place(relx = column1_x, rely = row2_y - 0.053, anchor = "center")
    keep_frames_menu.place(relx = column1_x, rely = row2_y, anchor = "center")

def place_input_resolution_textbox():
    resize_factor_button  = create_info_button(open_info_input_resolution, "Input resolution %")
    resize_factor_textbox = create_text_box(selected_resize_factor) 

    resize_factor_button.place(relx = column0_x, rely = row3_y - 0.05, anchor = "center")
    resize_factor_textbox.place(relx = column0_x, rely = row3_y, anchor = "center")

def place_cpu_textbox():
    cpu_button  = create_info_button(open_info_cpu, "CPU number")
    cpu_textbox = create_text_box(selected_cpu_number)

    cpu_button.place(relx = column1_x, rely = row3_y - 0.05, anchor = "center")
    cpu_textbox.place(relx = column1_x, rely  = row3_y, anchor = "center")

def place_message_label():
    message_label = CTkLabel(
        master  = window, 
        textvariable = info_message,
        height       = 25,
        font         = bold11,
        fg_color     = "#ffbf00",
        text_color   = "#000000",
        anchor       = "center",
        corner_radius = 12
    )
    message_label.place(relx = column2_x, rely = row4_y - 0.075, anchor = "center")

def place_stop_button(): 
    stop_button = create_active_button(
        command = stop_button_command,
        text    = "STOP",
        icon    = stop_icon,
        width   = 140,
        height  = 30,
        border_color = "#EC1D1D"
    )
    stop_button.place(relx = column2_x, rely = row4_y, anchor = "center")

def place_generation_button(): 
    generation_button = create_active_button(
        command = generate_button_command,
        text    = "GENERATE",
        icon    = play_icon,
        width   = 140,
        height  = 30
    )
    generation_button.place(relx = column2_x, rely = row4_y, anchor = "center")



# Main functions ---------------------------

def on_app_close():
    window.grab_release()
    window.destroy()

    global selected_AI_model
    global selected_generation_option
    global selected_gpu
    
    global selected_keep_frames
    global selected_image_extension
    global selected_video_extension
    global resize_factor
    global cpu_number

    AI_model_to_save          = f"{selected_AI_model}"
    generation_option_to_save = f"{selected_generation_option}"
    gpu_to_save               = f"{selected_gpu}"
    keep_frames_to_save       = "Enabled" if selected_keep_frames == True else "Disabled"
    image_extension_to_save   = f"{selected_image_extension}"
    video_extension_to_save   = f"{selected_video_extension}"

    user_preference = {
        "default_AI_model":          AI_model_to_save,
        "default_generation_option": generation_option_to_save,
        "default_gpu":               gpu_to_save,
        "default_image_extension":   image_extension_to_save,
        "default_video_extension":   video_extension_to_save,
        "default_keep_frames":       keep_frames_to_save,
        "default_output_path":       selected_output_path.get(),
        "default_resize_factor":     str(selected_resize_factor.get()),
        "default_cpu_number":        str(selected_cpu_number.get()),
    }
    user_preference_json = json_dumps(user_preference)
    with open(USER_PREFERENCE_PATH, "w") as preference_file:
        preference_file.write(user_preference_json)

    stop_generation_process()

class App():
    def __init__(self, window):
        self.toplevel_window = None
        window.protocol("WM_DELETE_WINDOW", on_app_close)

        window.title('')
        window.geometry("675x675")
        window.resizable(False, False)
        window.iconbitmap(find_by_relative_path("Assets" + os_separator + "logo.ico"))

        place_github_button()
        place_telegram_button()

        place_app_name()
        place_output_path_textbox()

        place_AI_menu()
        place_generation_option_menu()
        place_input_resolution_textbox()

        place_gpu_menu()
        place_keep_frames_menu()
        place_cpu_textbox()

        place_image_output_menu()
        place_video_extension_menu()

        place_message_label()
        place_generation_button()

        place_loadFile_section()

if __name__ == "__main__":
    multiprocessing_freeze_support()

    processing_queue = multiprocessing_Queue(maxsize=1)

    set_appearance_mode("Dark")
    set_default_color_theme("dark-blue")

    window = CTk() 

    info_message           = StringVar()
    selected_output_path   = StringVar()
    selected_resize_factor = StringVar()
    selected_cpu_number    = StringVar()

    global selected_file_list
    global selected_AI_model
    global selected_generation_option
    global selected_gpu 
    global selected_keep_frames
    global resize_factor
    global cpu_number
    global selected_image_extension
    global selected_video_extension

    selected_file_list = []

    selected_AI_model          = default_AI_model
    selected_generation_option = default_generation_option
    selected_gpu               = default_gpu
    selected_image_extension   = default_image_extension
    selected_video_extension   = default_video_extension

    selected_keep_frames = True if default_keep_frames == "Enabled" else False

    selected_resize_factor.set(default_resize_factor)
    selected_cpu_number.set(default_cpu_number)
    selected_output_path.set(default_output_path)

    info_message.set("Hi :)")
    selected_resize_factor.trace_add('write', update_file_widget)

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