
# Standard library imports
import sys
from functools  import cache
from time       import sleep
from webbrowser import open as open_browser
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

from os import (
    sep       as os_separator,
    devnull   as os_devnull,
    environ   as os_environ,
    cpu_count as os_cpu_count,
    makedirs  as os_makedirs,
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
from PIL.Image import (
    open      as pillow_image_open,
    fromarray as pillow_image_fromarray
)

from moviepy.editor   import VideoFileClip 
from moviepy.video.io import ImageSequenceClip 

from onnx        import load as onnx_load 
from onnxruntime import (
    InferenceSession as onnxruntime_inferenceSession
)

from cv2 import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    COLOR_BGR2RGB,
    IMREAD_UNCHANGED,
    INTER_LINEAR,
    VideoCapture as opencv_VideoCapture,
    cvtColor     as opencv_cvtColor,
    imdecode     as opencv_imdecode,
    imencode     as opencv_imencode,
    cvtColor     as opencv_cvtColor,
    resize       as opencv_resize,
)

from numpy import (
    ndarray           as numpy_ndarray,
    ascontiguousarray as numpy_ascontiguousarray,
    frombuffer        as numpy_frombuffer,
    concatenate       as numpy_concatenate, 
    transpose         as numpy_transpose,
    expand_dims       as numpy_expand_dims,
    squeeze           as numpy_squeeze,
    clip              as numpy_clip,
    mean              as numpy_mean,
    concatenate       as numpy_concatenate,
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
    set_default_color_theme,
)

if sys.stdout is None: sys.stdout = open(os_devnull, "w")
if sys.stderr is None: sys.stderr = open(os_devnull, "w")

def find_by_relative_path(relative_path: str) -> str:
    base_path = getattr(sys, '_MEIPASS', os_path_dirname(os_path_abspath(__file__)))
    return os_path_join(base_path, relative_path)



app_name   = "FluidFrames"
version    = "3.6"
dark_color = "#080808"

githubme   = "https://github.com/Djdefrag/FluidFrames.RIFE"
telegramme = "https://linktr.ee/j3ngystudio"

AI_models_list                = [ 'RIFE_4.17', 'RIFE_4.15_Lite' ]
frame_generation_options_list = [ 'x2', 'x4', 'x8', 'x2-slowmotion', 'x4-slowmotion', 'x8-slowmotion' ]
gpus_list                     = [ 'GPU 1', 'GPU 2', 'GPU 3', 'GPU 4' ]
image_extension_list          = [ '.jpg', '.png', '.bmp', '.tiff' ]
video_extension_list          = [ '.mp4 (x264)', '.mp4 (x265)', '.avi' ]
save_frames_list              = [ 'Enabled', 'Disabled' ]
AI_precision_list             = [ 'Full precision', 'Half precision' ]

default_AI_model                = AI_models_list[0]
default_gpu                     = gpus_list[0]
default_frame_generation_option = frame_generation_options_list[0]
default_image_extension         = image_extension_list[0]
default_video_extension         = video_extension_list[0]
default_save_frames             = save_frames_list[0]
default_output_path             = "Same path as input files"
default_resize_factor           = str(50)
default_VRAM_limiter            = str(8)
default_cpu_number              = str(int(os_cpu_count()/2))

FFMPEG_EXE_PATH   = find_by_relative_path(f"Assets{os_separator}ffmpeg.exe")
FRAMES_FOR_CPU    = 30

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
    '.mp4', '.MP4',
    '.webm', '.WEBM',
    '.mkv', '.MKV',
    '.flv', '.FLV',
    '.gif', '.GIF',
    '.m4v', ',M4V',
    '.avi', '.AVI',
    '.mov', '.MOV',
    '.qt', '.3gp', 
    '.mpg', '.mpeg'
    ]

supported_video_extensions = [
    '.mp4', '.MP4',
    '.webm', '.WEBM',
    '.mkv', '.MKV',
    '.flv', '.FLV',
    '.gif', '.GIF',
    '.m4v', ',M4V',
    '.avi', '.AVI',
    '.mov', '.MOV',
    '.qt', '.3gp', 
    '.mpg', '.mpeg'
    ]



# ------------------ AI ------------------

def load_AI_model(
        selected_AI_model: str, 
        selected_gpu: str,
    ) -> onnxruntime_inferenceSession:

    AI_model_path   = find_by_relative_path(f"AI-onnx{os_separator}{selected_AI_model}.onnx")
    AI_model_loaded = onnx_load(AI_model_path)

    match selected_gpu:
        case 'GPU 1':
            backend = [('DmlExecutionProvider', {"device_id": "0"})]
        case 'GPU 2':
            backend = [('DmlExecutionProvider', {"device_id": "1"})]
        case 'GPU 3':
            backend = [('DmlExecutionProvider', {"device_id": "2"})]
        case 'GPU 4':
            backend = [('DmlExecutionProvider', {"device_id": "3"})]
    
    AI_model = onnxruntime_inferenceSession(
        path_or_bytes = AI_model_loaded.SerializeToString(), 
        providers     = backend
    )    

    return AI_model

def concatenate_frames(
        frame_1: numpy_ndarray, 
        frame_2: numpy_ndarray, 
    ) -> numpy_ndarray:

    frame_1 = frame_1 / 255
    frame_2 = frame_2 / 255

    input_images = numpy_concatenate((frame_1, frame_2), axis=2)

    return input_images

def preprocess_image(image: numpy_ndarray) -> numpy_ndarray:
    image_transposed          = numpy_transpose(image, (2, 0, 1))
    image_transposed_expanded = numpy_expand_dims(image_transposed, axis=0)

    return image_transposed_expanded

def process_image_with_AI_model(
        AI_model: onnxruntime_inferenceSession, 
        image: numpy_ndarray
        ) -> numpy_ndarray:
    
    onnx_input      = {AI_model.get_inputs()[0].name: image}
    onnx_output     = AI_model.run(None, onnx_input)[0] 
    output_squeezed = numpy_squeeze(onnx_output, axis=0)
    output_squeezed_clamped = numpy_clip(output_squeezed, 0, 1)
    output_squeezed_clamped_transposed = numpy_transpose(output_squeezed_clamped, (1, 2, 0)).astype(float32)

    return output_squeezed_clamped_transposed

def AI_interpolation(
        AI_model: onnxruntime_inferenceSession, 
        frame_1: numpy_ndarray,
        frame_2: numpy_ndarray
    ) -> numpy_ndarray:

    concatenated_frames = concatenate_frames(frame_1, frame_2).astype(float32)
    AI_input  = numpy_ascontiguousarray(concatenated_frames)
    AI_input  = preprocess_image(AI_input)
    AI_output = process_image_with_AI_model(AI_model, AI_input)
    output_frame = (AI_output * 255).astype(uint8)
    
    return output_frame

def save_multiple_frames_async(
        frames_to_save: list[numpy_ndarray],
        frame_paths_to_save: list[str]
    ) -> None:
    
    for index in range(len(frames_to_save)):
        image_write(
            frame_paths_to_save[index], 
            frames_to_save[index]
        )

def AI_generate_frames(
        AI_model: onnxruntime_inferenceSession,
        frame1: numpy_ndarray, 
        frame2: numpy_ndarray, 
        frame_1_name: str,
        frame_2_name: str,
        frame_base_name: str,
        all_video_frames_path_list: list,
        selected_image_extension: str, 
        fluidification_factor: int
        ) -> list:
    
    frames_to_generate = fluidification_factor - 1

    frames_to_save = []
    frame_paths_to_save = []

    if frames_to_generate == 1: 
        # fluidification x2
        frame_1_1_name = f"{frame_base_name}_.1{selected_image_extension}"
        frame_1_1 = AI_interpolation(AI_model, frame1, frame2)        
        
        frames_to_save.extend([frame1, frame_1_1, frame2])
        frame_paths_to_save.extend([frame_1_name, frame_1_1_name, frame_2_name])
        all_video_frames_path_list.extend([frame_1_name, frame_1_1_name, frame_2_name])

    elif frames_to_generate == 3: 
        # fluidification x4
        frame_1_1_name = f"{frame_base_name}_.1{selected_image_extension}"
        frame_1_2_name = f"{frame_base_name}_.2{selected_image_extension}"
        frame_1_3_name = f"{frame_base_name}_.3{selected_image_extension}"

        frame_1_2 = AI_interpolation(AI_model, frame1, frame2)
        frame_1_1 = AI_interpolation(AI_model, frame1, frame_1_2)
        frame_1_3 = AI_interpolation(AI_model, frame_1_2, frame2)

        frames_to_save.extend([frame1, frame_1_1, frame_1_2, frame_1_3, frame2])
        frame_paths_to_save.extend([frame_1_name, frame_1_1_name, frame_1_2_name, frame_1_3_name, frame_2_name])
        all_video_frames_path_list.extend([frame_1_name, frame_1_1_name, frame_1_2_name, frame_1_3_name, frame_2_name])

    elif frames_to_generate == 7: 
        # fluidification x8
        frame_1_1_name = f"{frame_base_name}_.1{selected_image_extension}"
        frame_1_2_name = f"{frame_base_name}_.2{selected_image_extension}"
        frame_1_3_name = f"{frame_base_name}_.3{selected_image_extension}"
        frame_1_4_name = f"{frame_base_name}_.4{selected_image_extension}"
        frame_1_5_name = f"{frame_base_name}_.5{selected_image_extension}"
        frame_1_6_name = f"{frame_base_name}_.6{selected_image_extension}"
        frame_1_7_name = f"{frame_base_name}_.7{selected_image_extension}"

        frame_1_4 = AI_interpolation(AI_model, frame1, frame2)
        frame_1_2 = AI_interpolation(AI_model, frame1, frame_1_4)
        frame_1_1 = AI_interpolation(AI_model, frame1, frame_1_2)
        frame_1_3 = AI_interpolation(AI_model, frame_1_2, frame_1_4)
        frame_1_6 = AI_interpolation(AI_model, frame_1_4, frame2)
        frame_1_5 = AI_interpolation(AI_model, frame_1_4, frame_1_6)
        frame_1_7 = AI_interpolation(AI_model, frame_1_6, frame2)

        frames_to_save.extend([frame1, frame_1_1, frame_1_2, frame_1_3, frame_1_4, frame_1_5, frame_1_6, frame_1_7, frame2])
        frame_paths_to_save.extend([frame_1_name, frame_1_1_name, frame_1_2_name, 
                                    frame_1_3_name, frame_1_4_name, frame_1_5_name, 
                                    frame_1_6_name, frame_1_7_name, frame_2_name])
        all_video_frames_path_list.extend([frame_1_name, frame_1_1_name, frame_1_2_name, 
                                    frame_1_3_name, frame_1_4_name, frame_1_5_name, 
                                    frame_1_6_name, frame_1_7_name, frame_2_name])

    thread = Thread(
        target = save_multiple_frames_async,
        args = (
            frames_to_save, 
            frame_paths_to_save
        )
    )
    thread.start()

    return all_video_frames_path_list



# GUI utils ---------------------------

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

class ScrollableImagesTextFrame_framegeneration(CTkScrollableFrame):

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

        self.file_list      = selected_file_list
        self.resize_factor  = resize_factor
        self.frame_generation_factor = frame_generation_factor

        self.label_list = []
        self._create_widgets()

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
        label.grid(row = index_row, column = 0, 
                   pady = (3, 3), padx = (3, 3), 
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
        
    def get_selected_file_list(self) -> list: 
        return self.file_list  

    def set_frame_generation_factor(self, frame_generation_factor) -> None:
        self.frame_generation_factor = frame_generation_factor

    def set_resize_factor(self, resize_factor) -> None:
        self.resize_factor = resize_factor

    @cache
    def extract_file_icon(self, file_path) -> CTkImage:
        max_size = 50

        if check_if_file_is_video(file_path):
            cap = opencv_VideoCapture(file_path)
            _, frame = cap.read()
            frame = opencv_cvtColor(frame, COLOR_BGR2RGB)
            ratio = min(max_size / frame.shape[0], max_size / frame.shape[1])
            icon = CTkImage(
                pillow_image_fromarray(frame, mode="RGB"), 
                size = (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio))
            )
            cap.release()
            return icon

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
            video_icon = self.extract_file_icon(file_path)

            match self.resize_factor:
                case 0:
                    video_infos = (
                        f"{video_name}\n"
                        f"{minutes}m:{round(seconds)}s • {num_frames}frames • {width}x{height} • {round(frame_rate, 2)}fps"
                    )
                    return video_infos, video_icon
                case _:
                    resized_height = int(height * (self.resize_factor/100))
                    resized_width  = int(width * (self.resize_factor/100))

                    if   "x2" in self.frame_generation_factor: 
                        generation_factor = 2
                    elif "x4" in self.frame_generation_factor: 
                        generation_factor = 4
                    elif "x8" in self.frame_generation_factor: 
                        generation_factor = 8

                    frame_generated_fps = frame_rate * generation_factor

                    if "slowmotion" in self.frame_generation_factor:
                        slowmotion = True
                    else:
                        slowmotion = False

                    if slowmotion:
                        duration_slowmotion = (num_frames/frame_rate) * generation_factor
                        minutes_slowmotion  = int(duration_slowmotion/60)
                        seconds_slowmotion  = duration_slowmotion % 60
                        video_infos = (
                            f"{video_name}\n"
                            f"{minutes}m:{round(seconds)}s • {num_frames}frames • {width}x{height} : {round(frame_rate, 2)}fps\n"
                            f"AI input {resized_width}x{resized_height} ➜ {resized_width}x{resized_height} : {round(frame_rate, 2)}fps : {minutes_slowmotion}m:{round(seconds_slowmotion)}s"
                        )
                    else:
                        video_infos = (
                            f"{video_name}\n"
                            f"{minutes}m:{round(seconds)}s • {num_frames}frames • {width}x{height} : {round(frame_rate, 2)}fps\n"
                            f"AI input {resized_width}x{resized_height} ➜ {resized_width}x{resized_height} : {round(frame_generated_fps, 2)}fps"
                        )

                    return video_infos, video_icon

    def _destroy_(self) -> None:
        self.file_list = []
        self.destroy()
        place_loadFile_section()

    def clean_file_list(self) -> None:
        for label in self.label_list:
            label.grid_forget()

def update_file_widget(a, b, c) -> None:
    try:
        global scrollable_frame_file_list
        scrollable_frame_file_list
    except:
        return
        
    try:
        resize_factor = int(float(str(selected_resize_factor.get())))
    except:
        resize_factor = 0
    
    scrollable_frame_file_list.clean_file_list()
    scrollable_frame_file_list.set_resize_factor(resize_factor)
    scrollable_frame_file_list.set_frame_generation_factor(selected_frame_generation_option)
    scrollable_frame_file_list._create_widgets()

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

def create_option_menu(command: Callable, values: list) -> CTkOptionMenu:
    
    return CTkOptionMenu(
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

def create_text_box(textvariable: StringVar) -> CTkEntry:
    
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

def create_text_box_output_path(
        textvariable: StringVar
        ) -> CTkEntry:

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
        file_data      = file.read()
        numpy_buffer   = numpy_frombuffer(file_data, uint8)
        opencv_decoded = opencv_imdecode(numpy_buffer, flags)
        return opencv_decoded

def remove_dir(name_dir: str) -> None:
    if os_path_exists(name_dir): remove_directory(name_dir)

def create_dir(name_dir: str) -> None:
    if os_path_exists(name_dir):     remove_directory(name_dir)
    if not os_path_exists(name_dir): os_makedirs(name_dir, mode=0o777)



# Image/video Utils functions ------------------------

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
        frame_2_resized = opencv_resize(frame_2, (target_width, target_height), interpolation = INTER_LINEAR)
        return frame_1_resized, frame_2_resized

def extract_video_fps(
        video_path: str
        ) -> float:
    
    video_capture = opencv_VideoCapture(video_path)
    frame_rate    = video_capture.get(CAP_PROP_FPS)
    video_capture.release()
    return frame_rate

def extract_video_frames_and_audio(
        processing_queue: multiprocessing_Queue,
        file_number: int,
        target_directory: str,
        video_path: str, 
        selected_image_extension: str,
        cpu_number: int
    ) -> tuple[list[str], str]:

    create_dir(target_directory)

    # Audio extraction
    with VideoFileClip(video_path) as video_file_clip:
        try: 
            write_process_status(processing_queue, f"{file_number}. Extracting video audio")
            audio_path = f"{target_directory}{os_separator}audio.mp3"
            video_file_clip.audio.write_audiofile(audio_path, verbose = False, logger = None)
        except:
            pass

    # Video frame extraction
    frames_number_to_save = cpu_number * FRAMES_FOR_CPU

    video_capture = opencv_VideoCapture(video_path)
    frame_count   = int(video_capture.get(CAP_PROP_FRAME_COUNT))

    frames_to_save      = []
    frames_path_to_save = []
    video_frames_list   = []

    for frame_number in range(frame_count):
        success, frame = video_capture.read()
        if success:
            frames_to_save.append(frame)
            
            frame_path = f"{target_directory}{os_separator}frame_{frame_number:03d}{selected_image_extension}"
            frames_path_to_save.append(frame_path)
            video_frames_list.append(frame_path)

            if len(frames_to_save) == frames_number_to_save:
                percentage_extraction = (frame_number / frame_count) * 100
                write_process_status(processing_queue, f"{file_number}. Extracting video frames ({round(percentage_extraction, 2)}%)")

                pool = ThreadPool(cpu_number)
                pool.starmap(image_write, zip(frames_path_to_save, frames_to_save))
                pool.close()
                pool.join()
                frames_to_save      = []
                frames_path_to_save = []

    video_capture.release()

    if len(frames_to_save) > 0:
        pool = ThreadPool(cpu_number)
        pool.starmap(image_write, zip(frames_path_to_save, frames_to_save))
        pool.close()
        pool.join()
    
    return video_frames_list, audio_path

def video_reconstruction_by_frames(
        video_path: str, 
        audio_path: str,
        selected_output_path: str,
        all_video_frames_paths: list, 
        selected_AI_model: str,
        fluidification_factor: int,
        slowmotion: bool,
        resize_factor: int,
        cpu_number: int,
        selected_video_extension: str
        ) -> None:
    
    frame_rate = extract_video_fps(video_path)

    if not slowmotion: frame_rate = frame_rate * fluidification_factor

    match selected_video_extension:
        case '.mp4 (x264)':
            selected_video_extension = '.mp4'
            codec = 'libx264'
        case '.mp4 (x265)':
            selected_video_extension = '.mp4'
            codec = 'libx265'
        case '.avi':
            selected_video_extension = '.avi'
            codec = 'png'

    output_path = prepare_output_video_filename(
        video_path, 
        selected_output_path,
        selected_AI_model,
        fluidification_factor, 
        slowmotion, 
        resize_factor, 
        selected_video_extension
    )

    clip = ImageSequenceClip.ImageSequenceClip(all_video_frames_paths, fps = frame_rate)
    if slowmotion:
        clip.write_videofile(
            output_path,
            fps     = frame_rate,
            codec   = codec,
            bitrate = '16M',
            verbose = False,
            logger  = None,
            threads = cpu_number,
            preset  = "ultrafast"
        ) 
    else:
        clip.write_videofile(
            output_path,
            fps     = frame_rate,
            audio   = audio_path if os_path_exists(audio_path) else None,
            codec   = codec,
            bitrate = '16M',
            verbose = False,
            logger  = None,
            threads = cpu_number,
            preset  = "ultrafast"
        )    

def calculate_time_to_complete_video(
        time_for_frame: float,
        remaining_frames: int,
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
        percent_complete = (frame_index + 1) / how_many_frames * 100 

        time_left = calculate_time_to_complete_video(
            time_for_frame   = average_processing_time,
            remaining_frames = how_many_frames - frame_index
        )
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
                stop_thread()

            elif actual_step == STOP_STATUS:
                info_message.set(f"Fluidify stopped")
                stop_fluidify_process()
                stop_thread()

            elif ERROR_STATUS in actual_step:
                error_message = f"Error during fluidify process :("
                error = actual_step.replace(ERROR_STATUS, "")
                info_message.set(error_message)
                show_error_message(error)
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
    global process_frame_generation_orchestrator

    try:
        process_frame_generation_orchestrator
    except:
        pass
    else:
        process_frame_generation_orchestrator.terminate()
        process_frame_generation_orchestrator.join()

def stop_button_command() -> None:
    stop_fluidify_process()
    write_process_status(processing_queue, f"{STOP_STATUS}")

def prepare_output_video_filename(
        video_path: str, 
        selected_output_path: str,
        selected_AI_model: str,
        fluidification_factor: int, 
        slowmotion: bool, 
        resize_factor: int, 
        selected_video_extension: str
        ) -> str:
    
    if selected_output_path == default_output_path:
        file_path_no_extension, _ = os_path_splitext(video_path)
        output_path = file_path_no_extension
    else:
        file_name   = os_path_basename(video_path)
        output_path = f"{selected_output_path}{os_separator}{file_name}"

    # Selected AI model
    to_append = f"_{selected_AI_model}x{str(fluidification_factor)}"

    # Slowmotion?
    if slowmotion: to_append += f"_slowmo_"

    # Selected resize
    to_append += f"_Resize-{str(int(resize_factor * 100))}"

    # Video output
    to_append += f"{selected_video_extension}"

    output_path += to_append

    return output_path

def prepare_output_video_frames_directory_name(
        video_path: str, 
        selected_output_path: str,
        selected_AI_model: str,
        fluidification_factor: int, 
        slowmotion: bool, 
        resize_factor: int, 
        ) -> str:
    
    if selected_output_path == default_output_path:
        file_path_no_extension, _ = os_path_splitext(video_path)
        output_path = file_path_no_extension
    else:
        file_name   = os_path_basename(video_path)
        output_path = f"{selected_output_path}{os_separator}{file_name}"

    # Selected AI model
    to_append = f"_{selected_AI_model}x{str(fluidification_factor)}"

    # Slowmotion?
    if slowmotion: to_append += f"_slowmo_"

    # Selected resize
    to_append += f"_Resize-{str(int(resize_factor * 100))}"

    output_path += to_append

    return output_path

def get_video_target_resolution(
        first_video_frame: numpy_ndarray, 
        resize_factor: int
        ) -> tuple:
    
    temp_frame    = image_read(first_video_frame)
    target_height = int(temp_frame.shape[0] * resize_factor)
    target_width  = int(temp_frame.shape[1] * resize_factor) 

    return target_height, target_width

def check_fluidification_option(
        selected_frame_generation_option: str
        ) -> tuple:
    
    slowmotion = False
    fluidification_factor = 0

    if 'slowmotion' in selected_frame_generation_option: slowmotion = True

    if   '2' in selected_frame_generation_option: fluidification_factor = 2
    elif '4' in selected_frame_generation_option: fluidification_factor = 4
    elif '8' in selected_frame_generation_option: fluidification_factor = 8

    return fluidification_factor, slowmotion

def fludify_button_command() -> None: 
    global selected_file_list
    global selected_AI_model
    global selected_frame_generation_option
    global selected_gpu
    global selected_image_extension
    global selected_video_extension
    global resize_factor
    global cpu_number
    global selected_save_frames

    global process_frame_generation_orchestrator
    
    if user_input_checks():
        info_message.set("Loading")

        print("=" * 50)
        print(f"> Starting fluidify:")
        print(f"   Files to fluidify: {len(selected_file_list)}")
        print(f"   Output path: {(selected_output_path.get())}")
        print(f"   Selected AI model: {selected_AI_model}")
        print(f"   Selected fluidify option: {selected_frame_generation_option}")
        print(f"   Selected image output extension: {selected_image_extension}")
        print(f"   Selected video output extension: {selected_video_extension}")
        print(f"   Resize factor: {int(resize_factor * 100)}%")
        print(f"   Cpu number: {cpu_number}")
        print(f"   Save frames: {selected_save_frames}")
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
                selected_frame_generation_option, 
                selected_image_extension, 
                selected_video_extension, 
                resize_factor, 
                cpu_number, 
                selected_save_frames
            )
        )
        process_frame_generation_orchestrator.start()

        thread_wait = Thread(
            target = check_fluidify_steps
        )
        thread_wait.start()

def frame_generation_orchestrator(
        processing_queue: multiprocessing_Queue,
        selected_file_list: list,
        selected_output_path: str,
        selected_AI_model: str,
        selected_gpu: str,
        selected_frame_generation_option: str,
        selected_image_extension: str,
        selected_video_extension: str,
        resize_factor: int,
        cpu_number: int,
        selected_save_frames: bool
        ) -> None:
            
    fluidification_factor, slowmotion = check_fluidification_option(selected_frame_generation_option)

    try:
        write_process_status(processing_queue, f"Loading AI model")

        AI_model = load_AI_model(selected_AI_model, selected_gpu)

        how_many_files = len(selected_file_list)
        for file_number in range(how_many_files):
            file_path   = selected_file_list[file_number]
            file_number = file_number + 1

            video_frame_generation(
                processing_queue,
                file_path, 
                file_number,
                selected_output_path,
                AI_model,
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

def video_frame_generation(
        processing_queue: multiprocessing_Queue,
        video_path: str, 
        file_number: int,
        selected_output_path: str,
        AI_model: onnxruntime_inferenceSession,
        selected_AI_model: str,
        fluidification_factor: int, 
        slowmotion: bool, 
        resize_factor: int, 
        selected_image_extension: str, 
        selected_video_extension: str,
        cpu_number: int, 
        selected_save_frames: bool
        ) -> None:
        
    target_directory = prepare_output_video_frames_directory_name(
        video_path, 
        selected_output_path,
        selected_AI_model, 
        fluidification_factor, 
        slowmotion,
        resize_factor
    )
    
    write_process_status(processing_queue, f"{file_number}. Extracting video frames")
    frame_list_paths, audio_path = extract_video_frames_and_audio(
        processing_queue, 
        file_number, 
        target_directory, 
        video_path, 
        selected_image_extension,
        cpu_number
    )

    target_height, target_width = get_video_target_resolution(
        frame_list_paths[0], 
        resize_factor
    )  

    write_process_status(processing_queue, f"{file_number}. Video frame generation")
    all_video_frames_paths = []
    frame_processing_times = []
    how_many_frames = len(frame_list_paths)

    for frame_index in range(how_many_frames-1):

        start_timer = timer()

        frame_base_name = os_path_splitext(frame_list_paths[frame_index])[0]
        frame_1_name = frame_list_paths[frame_index]
        frame_2_name = frame_list_paths[frame_index + 1]

        frame_1, frame_2 = resize_frames(
            frame_1 = image_read(frame_list_paths[frame_index]), 
            frame_2 = image_read(frame_list_paths[frame_index + 1]), 
            resize_factor = resize_factor, 
            target_width  = target_width, 
            target_height = target_height
        )

        all_video_frames_paths = AI_generate_frames(
            AI_model,
            frame_1, 
            frame_2, 
            frame_1_name,
            frame_2_name,
            frame_base_name,
            all_video_frames_paths, 
            selected_image_extension, 
            fluidification_factor
        )

        frame_processing_times.append(timer() - start_timer)
        average_processing_time = float(numpy_mean(frame_processing_times))
    
        update_process_status_videos(
            processing_queue = processing_queue, 
            file_number      = file_number,  
            frame_index      = frame_index, 
            how_many_frames  = how_many_frames,
            average_processing_time = average_processing_time
        )

    all_video_frames_paths = list(dict.fromkeys(all_video_frames_paths))

    # Video reconstruction
    write_process_status(processing_queue, f"{file_number}. Processing fluidified video")  
    video_reconstruction_by_frames(
        video_path,
        audio_path,
        selected_output_path,
        all_video_frames_paths, 
        selected_AI_model,
        fluidification_factor, 
        slowmotion, 
        resize_factor, 
        cpu_number, 
        selected_video_extension
    )

    if not selected_save_frames: remove_dir(target_directory)



# GUI utils function ---------------------------

def opengithub() -> None:   
    open_browser(githubme, new=1)

def opentelegram() -> None: 
    open_browser(telegramme, new=1)

def user_input_checks() -> None:
    global selected_file_list
    global selected_frame_generation_option
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

def open_files_action():
    info_message.set("Selecting files")

    uploaded_files_list    = list(filedialog.askopenfilenames())
    uploaded_files_counter = len(uploaded_files_list)

    supported_files_list    = check_supported_selected_files(uploaded_files_list)
    supported_files_counter = len(supported_files_list)
    
    print("> Uploaded files: " + str(uploaded_files_counter) + " => Supported files: " + str(supported_files_counter))

    if supported_files_counter > 0:
        global scrollable_frame_file_list

        try:
            resize_factor = int(float(str(selected_resize_factor.get())))
        except:
            resize_factor = 0

        scrollable_frame_file_list = ScrollableImagesTextFrame_framegeneration(
            master             = window, 
            selected_file_list = supported_files_list,
            resize_factor      = resize_factor,
            frame_generation_factor = selected_frame_generation_option,
            fg_color = dark_color, 
            bg_color = dark_color
        )
        
        scrollable_frame_file_list.place(
            relx = 0.0, 
            rely = 0.0, 
            relwidth  = 1.0, 
            relheight = 0.42
        )
        
        info_message.set("Ready")

    else: 
        info_message.set("Not supported files :(")

def open_output_path_action():
    asked_selected_output_path = filedialog.askdirectory()
    if asked_selected_output_path == "":
        selected_output_path.set(default_output_path)
    else:
        selected_output_path.set(asked_selected_output_path)



# GUI select from menus functions ---------------------------

def select_AI_from_menu(
        selected_option: str
        ) -> None:
    
    global selected_AI_model    
    selected_AI_model = selected_option

def select_framegeneration_option_from_menu(new_value: str):
    global selected_frame_generation_option    
    selected_frame_generation_option = new_value
    update_file_widget(1,2,3)

def select_gpu_from_menu(selected_option: str) -> None:
    global selected_gpu    
    selected_gpu = selected_option

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

def open_info_output_path():
    option_list = [
        "\n The default path is defined by the input files."
        + "\n For example uploading a file from the Download folder,"
        + "\n the app will save the generated files in the Download folder \n",

        " Otherwise it is possible to select the desired path using the SELECT button",
    ]

    CTkMessageBox(
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
    
    CTkMessageBox(
        messageType   = "info",
        title         = "AI model", 
        subtitle      = " This widget allows to choose between different RIFE models",
        default_value = default_AI_model,
        option_list   = option_list
    )

def open_info_frame_generation_option():
    option_list = [
        "\n FRAME GENERATION\n" + 
        "   • x2 ( doubles video framerate • 30fps => 60fps )\n" + 
        "   • x4 ( quadruples video framerate • 30fps => 120fps )\n" + 
        "   • x8 ( octuplicate video framerate • 30fps => 240fps )\n",

        "\n FRAME GENETATION/SLOWMOTION (no audio)\n" + 
        "   • x2-slowmotion ( slowmotion effect by a factor of 2 )\n" +
        "   • x4-slowmotion ( slowmotion effect by a factor of 4 )\n" +
        "   • x8-slowmotion ( slowmotion effect by a factor of 8 )\n"
    ]
    
    CTkMessageBox(
        messageType   = "info",
        title         = "AI frame generation", 
        subtitle      = " This widget allows to choose between different AI frame generation option",
        default_value = default_frame_generation_option,
        option_list   = option_list
    )

def open_info_gpu():
    option_list = [
        " Keep in mind that the more powerful the chosen gpu is, the faster the upscaling will be",
        
        " For optimal performance, it is essential to regularly update your GPU drivers",

        "\n Windows handles multiple GPUs by categorising them:\n" +
        "  • GPU High performance\n" + 
        "  • GPU Power saving\n" +
        "\n In the case of a single GPU, select 'High performance'\n"


    ]

    CTkMessageBox(
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
        " \n BMP\n  • highest quality\n  • slow and heavy file\n",
        " \n TIFF\n  • highest quality\n  • very slow and heavy file\n",
    ]

    CTkMessageBox(
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

    CTkMessageBox(
        messageType   = "info",
        title         = "Video output",
        subtitle      = "This widget allows to choose the extension of the upscaled video",
        default_value = default_video_extension,
        option_list   = option_list
    )

def open_info_save_frames():
    option_list = [
        "\n ENABLED \n FluidFrames.RIFE will create \n   • the fluidified video \n   • a folder containing all original and interpolated frames \n",
        "\n DISABLED \n FluidFrames.RIFE will create only the fluidified video \n"
    ]

    CTkMessageBox(
        messageType   = "info",
        title         = "Save frames",
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

    CTkMessageBox(
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

    CTkMessageBox(
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

    text_drop = """ SUPPORTED FILES \n\n IMAGES • jpg png tif bmp webp heic \n VIDEOS • mp4 webm mkv flv gif avi mov mpg qt 3gp """

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
    AI_menu        = create_option_menu(select_AI_from_menu, AI_models_list)

    AI_menu_button.place(relx = column0_x, rely = row1_y - 0.053, anchor = "center")
    AI_menu.place(relx = column0_x, rely = row1_y, anchor = "center")

def place_framegeneration_option_menu():

    fluidity_option_button = create_info_button(open_info_frame_generation_option, "AI frame generation")
    fluidity_option_menu   = create_option_menu(select_framegeneration_option_from_menu, frame_generation_options_list)

    fluidity_option_button.place(relx = column0_x, rely = row2_y - 0.05, anchor = "center")
    fluidity_option_menu.place(relx = column0_x, rely = row2_y, anchor = "center")

def place_gpu_menu():
    gpu_button = create_info_button(open_info_gpu, "GPU")
    gpu_menu   = create_option_menu(select_gpu_from_menu, gpus_list)
    
    gpu_button.place(relx = column1_x, rely = row1_y - 0.053, anchor = "center")
    gpu_menu.place(relx = column1_x, rely  = row1_y, anchor = "center")

def place_image_output_menu():
    file_extension_button = create_info_button(open_info_AI_output, "Image output")
    file_extension_menu   = create_option_menu(select_image_extension_from_menu, image_extension_list)
    
    file_extension_button.place(relx = column2_x, rely = row1_y - 0.05, anchor = "center")
    file_extension_menu.place(relx = column2_x, rely = row1_y, anchor = "center")

def place_video_extension_menu():
    video_extension_button = create_info_button(open_info_video_extension, "Video output")
    video_extension_menu   = create_option_menu(select_video_extension_from_menu, video_extension_list)
    
    video_extension_button.place(relx = column2_x, rely = row2_y - 0.05, anchor = "center")
    video_extension_menu.place(relx = column2_x, rely = row2_y, anchor = "center")

def place_save_frames_menu():

    save_frames_button = create_info_button(open_info_save_frames, "Save frames")
    save_frames_menu   = create_option_menu(select_save_frame_from_menu, save_frames_list)
    
    save_frames_button.place(relx = column1_x, rely = row2_y - 0.053, anchor = "center")
    save_frames_menu.place(relx = column1_x, rely = row2_y, anchor = "center")

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

def place_fluidify_button(): 
    fluidify_button = create_active_button(
        command = fludify_button_command,
        text    = "FLUIDIFY",
        icon    = play_icon,
        width   = 140,
        height  = 30
    )
    fluidify_button.place(relx = column2_x, rely = row4_y, anchor = "center")



# Main functions ---------------------------

def on_app_close():
    window.grab_release()
    window.destroy()
    stop_fluidify_process()

class App():
    def __init__(self, window):
        self.toplevel_window = None
        window.protocol("WM_DELETE_WINDOW", on_app_close)

        window.title('')
        window.geometry("675x675")
        window.resizable(False, False)
        window.iconbitmap(find_by_relative_path("Assets" + os_separator + "logo.ico"))

        place_app_name()
        place_output_path_textbox()
        place_github_button()
        place_telegram_button()

        place_AI_menu()
        place_framegeneration_option_menu()
        place_gpu_menu()

        place_image_output_menu()
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

    if os_path_exists(FFMPEG_EXE_PATH): 
        os_environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_EXE_PATH

    window = CTk() 

    info_message            = StringVar()
    selected_output_path    = StringVar()
    selected_resize_factor  = StringVar()
    selected_cpu_number     = StringVar()

    global selected_file_list
    global selected_AI_model
    global selected_frame_generation_option
    global selected_gpu 
    global selected_save_frames
    global resize_factor
    global cpu_number
    global selected_image_extension
    global selected_video_extension

    selected_file_list = []

    selected_AI_model                = default_AI_model
    selected_frame_generation_option = default_frame_generation_option
    selected_gpu                     = default_gpu
    selected_image_extension         = default_image_extension
    selected_video_extension         = default_video_extension

    selected_save_frames = True if default_save_frames == "Enabled" else False

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