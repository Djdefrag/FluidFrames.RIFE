<div align="center">
    <br>
    <img src="https://user-images.githubusercontent.com/32263112/216588514-0ad68175-c65e-47ee-9ca8-d163572d9be9.png" width="175"> </a> 
    <br><br>FluidFrames.RIFE | video AI frame generation app <br><br>
    <a href="https://jangystudio.itch.io/fluidframesrife">
         <button>
            <img src="https://static.itch.io/images/badge-color.svg" width="225" height="70">
        </button>
    </a>
    <a href="https://store.steampowered.com/app/3228250/FluidFrames/">
        <button>
             <img src="https://images.squarespace-cdn.com/content/v1/5b45fae8b98a78d9d80b9c5c/1531959264455-E7B8MJ3VMPX0593VGCZG/button-steam-available-fixed-2.png" width="250" height="70">
        </button>                 
    </a>
</div>
<br>
<div align="center">
    <img src="https://github.com/user-attachments/assets/47c4a9ab-0f19-438d-8128-7bc1a09a0d54"> </a> 
</div>


## What is FluidFrames.RIFE?
FluidFrames.RIFE is a Windows app powered by RIFE AI to create frame-generated and slowmotion videos.

## Other AI projects.🤓
- https://github.com/Djdefrag/QualityScaler / QualityScaler - image/video AI upscaler app
- https://github.com/Djdefrag/RealScaler / RealScaler - image/video AI upscaler app (Real-ESRGAN)

## Credits.
- RIFE - https://github.com/megvii-research/ECCV2022-RIFE
- PraticalRIFE - https://github.com/hzwer/Practical-RIFE

## How is made. 🛠
FluidFrames is completely written in Python, from backend to frontend. 
- [x] pytorch (https://github.com/pytorch/pytorch)
- [x] onnx (https://github.com/onnx/onnx)
- [x] onnxconverter-common (https://github.com/microsoft/onnxconverter-common)
- [x] onnxruntime-directml (https://github.com/microsoft/onnxruntime)
- [x] customtkinter (https://github.com/TomSchimansky/CustomTkinter)
- [x] openCV (https://github.com/opencv/opencv)
- [x] moviepy (https://github.com/Zulko/moviepy)
- [x] pyInstaller (https://github.com/pyinstaller/pyinstaller)

## Make it work by yourself. 👨‍💻
Prerequisites.
- Python installed on your pc (https://www.python.org/downloads/release/python-3119/)
- VSCode installed on your pc (https://code.visualstudio.com/)
- FFMPEG.exe downloaded (https://www.gyan.dev/ffmpeg/builds/) RELEASE BUILD > ffmpeg-release-essentials.7z

Getting started.
- Download the project on your PC (Green button Code > Download ZIP)
- Extract the project from the .zip
- Extract FFMPEG.exe in /Assets folder
- Open the project with VSCode (Drag&Drop the project directory on VSCode)
- Click on FluidFrames.py from left bar (VSCode will ask to install Python plugins)
- Install dependencies. In VSCode there is the "Terminal" panel, click there and execute the command "pip install -r requirements.txt"
- Close VSCode and re-open it (this will refresh all the dependecies installed)
- Click on the "Play button" in the upper right corner of VSCode

## Requirements. 🤓
- [ ] Windows 11 / Windows 10
- [ ] RAM >= 8Gb
- [ ] Any Directx12 compatible GPU with >= 2GB VRAM

## Features.
- [x] Elegant and easy to use GUI
- [x] Resize video before interpolation
- [x] Multiple GPUs support
- [x] Compatible video  - mp4, wemb, gif, mkv, flv, avi, mov, qt
- [x] Video frame-generation STOP&RESUME
- [x] PRIVACY FOCUSED - no internet connection required / everything is on your PC
- [x] Video frames generation x2 / x4 / x8
   - 30fps => x2 => 60fps
   - 30fps => x4 => 120fps
   - 30fps => x8 => 240fps
 - [x] Video slowmotion x2 /x4
   - 30fps => x2_slowmotion => 30fps - 2 times slower
   - 30fps => x4_slowmotion => 30fps - 4 times slower
   - 30fps => x8_slowmotion => 30fps - 8 times slower

## Next steps. 🤫
- [x] 1.X versions
    - [x] Switch to Pytorch-directml to support all Directx12 compatible gpu (AMD, Intel, Nvidia)
    - [x] New GUI with Windows 11 style
    - [x] Include audio for processed video
    - [x] Optimizing video frame resize and extraction speed
    - [x] Multi GPU support (for pc with double GPU, integrated + dedicated)
    - [x] Python 3.10 (expecting ~10% more performance)
    - [x] Slowmotion function
- [x] 2.X versions
    - [x] New, completely redesigned graphical interface based on @customtkinter
    - [x] Fluidify multiple videos at once
    - [x] Save AI generated frames as files
    - [x] Support RIFE AI model updates
    - [x] Support for RIFE_Lite AI model (a faster and lighter version of RIFE) 
- [ ] 3.x versions (now under development)
    - [x] New AI engine powered by onnxruntime-directml (https://github.com/microsoft/onnxruntime)
    - [x] Python 3.11 (performance improvements)
    - [x] Python 3.12 (performance improvements)
    - [x] Display frame-generated videos info in the GUI
    - [x] FFMPEG 7 (latest release)
    - [x] Saving user settings (AI model, GPU, CPU etc.)
    - [x] Video frame-generation STOP&RESUME


### Some Examples.
#### Videos
1. Original / x4 / x2-slomotion

![giphy](https://github.com/Djdefrag/FluidFrames.RIFE/assets/32263112/eebc82fd-8218-4f40-b969-b74c9dd6f2e8)

https://github.com/Djdefrag/FluidFrames.RIFE/assets/32263112/e8e728b4-a2f5-4a74-8f04-5a5977c69fc4

https://github.com/Djdefrag/FluidFrames.RIFE/assets/32263112/21007233-b7ff-4836-a207-cfe3ed23ed28


3. Original / x4 / x4-slomotion

https://user-images.githubusercontent.com/32263112/235297757-5daf129e-4e19-4b8b-b6c8-b661ac1028db.mp4

https://user-images.githubusercontent.com/32263112/235297763-26bf9fdd-3d40-4aba-8688-5ef85a532ed0.mp4

https://user-images.githubusercontent.com/32263112/235297767-0adc4635-a43e-4c37-bd15-a24e1dd47f32.mp4


3. Original / x2

https://user-images.githubusercontent.com/32263112/222885925-a28122e8-92f8-4e53-b287-4ae17bb177c7.mp4

https://user-images.githubusercontent.com/32263112/222885933-f2e13869-984c-4192-8020-1668035e5cd3.mp4


4. Original / x2

![209639439-94c8774d-354e-4d56-9123-e1aa4af95e08](https://user-images.githubusercontent.com/32263112/221165591-3a0fb780-3ba8-4cf5-8405-fc83eb58ee66.gif)

https://user-images.githubusercontent.com/32263112/221165739-71dfd957-5d3d-481b-9a26-bb08d5affa6f.mp4


5. Original / x2 / x2-slomotion

https://user-images.githubusercontent.com/32263112/228229016-8b26c8f3-8a68-4b5e-b1ff-d52f9be76a03.mp4

https://user-images.githubusercontent.com/32263112/228229044-9d267a66-543e-43ca-890b-db6a70c29d0b.mp4

https://user-images.githubusercontent.com/32263112/228229083-d29a313f-3d28-4cdb-9d97-63410f28a608.mp4




