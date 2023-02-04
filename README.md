<div align="center">
    <br>
    <img src="https://user-images.githubusercontent.com/32263112/216588514-0ad68175-c65e-47ee-9ca8-d163572d9be9.png" width="175"> </a> 
    <br><br>FluidFrames.RIFE | video frames AI interpolation app (RIFE-HDv3) <br><br>
    <a href="https://jangystudio.itch.io/fluidframesrife">
         <img src="https://user-images.githubusercontent.com/86362423/162710522-c40c4f39-a6b9-48bc-84bc-1c6b78319f01.png" width="200">
    </a>
</div>
<br>
<div align="center">
    <img src="https://user-images.githubusercontent.com/32263112/216588704-752de2d7-d78c-4776-9935-f1d44ef4b8a8.PNG"> </a> 
</div>

## What is FluidFrames.RIFE?
FluidFrames.RIFE is a Windows app that uses RIFE-HDv3 artificial intelligence to doubling or quadrupling videos fps.

## Other AI projects.🤓

https://github.com/Djdefrag/QualityScaler / QualityScaler - image/video AI upscaler app (BSRGAN)

https://github.com/Djdefrag/RealESRScaler / RealESRScaler - image/video AI upscaler app (Real-ESRGAN)

## Credits.

RIFE - https://github.com/megvii-research/ECCV2022-RIFE

## How is made. 🛠

FluidFrames.RIFE is completely written in Python, from backend to frontend. 
External packages are:
- [ ] AI  -> torch / torch-directml
- [ ] GUI -> tkinter / tkdnd / sv_ttk
- [ ] Image/video -> openCV / moviepy
- [ ] Packaging   -> pyinstaller
- [ ] Miscellaneous -> pywin32 / win32mica

## Requirements. 🤓
- [ ] Windows 11 / Windows 10
- [ ] RAM >= 8Gb
- [ ] Directx12 compatible GPU:
    - [ ] any AMD >= Radeon HD 7000 series
    - [ ] any Intel HD Integrated >= 4th-gen core
    - [ ] any NVIDIA >=  GTX 600 series

## Features.

- [x] Easy to use GUI
- [x] Video frames interpolation
- [x] Drag&Drop video
- [x] Resize video before interpolation
- [x] Multiple gpu backend
- [x] Compatible video  - mp4, wemb, gif, mkv, flv, avi, mov, qt 

## Next steps. 🤫
- [ ] Update libraries 
    - [x] Python 3.10 (expecting ~10% more performance) 
    - [ ] Python 3.11 (expecting ~30% more performance)
- [ ] Add slowmotion function

## Known bugs.
- [ ] Filenames with non-latin symbols (for example kangy, cyrillic etc.) not supported - [Temp solution] rename files like "image" or "video"
- [ ] When running as Administrator, drag&drop is not working

### Some Examples.
#### Videos

Original

![all-might-scream](https://user-images.githubusercontent.com/32263112/216591290-36d770c9-6bd9-4dce-aca2-a83e7479c605.gif)

x4

https://user-images.githubusercontent.com/32263112/216591361-bc93b977-f312-4fa0-b255-33a7cbe43c29.mp4

Original

![103154](https://user-images.githubusercontent.com/32263112/216591421-4c24c1b3-9929-4806-acd6-28a7f61430b6.gif)

x4

https://user-images.githubusercontent.com/32263112/216591550-6b7ec75a-f371-43bf-9535-95a1182aa6f0.mp4



