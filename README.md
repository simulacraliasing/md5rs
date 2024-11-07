# A MegaDetector CLI tool

## Get started

To process a folder of cameratrap media with MegaDetector and export result in csv format, you can use the following command:

Windows:

`md5rs.exe -f <folder_to_process> -d 0 -i -m models/md_v5a.toml -max-frames 3 -e csv`

Linux/MacOS:

`md5rs -f <folder_to_process> -d 0 -i -m models/md_v5a.toml -max-frames 3 -e csv`

Supported export formats are `csv`, `json`.

Run `md5rs -h` to see all available options.

### Default Models

#### [MegaDetector](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md)

##### MegaDetector v5a

[FP32 model](https://huggingface.co/Simulacraliasing/Megadetector/resolve/main/md_v5a_d_pp.onnx?download=true)

[FP16 model](https://huggingface.co/Simulacraliasing/Megadetector/resolve/main/md_v5a_d_pp_fp16.onnx?download=true)

Converted onnx model from [md_v5a.0.0.pt](https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt) and add dynamic batch and some postprocessing.

##### MegaDetector v6b (Beta)

[FP32 model](https://huggingface.co/Simulacraliasing/Megadetector/resolve/main/md_v6b_d_pp.onnx?download=true)

[FP16 model](https://huggingface.co/Simulacraliasing/Megadetector/resolve/main/md_v6b_d_pp_fp16.onnx?download=true)

Converted onnx model from [MDV6b-yolov9c.pt](https://zenodo.org/records/11192829/files/MDV6b-yolov9c.pt?download=1) and add dynamic batch and some postprocessing.

## Key features

- [x] Multiple platform and device support based on ONNX Runtime
- [x] Support both image and video processing
- [x] Multithread at each step
- [x] Dynamic batch inference

## Supported devices and platforms

| Devices (Execution Providers)  | Linux | Windows | Macos |
| ------------------------------ | ----- | ------- | ----- |
| CPU (CoreML fo Apple silicon)  | ✅     | ✅       | ✅     |
| Nvidia GPU (CUDA/TensorRT)     | ✅     | ✅       |       |
| Intel GPU/NPU (OpenVINO)       | ✅     | ✅       |       |
| AMD GPU(DirectML)              |       | ✅       |       |
| Apple silicon GPU/NPU (CoreML) |       |         | ✅     |

*Other devices listed in ONNX Runtime execution providers should also be available, but are not built and tested yet.

## Prerequisites

### FFmpeg (for video processing)

FFmpeg binary should be automatically downloaded. In case of failure, you can download it from [here](https://ffmpeg.org/download.html)

### ONNX Runtime dynamic libraries(for model inference)

The automatically downloaded onnxruntime libs don't deliver OpenVINO support. Prebuilt ort libs can be downloaded from [here](https://github.com/simulacraliasing/md5rs/releases/tag/ort-prebuilt)

### Execution provider driver and dynamic libraries

Tested on CUDA 12.6, CUDNN 9.5, TensorRT 10.2, OpenVINO 2024.3

- [NVIDIA Driver](https://www.nvidia.com/en-us/drivers/)
- [CUDA](https://developer.nvidia.com/cuda-downloads)
- [CUDNN](https://developer.nvidia.com/cudnn)
- [TensorRT](https://developer.nvidia.com/tensorrt)
- [OpenVINO](https://storage.openvinotoolkit.org/repositories/openvino/packages/)

## Organize python script

The `organize.py` script is used to organize the cameratrap media into folders named with class names and blank in sequence. It takes the export result file as input and moves the media to the corresponding folders. The script can be used as follows:

`python organize.py --result result.csv --guess`

Sequence is determined by shoot time or filename extension pattern if shoot time is not available. Media shot in given time range is considered as a sequence. There is a guess model to determine sequence from filename extension pattern. For example, if filenames in a folder are `[IMG_0001.JPG, IMG_0002.JPG, IMG_0003.MOV, IMG_0004.JPG, IMG_0005.JPG, IMG_0006.MOV...]`, the guessed sequence is `[IMG_0001.JPG, IMG_0002.JPG, IMG_0003.MOV]`, `[IMG_0004.JPG, IMG_0005.JPG, IMG_0006.MOV]`... The guess model is experimental and may not work in all cases, use it with your own risk. You can always use the `--redo` option to restore.

## Known issues

FP16 model didn't use ANE(Accelerated Neural Engine) on Apple silicon. Use FP32 model instead.
