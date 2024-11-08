# Release Notes

## Known issues

- FP16 model didn't use ANE(Accelerated Neural Engine) on Apple silicon. Use FP32 model instead.

## Changes

### Version 0.1.3

Update dependencies `ffmpeg-sidecar` to 2.0, `ort` to 2.0.0.rc8, `onnxruntime` to 1.20.0.

Bug fixes:

- Fix potential panic caused by ffmpeg error

### Version 0.1.2

Features:

- Add multiple device support. Example to run 1 workers on device 0 and 2 worker on device 1:`$ md5rs -d 0 -w 1 -d 1 -w 2`, run both on integrated GPU and discrete GPU: `$ md5rs -d gpu -w 1 -d 0 -w 2`.

Bug fixes:

- Fix resize_with_pad wrong returned dimension.

### Version 0.1.1

Features:

- Switch default resize algorithm to `nearest` for faster preproces.
- Support custom model config file. Now `--model` option takes a `.toml` model config file path. And `--imgsz` option is removed. Check example model config file [here](./models/md_v5a_fp16.toml). Note that the input and output shape sequence and precision of custom model still need to be the same as predefined models.
- Check execution provider availability only once and save it in `epifo_{device}.json` file to save initialization time. If you want to force recheck, delete the file.
