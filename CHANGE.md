# Release Notes

## Known issues

- FP16 model didn't use ANE(Accelerated Neural Engine) on Apple silicon. Use FP32 model instead.

## Changes

### Version 0.1.1

Features:

- Switch default resize algorithm to `nearest` for faster preproces.
- Support custom model config file. Now `--model` option takes a `.toml` model config file path. And `--imgsz` option is removed. Check example model config file [here](./models/md_v5a_fp16.toml). Note that the input and output shape sequence and precision of custom model still need to be the same as predefined models.
- Check execution provider availability only once and save it in `epifo_{device}.json` file to save initialization time. If you want to force recheck, delete the file.
