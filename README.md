## OpenCV C++ multi focal stereo Camera Calibration
**Attention:** The calibration program is modified by [stereo-calibration](https://github.com/sourishg/stereo-calibration/), which can calibrate multi focal length stereo camera for MF-SLAM.

[MF-SLAM] Mingchi Feng, Jinglin Liu, Xin Wang, Chengnan Li, **MF-SLAM: Multi-focal SLAM**, Under review. **[](https://link.springer.com/chapter/10.1007/978-3-030-89134-3_45)**.
This repository contains some sources to calibrate the intrinsics of individual cameras and also the extrinsics of a stereo pair.

### Dependencies

- OpenCV
- popt

### Compilation

Compile all the files using the following commands.

```bash
mkdir build && cd build
cmake ..
make
```

Make sure your are in the `build` folder to run the executables.Cancel changes


### Stereo calibration for extrinisics

Once you have the intrinsics calibrated for both the left and the right cameras, you can use their intrinsics to calibrate the extrinsics between them.

```bash
./calibrate_stereo -n [num_imgs] -u [left_cam_calib] -v [right_cam_calib] -L [left_img_dir] -R [right_img_dir] -l [left_img_prefix] -r [right_img_prefix] -o [output_calib_file] -e [file_extension]
```

For example, if you calibrated the left and the right cameras using the images in the `calib_imgs/1/` directory, the following command to compute the extrinsics.

```bash
./mycalib_stereo -n 27 -u cam_left.yml -v cam_right.yml -L ../calib_imgs/1/ -R ../calib_imgs/1/ -l left -r right -o cam_stereo.yml -e jpg
```


