# calibdiff
A PyTorch-based differential camera calibration library for intrinsic, extrinsic, and stereo camera calibration.



## ▮ Install
```bash
pip3 install calibdiff
```
## ▮ Run Example

This example demonstrates the optimization of stereo camera intrinsic and extrinsic parameters using PyTorch and [LoFTR](https://github.com/zju3dv/LoFTR) for feature matching.

```bash
pip3 install calibdiff

git clone https://github.com/DIYer22/calibdiff

# This will automatically downloads example data to /tmp and do stereo optimize by LoFTR
python calibdiff/calibdiff/stereo_optimize.py
```

![stereo](https://user-images.githubusercontent.com/10448025/232319503-56566640-6ead-4813-8af8-90700500d057.jpg)  
*Stereo camera parameters optimization by LoFTR: left image shows the initial state, right image demonstrates the rectified effect after optimization.*