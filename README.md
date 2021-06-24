# SwinTPoseNet: SwinT-based 2D human pose estimation
The codebase is from https://github.com/microsoft/human-pose-estimation.pytorch. See it for required packages.
We used the SwinT backbone together with a DeeplabV3 Head, refering to:
```
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from timm.models.swin_transformer import swin_small_patch4_window7_224
```

To install this model, please clone and run 
```
pip install easydict
make
mkdir data
cd data
ln -s ${PATH-TO-COCO} coco
```

The coco dataset should be prepared as 
```
coco
  annotations
  train2017
  val2017
  test2017
```
