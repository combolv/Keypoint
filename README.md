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
### Main Result
We compared the pretrained SwinT backbone and HRNet backbone.
![Accuracy](https://user-images.githubusercontent.com/55729972/141292505-8b49c561-907d-44ce-a598-4654733dbcfd.png)
![Loss](https://user-images.githubusercontent.com/55729972/141292543-a3f8c28a-22c4-498b-ad51-1d729b5ef06b.png)
![AP](https://user-images.githubusercontent.com/55729972/141292560-46176cf4-fbf8-4400-acde-249f9e2900c7.png)
![AP_L](https://user-images.githubusercontent.com/55729972/141292610-d9f6a034-76fd-4b87-95a5-324301371fde.png)
![AP_M](https://user-images.githubusercontent.com/55729972/141292633-ddc34d2a-2351-4dea-84e5-7cf3baf90925.png)
![AP50](https://user-images.githubusercontent.com/55729972/141292654-e1093583-fe85-403f-a31f-5ef41c940208.png)
![AP75](https://user-images.githubusercontent.com/55729972/141292676-958c8e15-51d4-4bb7-96e2-2a848db6c0ca.png)
![AR](https://user-images.githubusercontent.com/55729972/141292730-c3f40372-70ca-4367-9669-f1a166de22f6.png)
![AR_50](https://user-images.githubusercontent.com/55729972/141292743-df4c3044-2bda-4304-91db-e309e021d9de.png)
![AR_75](https://user-images.githubusercontent.com/55729972/141292759-88021658-fd18-43b3-a72b-321957709374.png)
![AR_L](https://user-images.githubusercontent.com/55729972/141292779-10cf8871-9265-46b5-8eba-9308f110060c.png)
![AR_M](https://user-images.githubusercontent.com/55729972/141292790-ba39accd-cab5-4803-976c-471a245f3064.png)


