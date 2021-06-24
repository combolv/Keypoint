from timm.models.swin_transformer import swin_base_patch4_window12_384, swin_base_patch4_window7_224
from timm.models.swin_transformer import swin_small_patch4_window7_224

from torchvision.models.segmentation.segmentation import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch.nn.functional import interpolate
import torch

AVAILABLE_MODEL = {
    'swin_base_patch4_window12_384': swin_base_patch4_window12_384,
    'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
    'swin_small_patch4_window7_224': swin_small_patch4_window7_224
}


class swintBackbone(torch.nn.Module):
    '''
        [B, 3, H, W] -> [B, 1024, H/32, W/32]
    '''

    def __init__(self, name='swin_base_patch4_window12_384', pretrained=True):
        super(swintBackbone, self).__init__()
        swint = AVAILABLE_MODEL[name](pretrained)
        # extract backbone
        self.patch_embed = swint.patch_embed
        self.absolute_pos_embed = swint.absolute_pos_embed
        self.pos_drop = swint.pos_drop
        self.layers = torch.nn.Sequential(swint.layers[:-1])

    def forward(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        B, S, C = x.size()
        x = x.transpose(1, 2)
        x = x.view((B, C, int(S ** 0.5), -1))
        return x


class DeepLabV3Head(torch.nn.Module):
    '''
        [B, in_channels, H/in_stride, W/in_stride] -> [B, num_keypoints, H/down_scale, W/down_scale]
    '''

    def __init__(self, down_scale=1, num_keypoints=17, in_channels=1024, in_stride=32):
        super(DeepLabV3Head, self).__init__()
        self.down_scale = down_scale
        self.num_keypoints = num_keypoints
        self.head = DeepLabHead(in_channels, num_keypoints)
        self.in_stride = in_stride

    def forward(self, x):
        h, w = x.size()[-2:]
        scale = self.in_stride//self.down_scale//4
        x = interpolate(x, size=(h * scale, w * scale), mode='bilinear')
        x = self.head(x)
        final = self.in_stride//self.down_scale
        x = interpolate(x, size=(h * final,w * final),mode='bilinear')
        return x


class DeconvHead(torch.nn.Module):
    def __init__(self, down_scale=1, num_keypoints=17, in_channels=1024, in_stride=32):
        super(DeconvHead, self).__init__()
        self.down_scale = down_scale
        self.num_keypoints = num_keypoints
        self.head = DeepLabHead(in_channels//8, num_keypoints)
        self.in_channels = in_channels
        self.in_stride = in_stride
        self.deconv_layer_1 = self._get_deconv_layer(self.in_channels)
        self.deconv_layer_2 = self._get_deconv_layer(self.in_channels//2)
        self.deconv_layer_3 = self._get_deconv_layer(self.in_channels//4)

    def _get_deconv_layer(self, num_channel):
        return torch.nn.ConvTranspose2d(
                    in_channels=num_channel,
                    out_channels=num_channel//2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=True)

    def forward(self, x):
        x = self.deconv_layer_1(x)
        x = self.deconv_layer_2(x)
        x = self.deconv_layer_3(x)
        x = self.head(x)
        return x


class KeypointDetector(torch.nn.Module):
    def __init__(self, backbone, head):
        super(KeypointDetector, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        feature = self.backbone(x)
        pred = self.head(feature)
        return pred

if __name__ == '__main__':
    net = KeypointDetector(swintBackbone(pretrained=False), DeconvHead())
    a = torch.randn(2, 3, 384, 384)
    out = net(a)
    # net = KeypointDetector(swintBackbone(pretrained=False),DeepLabV3Head())
    # a = torch.randn(2,3,384,384)
    # out = net(a)
    print(net)
    print(f'Parameters: {sum(i.numel() for i in net.parameters()) / 1e6}M')
    print(out.size())
