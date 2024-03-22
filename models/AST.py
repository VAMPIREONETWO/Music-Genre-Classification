# Code is refactored base on https://github.com/YuanGongND/ast.
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os

from torchinfo import summary

# import wget
os.environ['TORCH_HOME'] = 'pretrained_models'
import timm
from timm.layers import to_2tuple, trunc_normal_, PatchEmbed
from typing import Callable, Optional


class ASTPatchEmbed(PatchEmbed):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 norm_layer: Optional[Callable] = None,
                 flatten: bool = True,
                 output_fmt: Optional[str] = None,
                 bias: bool = True,
                 strict_img_size: bool = True,
                 dynamic_img_pad: bool = False, ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            flatten=flatten,
            output_fmt=output_fmt,
            bias=bias,
            strict_img_size=strict_img_size,
            dynamic_img_pad=dynamic_img_pad)

        img_size = to_2tuple(img_size)  # (img_size,img_size)
        patch_size = to_2tuple(patch_size)  # (patch_size,patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class AST(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """

    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024):

        super(AST, self).__init__()
        # override timm input shape restriction
        self.v = timm.create_model('deit_base_distilled_patch16_384', pretrained=True,
                                   embed_layer=ASTPatchEmbed)

        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim),
                                      nn.Linear(self.original_embedding_dim, label_dim),
                                      nn.BatchNorm1d(label_dim),
                                      nn.Sigmoid())

        # combine block with dropout to improve ability to generalize
        self.blocks = nn.ModuleList()
        i = 0
        for blk in self.v.blocks:
            self.blocks.append(blk)
            i += 1
            if i == 3:
                self.blocks.append(nn.Dropout(0.5))
                i = 0

        # automatcially get the intermediate shape
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
        print('number of patches={:d}'.format(num_patches))

        # the linear projection layer
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))

        new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
        new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj

        # the positional embedding
        # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
        new_pos_embed = (self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches,
                                                                     self.original_embedding_dim).transpose(1, 2).
                         reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw))
        # cut (from middle) or interpolate the second dimension of the positional embedding
        if t_dim <= self.oringal_hw:
            new_pos_embed = new_pos_embed[:, :, :,
                            int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(
                                t_dim / 2) + t_dim]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim),
                                                            mode='bilinear')
        # cut (from middle) or interpolate the first dimension of the positional embedding
        if f_dim <= self.oringal_hw:
            new_pos_embed = new_pos_embed[:, :,
                            int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(
                                f_dim / 2) + f_dim, :]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
        # flatten the positional embedding
        new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
        # concatenate the above positional embedding with the cls token and distillation token of the deit model.
        self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast()
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        x = self.mlp_head(x)
        return x


if __name__ == '__main__':
    input_tdim = 100
    ast_mdl = AST(input_tdim=input_tdim)
    # input a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
    test_input = torch.rand([10, input_tdim, 128])
    test_output = ast_mdl(test_input)
    summary(ast_mdl, [test_input.shape])
    # output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
    print(test_output.shape)

