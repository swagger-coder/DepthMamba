from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block, Mlp
from timm.models.layers import DropPath as TimmDropPath,\
    to_2tuple, trunc_normal_

from util.pos_embed import get_2d_sincos_pos_embed

import torch.nn.functional as F

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from depth_anything.dpt import DPTHead

from tiny_vit.tiny_vit import PatchEmbed, ConvLayer, BasicLayer, PatchMerging

class TinyViT(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000,
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_sizes=[7, 7, 14, 7],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 use_checkpoint=False,
                 mbconv_expand_ratio=4.0,
                 local_conv_size=3,
                 layer_lr_decay=1.0,
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        activation = nn.GELU

        self.patch_embed = PatchEmbed(in_chans=in_chans,
                                      embed_dim=embed_dims[0],
                                      resolution=img_size,
                                      activation=activation)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs = dict(dim=embed_dims[i_layer],
                          input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)),
                          depth=depths[i_layer],
                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                          downsample=PatchMerging if (
                              i_layer < self.num_layers - 1) else None,
                          use_checkpoint=use_checkpoint,
                          out_dim=embed_dims[min(
                              i_layer + 1, len(embed_dims) - 1)],
                          activation=activation,
                          )
            if i_layer == 0:
                layer = ConvLayer(
                    conv_expand_ratio=mbconv_expand_ratio,
                    **kwargs,
                )
            else:
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs)
            self.layers.append(layer)

        # Classifier head
        # self.norm_head = nn.LayerNorm(embed_dims[-1])
        # self.head = nn.Linear(
        #     embed_dims[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

        # init weights
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)

    def set_layer_lr_decay(self, layer_lr_decay):
        decay_rate = layer_lr_decay

        # layers -> blocks (depth)
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]

        def _set_lr_scale(m, scale):
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(
                    lambda x: _set_lr_scale(x, lr_scales[i - 1]))
        assert i == depth
        # for m in [self.norm_head, self.head]:
        #     m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))

        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            for p in m.parameters():
                assert hasattr(p, 'lr_scale'), p.param_name

        self.apply(_check_lr_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'attention_biases'}

    def forward_features(self, x):
        # x: (N, C, H, W)
        x = self.patch_embed(x)
        output = []

        x = self.layers[0](x)
        output.append(x)
        start_i = 1

        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
            output.append(x)

        # x = x.mean(1)

        assert len(output) == 4

        return output

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm_head(x)
        x = self.head(x)
        return x

class DPT_TinyViT(nn.Module):
    def __init__(self, encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], embed_dim=1024,
                 use_bn=False, use_clstoken=False, localhub=True, aligned_blks_indices=2, 
                 embedding_distillation_func="L1", aligned_feature_projection_mode=None, 
                 aligned_feature_projection_dim=None):
        super(DPT_TinyViT, self).__init__()
        
        self.encoder = TinyViT( embed_dims=[64, 128, 160, 320],
                                depths=[2, 2, 6, 2],
                                num_heads=[2, 4, 5, 10],
                                window_sizes=[7, 7, 14, 7],
                                drop_path_rate=0.0)
        
        # dim = self.pretrained.blocks[0].attn.qkv.in_features
        if encoder == 'vits':
            dim = 384
        elif encoder == 'vitb':
            dim = 768
        else:
            dim = 1024
        
        #TODO 需要冻结decoder的参数
        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

        self.aligned_blks_indices = aligned_blks_indices
        if self.aligned_blks_indices is not None:
            assert embedding_distillation_func is not None
            distillation_loss_dict = dict(L1=nn.L1Loss(), L2=nn.MSELoss())
            self.distillation_criterion = distillation_loss_dict[embedding_distillation_func]

        if aligned_feature_projection_mode is not None:
            assert aligned_feature_projection_dim is not None
            assert aligned_feature_projection_dim[0] == embed_dim
            if aligned_feature_projection_mode == 'fc-1layer':
                student_feature_dim, teacher_feature_dim = aligned_feature_projection_dim
                self.aligned_feature_projection_heads = nn.ModuleList([
                    nn.Linear(student_feature_dim, teacher_feature_dim)
                    for i in range(len(self.aligned_blks_indices))]
                )
            elif aligned_feature_projection_mode == 'mlp-1layer':
                student_feature_dim, teacher_feature_dim = aligned_feature_projection_dim
                self.aligned_feature_projection_heads = nn.ModuleList([
                    Mlp(in_features=student_feature_dim, hidden_features=teacher_feature_dim, out_features= teacher_feature_dim, act_layer=nn.GELU, drop=self.dropout)
                    for i in range(len(self.aligned_blks_indices))]
                )
            elif aligned_feature_projection_mode == 'mlp-2layer':
                student_feature_dim, teacher_feature_dim = aligned_feature_projection_dim
                self.aligned_feature_projection_heads = nn.ModuleList([
                    nn.Sequential(*[
                    Mlp(in_features=_feature_dim, hidden_features=teacher_feature_dim, out_features= teacher_feature_dim, act_layer=nn.GELU, drop=0.0),
                    Mlp(in_features=teacher_feature_dim, hidden_features=teacher_feature_dim, out_features= teacher_feature_dim, act_layer=nn.GELU, drop=0.0)])
                    for i in range(len(self.aligned_blks_indices))]
                )
        else:
            self.aligned_feature_projection_heads = None
    
    def forward_encoder(self, x):
        h, w = x.shape[-2:]

        features = self.encoder.forward_features(x)

        return features, h, w

    def forward_decoder(self, features, h, w):
        patch_h, patch_w = h // 14, w // 14

        depth = self.depth_head(features, patch_h, patch_w)

        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        #TODO 需要考虑多图情况下的处理
        return depth.squeeze(1) 
    
    def forward_distillation_loss_embedding(self, features_teacher, features_student):
        assert isinstance(features_teacher, list) and isinstance(features_student, list)
        assert len(features_teacher) == len(features_student)
        loss_distillation_embedding = dict()
        if self.aligned_feature_projection_heads is not None:
            for feature_teacher, feature_student, blk_idx, projection_head in zip(
                    features_teacher, features_student,
                    self.aligned_blks_indices, self.aligned_feature_projection_heads):
                loss_distillation_embedding[f'align_block{blk_idx}'] = \
                    self.distillation_criterion(F.normalize(feature_teacher.detach(), dim=-1), F.normalize(projection_head(feature_student), dim=-1))
        else:
            for feature_teacher, feature_student, blk_idx in zip(features_teacher, features_student,
                                                                 len(feature_student)):
                if blk_idx in self.aligned_blks_indices:
                    loss_distillation_embedding[f'align_block{blk_idx}'] = \
                        self.distillation_criterion(feature_teacher.detach(), feature_student)

        return loss_distillation_embedding
    
    def forward_loss(self, teacher_prediction, pred):
        #TODO 目前使用最简单的MSE Loss，后续可以考虑使用更复杂的loss
        return F.mse_loss(pred, teacher_prediction)

    

    def forward(self, x, features_teacher, teacher_prediction = None):
        assert features_teacher is not None
        features_student, h, w = self.forward_encoder(x)
        pred = None
        loss = 0.
        # pred = self.forward_decoder(features_student, h, w)

        loss_distillation_embedding = self.forward_distillation_loss_embedding(features_teacher, features_student)
        
        # loss = self.forward_loss(teacher_prediction, pred)

        return loss, loss_distillation_embedding, pred

class DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False, localhub=True):
        super(DPT_DINOv2, self).__init__()
        
        assert encoder in ['vits', 'vitb', 'vitl']
        
        # in case the Internet connection is not stable, please load the DINOv2 locally
        if localhub:
            self.pretrained = torch.hub.load('torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder), source='local', pretrained=False)
        else:
            self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))
        
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        
        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
    
    def forward_encoder(self, x):
        h, w = x.shape[-2:]

        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True) # [visual_token, class_token]

        return features, h, w
    
    def forward_decoder(self, features, h, w):
        patch_h, patch_w = h // 14, w // 14

        depth = self.depth_head(features, patch_h, patch_w)

        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        #TODO 需要考虑多图情况下的处理
        return depth.squeeze(1) 

    def forward(self, x):
        h, w = x.shape[-2:]
        # start_time = time.perf_counter()
        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True) # [visual_token, class_token]
        # elapsed_time = time.perf_counter() - start_time
        # print('features Elapsed time: {:.2f}s'.format(elapsed_time))
        
        patch_h, patch_w = h // 14, w // 14

        # start_time = time.perf_counter()
        depth = self.depth_head(features, patch_h, patch_w) # [1, 1, 518, 826]
        # elapsed_time = time.perf_counter() - start_time
        # print('depth Elapsed time: {:.2f}s'.format(elapsed_time))
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth.squeeze(1)

class DepthAnything(DPT_DINOv2, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__(**config)


