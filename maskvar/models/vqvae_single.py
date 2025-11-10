from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .basic_vae import Decoder, Encoder
from .quant import VectorQuantizer2


class VQVAE_Single(nn.Module):
    """
    
    Args:
        vocab_size      : codebook size, number of tokens in discrete latent space
        z_channels      : number of channels in latent space  
        ch              : base number of channels for building network layers
        dropout         : dropout rate for preventing overfitting
        beta            : weight for commitment loss, controls difference between encoder output and quantized vectors
        using_znorm     : whether to normalize when computing nearest neighbors
        quant_conv_ks   : kernel size for quantization convolution layer
        quant_resi      : residual connection ratio, 0.5 means \phi(x) = 0.5*conv(x) + (1-0.5)*x
        share_quant_resi: number of \phi layers shared across different scales
        default_qresi_counts: default number of quantization residual layers, 0 means auto-set to length of v_patch_nums
        v_patch_nums    : number of patches per scale, h_{1 to K} = w_{1 to K} = v_patch_nums[k]
        test_mode       : whether in test mode
    """
    def __init__(
        self, 
        vocab_size=4096,        # codebook size, number of tokens in discrete latent space
        z_channels=32,          # number of channels in latent space
        ch=128,                 # base number of channels for building network layers
        dropout=0.0,            # dropout rate for preventing overfitting
        beta=0.25,              # weight for commitment loss, controls difference between encoder output and quantized vectors
        using_znorm=False,      # whether to normalize when computing nearest neighbors
        quant_conv_ks=3,        # kernel size for quantization convolution layer
        quant_resi=0.5,         # residual connection ratio, 0.5 means \phi(x) = 0.5*conv(x) + (1-0.5)*x
        share_quant_resi=4,     # number of \phi layers shared across different scales
        default_qresi_counts=0, # default number of quantization residual layers, 0 means auto-set to length of v_patch_nums
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # number of patches per scale, h_{1 to K} = w_{1 to K} = v_patch_nums[k]
        test_mode=True,         # whether in test mode
        ddconfig=dict(in_channels=1, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,   # channel multipliers for building network layers
                using_sa=True, using_mid_sa=True,),
    ):
        super().__init__()
        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels  # V: codebook size, Cvae: latent space channels
        
        # Network configuration based on CompVis's vq-f16 config but using single channel input
        if ddconfig is None:
            ddconfig = dict(
                dropout=dropout, ch=ch, z_channels=z_channels,
                in_channels=1, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,   # Channel multipliers for building network layers
                using_sa=True, using_mid_sa=True,                           # Whether to use self-attention
            )
        else:
            ddconfig = dict(
                dropout=dropout, ch=ch, z_channels=z_channels,
                **ddconfig,                         # Whether to use self-attention
            )
        ddconfig.pop('double_z', None)  # Remove double_z parameter since only KL-VAE needs it
        
        # Initialize encoder and decoder
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)
        
        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig['ch_mult'])-1)  # Downsampling rate
        
        # Initialize vector quantizer
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta,
            default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, quant_resi=quant_resi, share_quant_resi=share_quant_resi,
        )
        
        # Convolution layers before and after quantization
        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        self.post_quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        
        # If in test mode, set model to eval state and freeze parameters
        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]
    
    def forward(self, inp, ret_usages=False):   # -> rec_B1HW, idx_N, loss
        """Forward pass function, used for training process
        
        Args:
            inp: Input single-channel image, shape [B, 1, H, W]
            ret_usages: Whether to return codebook usage
            
        Returns:
            rec_B1HW: Reconstructed single-channel image, shape [B, 1, H, W]
            usages: Codebook usage (if ret_usages=True)
            vq_loss: Vector quantization loss
        """
        f_hat, usages, vq_loss = self.quantize(self.quant_conv(self.encoder(inp)), ret_usages=ret_usages)
        return self.decoder(self.post_quant_conv(f_hat)), usages, vq_loss
    
    def fhat_to_img(self, f_hat: torch.Tensor):
        """Convert quantized features to single-channel image
        
        Args:
            f_hat: Quantized feature tensor
            
        Returns:
            Reconstructed single-channel image, clipped to [-1, 1]
        """
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
    
    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:
        """Convert single-channel image to multi-scale index list
        
        Args:
            inp_img_no_grad: Input single-channel image (no gradient), shape [B, 1, H, W]
            v_patch_nums: Number of patches per scale
            
        Returns:
            List[Bl]: Multi-scale index list, each element is an index tensor with batch size B and length l
        """
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)
    
    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        """Convert multi-scale index list to single-channel image
        
        Args:
            ms_idx_Bl: Multi-scale index list
            same_shape: Whether to convert features to the maximum scale
            last_one: Whether to return only the last scale's reconstruction result
            
        Returns:
            Reconstructed single-channel image or image list, shape [B, 1, H, W]
        """
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            ms_h_BChw.append(self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn))
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)
    
    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        """Convert multi-scale features to single-channel image
        
        Args:
            ms_h_BChw: Multi-scale feature list
            all_to_max_scale: Whether to convert features to the maximum scale
            last_one: Whether to return only the last scale's reconstruction result
            
        Returns:
            Reconstructed single-channel image or image list, shape [B, 1, H, W]
        """
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True))).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]
    
    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False) -> List[torch.Tensor]:
        """Convert single-channel image to reconstructed image
        
        Args:
            x: Input single-channel image, shape [B, 1, H, W]
            v_patch_nums: Number of patches per scale
            last_one: Whether to return only the last scale's reconstruction result
            
        Returns:
            Reconstructed single-channel image or image list, shape [B, 1, H, W]
        """
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1])).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in ls_f_hat_BChw]
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        if 'quantize.ema_vocab_hit_SV' in state_dict and state_dict['quantize.ema_vocab_hit_SV'].shape[0] != self.quantize.ema_vocab_hit_SV.shape[0]:
            state_dict['quantize.ema_vocab_hit_SV'] = self.quantize.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign) 