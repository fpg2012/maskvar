import torch
from torch import nn
import torch.nn.functional as F
from .vqvae_single import VQVAE_Single
from .transformer import TransformerSimple, TransformerCross, BlockCross, BlockSimple

class TransformerCrossTwoKey(nn.Module):

    def __init__(self, dim=256, num_heads=8, num_blocks=2, image_cross_layers=[0, 2], click_cross_layers=[0, 2]):
        super(TransformerCrossTwoKey, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.image_cross_layers = image_cross_layers
        self.click_cross_layers = click_cross_layers

        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            if i in self.image_cross_layers or i in self.click_cross_layers:
                block = BlockCross(dim=dim, num_heads=self.num_heads)
            else:
                block = BlockSimple(dim=dim, num_heads=self.num_heads)
            self.blocks.append(block)
    
    def forward(self, x, image_embedding, prompt_embedding):
        """
        x: (B, L, C)
        image_embedding: (B, Lk, C)
        prompt_embedding: (B, Lp, C)
        """
        for i, block in enumerate(self.blocks):
            if i in self.image_cross_layers:
                x = block(x, image_embedding)
            elif i in self.click_cross_layers:
                x = block(x, prompt_embedding)
            else:
                x = block(x)
        return x

class MaskGIT(nn.Module):

    def __init__(self, vqvae: VQVAE_Single, 
                 image_size=256, patch_size=8, dim=256, 
                 num_heads=8, num_blocks=6, vocab_size=4096, 
                 image_cross_layers=[0, 3], click_cross_layers=[1, 4],
                 freeze_vqvae: bool = True):
        super().__init__()
        self.image_size = image_size
        self.vqvae = vqvae
        self.dim = dim
        self.patch_size = patch_size
        self.vocab_size = vocab_size
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)

        # self.pos_embedding = nn.Embedding(self.num_patches, dim)
        self.token_embedding = nn.Embedding(self.vocab_size + 1, dim)

        self.transformer = TransformerCrossTwoKey(
            dim=dim, num_heads=num_heads, num_blocks=num_blocks, 
            image_cross_layers=image_cross_layers, click_cross_layers=click_cross_layers
            )
        self.classifier = nn.Linear(dim, vocab_size)

        if freeze_vqvae:
            for param in self.vqvae.parameters():
                param.requires_grad = False
    
    def forward(self, x, x_pos, image_embedding, prompt_embedding, dense_pe):
        """
        x: (B, L) int
        x_pos: (B, L) int
        image_embedding: (B, L, C)
        prompt_embedding: (B, Lp, C)
        dense_pe: (1, L, C)
        """
        B, L = x.shape
        C = self.dim
        Lp = prompt_embedding.shape[1]

        tokens = self.token_embedding(x) # (B, L, C)
        # tokens = tokens + self.pos_embedding(x_pos)
        dense_pe = dense_pe.repeat(B, 1, 1)
        x_pos_exp = x_pos.view(B, L, 1).expand(-1, -1, C)
        dense_pe_shuffled = torch.gather(dense_pe, dim=1, index=x_pos_exp)
        tokens = tokens + dense_pe_shuffled
        # image_tokens = image_embedding + self.pos_embedding(torch.arange(image_embedding.shape[1], device=x.device))
        image_tokens = image_embedding
        logits = self.transformer(tokens, image_tokens, prompt_embedding)
        logits = self.classifier(logits)
        return logits
    
if __name__ == "__main__":
    vqvae = VQVAE_Single()
    vqvae.to('cuda')
    maskgit = MaskGIT(vqvae)
    maskgit.to("cuda")
    x = torch.randint(0, 100, (1, 100)).to(device='cuda')
    x_pos = torch.arange(100).to(device='cuda')
    image_embedding = torch.randn(1, 16*16, 256).to(device='cuda')
    prompt_embedding = torch.randn(1, 3, 256).to(device='cuda')
    print(maskgit(x, x_pos, image_embedding, prompt_embedding).shape)

    # model profile
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            maskgit(x, x_pos, image_embedding, prompt_embedding)
    print(prof.key_averages().table(sort_by="cuda_time_total"))