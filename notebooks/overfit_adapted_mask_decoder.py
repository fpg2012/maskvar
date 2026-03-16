import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.attention.flex_attention import create_block_mask
from einops import rearrange, repeat
from tqdm import tqdm
import json
import argparse

from maskvar.maskseg_build_everything import builder_map
from maskvar.models.simple_ar.adapted_mask_decoder import AdaptedMaskDecoder
from maskvar.models.simple_ar.adapted_twt import AdaptedTwoWayTransformer

device = 'cuda'

class DummyTester(nn.Module):

    def __init__(self, amd: AdaptedMaskDecoder, max_len=100, vocab_size=1024, embed_dim=256, device='cuda'):
        super().__init__()
        self.amd = amd
        self.cls = nn.Linear(embed_dim, 1024)
        self.max_len = max_len
        self.block_mask = None
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.device = device

    def init_block_mask(self):
        """causal mask"""
        def mask_mod(b, h, q_idx, k_idx):
            """Mask function: returns True if query and key tokens are at the same scale."""
            return q_idx <= k_idx

        # Create block mask using PyTorch's flex_attention utility
        self.block_mask = create_block_mask(
            mask_mod,
            B=None,  # Batch dimension (None means support any batch size)
            H=None,  # Head dimension (None means support any number of heads)
            Q_LEN=self.max_len,
            KV_LEN=self.max_len,
            device=self.device
        )
    
    def forward(self, x, x_pe, image_embedding, image_pe, point_embedding):
        x = self.emb(x)
        x = x + x_pe
        qs, qm = self.amd(
            image_embeddings=image_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=None,
            dense_prompt_embeddings=None,
            mask_tokens=x,
            mask_tokens_pe=x_pe,
            self_attn_mask=self.block_mask,
        )
        logits = self.cls(qm)
        return logits
    
    @torch.no_grad()
    def ar_inference(self):
        sos = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        pass
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    max_len = 50
    batch_size = 4
    vocab_size = 5
    embed_dim = 256
    h = w = 32

    # create random data
    dummy_seqs = torch.randint(1, vocab_size, (batch_size, max_len-1)).to(device)
    # prepend sos
    sos = torch.full((batch_size, 1), 0, dtype=torch.long).to(device)
    dummy_seqs = torch.cat([sos, dummy_seqs], dim=1).to(device)

    # random image_embedding (B, Lk, C)
    image_embedding = torch.rand((batch_size, h*w, embed_dim)).to(device)
    # random point_embedding (B, Lq, C)
    point_embedding = torch.rand((1, 3, embed_dim)).to(device)
    point_embedding = repeat(point_embedding, '1 Lq C -> B Lq C', B=batch_size)
    # random x pe (1, max_len, C)
    x_pe = torch.rand((1, max_len, embed_dim)).to(device)
    x_pe = torch.zeros_like(x_pe) # try no pe
    # x_pe = repeat(x_pe, '1 L C -> B L C', B=batch_size)
    # random image pe (1, Lk, C)
    image_pe = torch.rand((1, h*w, embed_dim)).to(device)
    image_pe = torch.zeros_like(image_pe) # try no pe
    # image_pe = repeat(image_pe, '1 L C -> B L C', B=batch_size)

    twt = AdaptedTwoWayTransformer(
        depth=2,
        embedding_dim=256,
        num_heads=4,
        mlp_dim=2048
    )
    adapted_mask_decoder = AdaptedMaskDecoder(
        transformer_dim=256,
        transformer=twt,
    )
    model = DummyTester(adapted_mask_decoder, max_len=max_len, vocab_size=vocab_size, embed_dim=embed_dim, device=device)
    model.to(device)
    model.init_block_mask()

    # log random data
    print(f"dummy_seqs: {dummy_seqs}")

    optimizer = AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.n_epoch):
        optimizer.zero_grad()
        logits = model(dummy_seqs, x_pe, image_embedding, image_pe, point_embedding)
        logits = logits[:, :-1, :]
        acc = (logits.argmax(dim=-1) == dummy_seqs[:, 1:]).float().mean()

        logits = rearrange(logits, 'b l c -> b c l')
        loss = F.cross_entropy(logits, dummy_seqs[:, 1:])
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: loss={loss.item():.4f}, acc={acc.item():.4f}")

    # inference test
    pass
