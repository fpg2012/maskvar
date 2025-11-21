import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from einops import rearrange
from tqdm import tqdm
import json
import argparse

from maskvar.models.basic_var import SelfAttention_v2
from maskvar.models.basic_var import CrossAttnBlock
from maskvar.utils.loss import FocalLossGeneral

device = 'cuda'

class DummyMLP(nn.Module):

    def __init__(self, embed_dim=256, mlp_ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * mlp_ratio)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(embed_dim * mlp_ratio, embed_dim)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class DummyAttention(nn.Module):

    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.proj = nn.Linear(embed_dim, embed_dim * 3)
    
    def forward(self, x, block_mask):
        B, L, C = x.shape
        qkv = self.proj(x) # B L 3*C
        qkv = rearrange(qkv, 'B L (QKV H c) -> QKV B H L c', QKV=3, H=self.num_heads, c=self.head_dim)
        q, k, v = qkv.unbind(dim=0) # B H L c
        y = flex_attention(q, k, v, block_mask=block_mask) # B H L c
        return rearrange(y, 'B H L c -> B L (H c)')

class DummyTransformerBlock(nn.Module):
    
    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.ffn = DummyMLP(embed_dim=dim, mlp_ratio=4)
        self.attn = SelfAttention_v2(
            block_idx=0,
            embed_dim=dim,
            num_heads=4,
            proj_drop=0.1,
            attn_l2_norm=False,
        )
        # self.attn = DummyAttention(embed_dim=dim, num_heads=num_heads)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        # self.layer_norm = nn.Identity()
    
    def forward(self, x: torch.Tensor, cond=None, prompt_cond=None, block_mask=None):
        """
        x: [B, L, C]
        """
        x = x + self.attn(self.layer_norm1(x), block_mask=block_mask)
        x = x + self.ffn(self.layer_norm2(x))
        return x

class DummyTransformer(nn.Module):
    def __init__(self, dim=256, depth=2, vocab_size=50, device='cpu', max_len=20, num_heads=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            DummyTransformerBlock(dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        # self.blocks = nn.ModuleList([
        #     CrossAttnBlock(
        #         cond_dim=dim, shared_aln=False,
        #         block_idx=block_idx, embed_dim=dim, norm_layer=nn.LayerNorm, num_heads=num_heads, mlp_ratio=4,
        #         drop=0, drop_path=0, last_drop_p=0,
        #         attn_l2_norm=False,
        #     )
        #     for block_idx in range(depth)
        # ])
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed = nn.Embedding(self.vocab_size, dim) # pad=10, sos=11, eos=12
        self.cls = nn.Linear(dim, self.vocab_size)

        self.device = device
        self.max_len = max_len
        self.block_mask = None
        self.pos_emb = nn.Embedding(max_len, dim)

    def init_block_mask(self, length=None):
        # causal mask
        def mask_mod(b, h, q_idx, k_idx) -> bool:
            return (q_idx >= k_idx)
        
        self.block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=self.max_len, KV_LEN=self.max_len, device=device)
        
    
    def forward(self, x: torch.Tensor):
        """
        x: (B, L)
        """

        if self.block_mask is None:
            self.init_block_mask()
        
        x = self.embed(x)
        x = x + self.pos_emb(torch.arange(self.max_len, device=device))

        for block in self.blocks:
            x = block(x, cond=None, prompt_cond=None, block_mask=self.block_mask)
        logits = self.cls(x)
        return logits
    
    @torch.no_grad()
    def autogressive_infer(self, x):
        """
        x: (B, L)
        """
        B, L = x.shape
        x = self.embed(x)
        x = x + self.pos_emb(torch.arange(L, device=device))

        for block in self.blocks:
            x = block(x, cond=None, prompt_cond=None, block_mask=None)
        logits = self.cls(x)
        return logits

def overfit(model, loss, n_epoch):
    model.train()
    # dummy_seq = [[11, 1, 0, 3, 8, 4, 3, 2, 1, 4, 3, 0, 4, 2, 3, 4, 12, 10, 10, 10],
    #              [11, 4, 3, 3, 3, 1, 4, 0, 1, 8, 3, 2, 4, 0, 2, 4, 12, 10, 10, 10],
    #              [11, 1, 8, 3, 4, 3, 3, 4, 3, 2, 0, 4, 0, 4, 2, 1, 12, 10, 10, 10],
    #              [11, 3, 2, 8, 1, 3, 3, 0, 2, 4, 4, 4, 4, 1, 3, 0, 12, 10, 10, 10],]
    dummy_seq = [[11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 12, 10, 10, 10],
                 [11, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 2, 12, 10, 10, 10],
                 [11, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 2, 12, 10, 10, 10],
                 [11, 1, 1, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 12, 10, 10, 10],
                ]
    valid_len = 17
    dummy_seq = torch.tensor(dummy_seq, dtype=torch.long, device=device)
    print('dummy_seq.shape:', dummy_seq.shape)

    opt = AdamW(model.parameters(), lr=1e-3)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        # display loss in tqdm bar
        loss_seq = []
        with tqdm(range(n_epoch)) as tqdm_instance:
            for epoch in tqdm_instance:
                opt.zero_grad()
                logits = model(dummy_seq)
                L = loss(logits[:, :valid_len-1, :].permute(0, 2, 1), dummy_seq[:, 1:valid_len])
                Lmean = L.mean()
                Lmean.backward()
                opt.step()
                
                acc = (logits[:, :valid_len-1, :].argmax(dim=-1) == dummy_seq[:, 1:valid_len]).float().mean()

                # 在进度条中显示损失
                loss_seq.append(Lmean.item())
                tqdm_instance.set_postfix(loss=f"{Lmean.item():.4f}", acc=f"{acc.item()*100:.2f}%")
        return loss_seq

def autogressive_infer(model, device):
    model.eval()
    with torch.no_grad():
        x = [11]
        for _ in range(20):
            logits = model.autogressive_infer(torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0))
            # sample from logits
            next_token = torch.multinomial(F.softmax(logits[:, -1], dim=-1), 1).item()
            x.append(next_token)
            if next_token == 12:
                break
    return x
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--exp', type=str, required=True)
    args = parser.parse_args()

    n_epoch = args.n_epoch
    device = args.device
    exp_name = args.exp

    loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='none')
    # loss = FocalLossGeneral(alpha=0.1, gamma=2.0, label_smooth=0, reduction='none')

    model = DummyTransformer(depth=1, dim=128, vocab_size=50, device=device, num_heads=2)
    model.to(device)
    model.train()
    model = torch.compile(model)

    loss_seq = overfit(model, loss, n_epoch)
    with open(f'loss_{exp_name}.json', 'w') as f:
        json.dump(loss_seq, f)
    # torch.save(model.state_dict(), f'../ckpt/{exp_name}.pth')
    for i in range(20):
        test_out = autogressive_infer(model, device)
        print(test_out)
        
