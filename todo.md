TODO List
---

12.20

Now converged.

train script:

```
python train_scripts/train_simple_var.py --device 'cuda' --outdir out/simple_var_4_debug --num_iters 100 --save_interval 50 --val_
iters 16
```

---

12.8

Reimplemented var as `simple_var`. Finally get it converged...

Next:

1. add cross attention to simple_var
    - prompt encoder
    - image feature adapter
    - positional embedding from sam prompt encoder [!important]
2. train on more than one images

---

11.11

1. do unit test (IMPORTANT)
2. profile new model

unit test (easy to hard)

- [x] clicker
- [x] sam
- [x] tinyvit.py (mobile sam)
- [x] vqvae_single
  - [x] img2img, idx2img, img2idx
  - [x] to model input (likely correct) (considering a simpler one)
- [ ] flex_maskvar
  - [x] self attention (v2)
  - [x] cross attention (v2) (likely correct)
  - [ ] positional embedding
  - [ ] prompt encoder pe
  - [ ] image feature adapter
  - [ ] autogressive inference
  - [ ] layer norm (should be added)
- [ ] loss function
- [ ] maskvar (low priority)

----

10.4

0. use flex attention
1. use windowed attention for cross-attention
2. use HRSAM as the encoder
3. profile new model