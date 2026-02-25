# Minimal PyTorch Transformer for 10-Digit Addition

A PyTorch replication of the minimal transformer for 10-digit integer addition,
achieving **99.65% exact-match accuracy with 763 parameters**.

This work replicates and extends [yhavinga/gpt-acc-jax](https://github.com/yhavinga/gpt-acc-jax),
which reported 99.69% accuracy with 777 parameters using JAX/Flax. We match this
result in PyTorch with a slightly leaner architecture (no final LayerNorm), and
document an interesting grokking phenomenon observed during training.

---

## Result

| Model | Parameters | Accuracy | Framework |
|-------|-----------|----------|-----------|
| [yhavinga/gpt-acc-jax](https://github.com/yhavinga/gpt-acc-jax) pico-7d-ff14-lr02 | 777 | 99.69% | JAX/Flax |
| **This work** | **763** | **99.65%** | **PyTorch** |

### Parameter Breakdown

```
Token embeddings:     14 × 7  =  98
Position embeddings:  35 × 7  = 245
LayerNorm 1 (w+b):     7 × 2  =  14
QKV projection:        7 × 21 = 147
Attention output:      7 × 7  =  49
LayerNorm 2 (w+b):     7 × 2  =  14
FFN up:                7 × 14 =  98
FFN down:             14 × 7  =  98
Output proj (tied):            =   0   ← shared with token embeddings
─────────────────────────────────────
TOTAL:                           763
```

The 14-parameter difference from the JAX model is the absence of a final
LayerNorm before the output projection.

---

## The Grokking Jump

The most striking observation from this training run was a sudden **grokking**
event during the second training phase:

```
step   6000  full_acc  95.6%
step   7000  full_acc  78.4%   ← temporary regression
step   8000  full_acc  99.6%   ← sudden grokking jump
```

The model trained for **400,000 steps** (curriculum + full 10-digit) and
plateaued at ~85% accuracy. A second training pass with a fresh cosine LR
schedule triggered grokking at step 8,000 — a jump from 78% to 99.6% in a
single 1,000-step window.

This suggests the original long training run had built the correct internal
representations, but the learning rate had flatlined at its minimum (0.002).
Restarting with a proper cosine decay to near-zero allowed those representations
to "click into place."

---

## Architecture

A minimal decoder-only transformer:

- **Layers:** 1
- **Hidden dim (d_model):** 7
- **Attention heads:** 1
- **FFN dim:** 14 (2× expansion ratio)
- **Vocab size:** 14 (digits 0–9, `+`, `=`, `<PAD>`, `<EOS>`)
- **Context length:** 35
- **Tied embeddings:** Yes (input and output share the same matrix)
- **FFN bias:** No
- **Positional embeddings:** Learned

### Data Format

```
Input:   0000000042+0000000007=       (operands normal order, zero-padded to 10 digits)
Target:  94000000000                  (sum reversed, zero-padded to 11 digits)
```

The sum is reversed so that the ones digit (and carry) comes first, aligning
carry propagation with the autoregressive generation direction.

---

## Training

### Phase 1: Curriculum (400k steps)

Three-phase curriculum matching the JAX training setup:

| Phase | Digits | Steps |
|-------|--------|-------|
| 1 | 1–3 | 20,000 |
| 2 | 1–6 | 40,000 |
| 3 | 1–10 | 340,000 |

Optimizer: AdamW, lr=0.02, weight_decay=0.1, batch_size=128, cosine schedule.

### Phase 2: Second-pass (8k steps, early stopped)

After plateauing at ~85%, a second training pass was run with:
- Fresh cosine LR schedule (peak 0.005 → ~0.0001)
- batch_size=256
- weight_decay=0.01
- Full 10-digit problems only

The model grokked at step 8,000 and training stopped early at 99.6%.

---

## Usage

### Training from scratch

```bash
python pytorch_model_fixed.py
```

This runs the full 400k-step curriculum and saves `gpt_addition.pt`.

### Continuing from a checkpoint

```bash
python continue_training.py
```

Loads `gpt_addition.pt` and runs the second-pass schedule.
Saves best checkpoint to `gpt_addition_v2.pt`.

### Interactive inference

```python
import torch
from pytorch_model_fixed import GPTAddition, add

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPTAddition().to(device)
model.load_state_dict(torch.load("gpt_addition_v2.pt", map_location=device))

add(model, device, "1234567890", "9876543210")
# 1234567890 + 9876543210 = 11111111100  ✓
```

---

## Requirements

```
torch>=2.0
```

No other dependencies. Training on CPU is feasible but slow (~1 day for the
full 400k-step run). A GPU reduces this to ~1–2 hours.

---

## Citation

If you use this work, please also cite the original JAX implementation:

```bibtex
@misc{yhavinga2024gptaccjax,
  author       = {yhavinga},
  title        = {Training the Smallest Transformer for 10-Digit Addition},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/yhavinga/gpt-acc-jax}}
}
```

---

## License

MIT
