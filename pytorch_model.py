import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random


# ── Vocabulary ────────────────────────────────────────────────────────────────
# Matches JAX config.py TOKENS exactly:
# 0-9: digits, 10: '+', 11: '=', 12: <PAD>, 13: <EOS>
VOCAB  = "0123456789+= "          # indices 0-12; ' ' used for PAD display
PAD_ID = 12
EOS_ID = 13
CTX    = 35

def encode(s):
    return [VOCAB.index(c) for c in s]

def decode(ids):
    return "".join(
        VOCAB[i] if i < len(VOCAB) else ""
        for i in ids if i != PAD_ID and i != EOS_ID
    )


# ── Data ──────────────────────────────────────────────────────────────────────
def make_example(min_digits=1, max_digits=10):
    """Generate a single training example matching the JAX data.py format.

    Key details from the JAX code:
    - Operands are in NORMAL order, zero-padded to 10 digits
    - Only the SUM is reversed (to align carry propagation with generation)
    - n_digits is sampled per example from [min_digits, max_digits]
    - Always pads to max_digits=10 regardless of actual n_digits
    - Uses separate input and target arrays (target is shifted by 1)
    """
    n_digits = random.randint(min_digits, max_digits)
    max_val = 10 ** n_digits
    a = random.randint(0, max_val - 1)
    b = random.randint(0, max_val - 1)
    c = a + b

    # Operands: normal order, zero-padded to 10 digits
    a_str = str(a).zfill(10)
    b_str = str(b).zfill(10)
    # Sum: reversed, zero-padded to 11 digits
    c_str = str(c).zfill(11)[::-1]

    input_str  = f"{a_str}+{b_str}="       # 10+1+10+1 = 22 chars
    target_str = c_str                       # 11 chars

    input_tokens  = encode(input_str)        # length 22
    target_tokens = encode(target_str)       # length 11

    # Matching JAX generate_batch():
    #   full_seq    = input_tokens + target_tokens           (len 33)
    #   full_target = input_tokens[1:] + target_tokens + [EOS]  (len 33)
    full_seq    = input_tokens + target_tokens
    full_target = input_tokens[1:] + target_tokens + [EOS_ID]

    seq_len = len(full_seq)                  # 33

    # Pad to CTX
    ids = full_seq    + [PAD_ID] * (CTX - seq_len)
    tgt = full_target + [PAD_ID] * (CTX - len(full_target))

    # Mask: loss only on answer portion (after '=') + EOS
    # In the JAX code:
    #   eq_pos = len(input_tokens) - 1   → 21  (the '=' token)
    #   mask[eq_pos : eq_pos + len(target_tokens) + 1] = 1.0
    # This means mask[21..32] = 1 (12 positions: 11 answer digits + EOS)
    eq_pos   = len(input_tokens) - 1          # 21
    n_target = len(target_tokens) + 1         # 12 (11 digits + EOS)
    mask = ([0.0] * eq_pos
          + [1.0] * n_target
          + [0.0] * (CTX - eq_pos - n_target))

    assert len(ids)  == CTX, f"ids length {len(ids)} != {CTX}"
    assert len(tgt)  == CTX, f"tgt length {len(tgt)} != {CTX}"
    assert len(mask) == CTX, f"mask length {len(mask)} != {CTX}"

    return ids, tgt, mask


def make_batch(batch_size, min_digits=1, max_digits=10):
    examples = [make_example(min_digits, max_digits) for _ in range(batch_size)]
    ids   = torch.tensor([e[0] for e in examples])
    tgts  = torch.tensor([e[1] for e in examples])
    masks = torch.tensor([e[2] for e in examples])
    return ids, tgts, masks


# ── Model ─────────────────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, ctx_len):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        mask = torch.triu(torch.ones(ctx_len, ctx_len), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=2)
        scale   = 1.0 / math.sqrt(k.size(-1))
        scores  = (q @ k.transpose(-2, -1)) * scale
        scores  = scores.masked_fill(self.mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return self.proj(weights @ v)


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, ctx_len):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, ctx_len)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = FFN(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTAddition(nn.Module):
    def __init__(self, vocab_size=14, ctx_len=CTX,
                 d_model=7, n_heads=1, d_ff=14, n_layers=1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(ctx_len, d_model)
        self.blocks  = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, ctx_len)
            for _ in range(n_layers)
        ])
        self.output_proj        = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.tok_emb.weight   # tied weights

        self._init_weights()

    def _init_weights(self):
        """Match Flax default initialization:
        - LayerNorm: scale=1, bias=0
        - Embeddings: normal(std=0.02)
        - Linear weights: Lecun normal (1/sqrt(fan_in))
        """
        for name, p in self.named_parameters():
            if 'ln' in name:
                if 'weight' in name:
                    nn.init.ones_(p)
                elif 'bias' in name:
                    nn.init.zeros_(p)
            elif 'tok_emb' in name or 'pos_emb' in name:
                nn.init.normal_(p, std=0.02)
            elif p.dim() >= 2:
                nn.init.kaiming_normal_(p, mode='fan_in', nonlinearity='linear')
            else:
                nn.init.zeros_(p)

    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


# ── Loss ──────────────────────────────────────────────────────────────────────
def masked_cross_entropy(logits, targets, mask):
    """Masked cross-entropy matching the JAX train_step.

    logits:  [B, T, vocab]
    targets: [B, T]  (pre-shifted targets from make_example)
    mask:    [B, T]
    """
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        reduction="none"
    ).view(targets.size(0), -1)
    return (loss * mask).sum() / (mask.sum() + 1e-8)


# ── Inference ─────────────────────────────────────────────────────────────────
def generate(model, a, b, device):
    """Generate answer autoregressively.

    Input format: operands in normal order, zero-padded to 10 digits.
    Output: reversed sum digits, decoded back to integer.
    """
    model.eval()
    a_str  = str(a).zfill(10)
    b_str  = str(b).zfill(10)
    prompt = f"{a_str}+{b_str}="
    idx    = torch.tensor([encode(prompt)]).to(device)
    gen    = []
    with torch.no_grad():
        for _ in range(12):
            next_id = model(idx)[0, -1, :].argmax().item()
            if next_id == EOS_ID or next_id == PAD_ID or idx.size(1) >= CTX:
                break
            if 0 <= next_id <= 9:
                gen.append(next_id)
            idx = torch.cat([idx, torch.tensor([[next_id]]).to(device)], dim=1)
    model.train()
    # gen contains reversed digits of the sum; reverse back to normal order
    rev_str = "".join(str(d) for d in gen)
    try:
        return str(int(rev_str[::-1])) if rev_str else "0"
    except ValueError:
        return "0"


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, device, n_examples=200, n_digits=10):
    model.eval()
    correct = 0
    for _ in range(n_examples):
        a = random.randint(0, 10**n_digits - 1)
        b = random.randint(0, 10**n_digits - 1)
        pred = generate(model, a, b, device)
        correct += int(pred == str(a + b))
    model.train()
    return correct / n_examples


@torch.no_grad()
def evaluate_phase(model, device, n_examples=200, n_digits=3):
    """Evaluate accuracy on problems within the current phase difficulty."""
    model.eval()
    correct = 0
    for _ in range(n_examples):
        nd = random.randint(1, n_digits)
        a = random.randint(0, 10**nd - 1)
        b = random.randint(0, 10**nd - 1)
        pred = generate(model, a, b, device)
        correct += int(pred == str(a + b))
    model.train()
    return correct / n_examples


def add(model, device, a, b):
    a, b = int(a), int(b)
    answer = generate(model, a, b, device)
    print(f"{a} + {b} = {answer}  (expected {a+b}, "
          f"{'✓' if answer == str(a+b) else '✗'})")


# ── Curriculum ────────────────────────────────────────────────────────────────
# Default from JAX config.py (tuned for 128d/4L baseline model)
CURRICULUM_PHASES = [
    # (min_digits, max_digits, n_steps)
    (1, 3,   5_000),
    (1, 6,  10_000),
    (1, 10, 30_000),
]

# Scaled up for the tiny 777-param pico model (needs longer to grok)
# v1: 200k total — reached ~55% by step 170k, still improving
# v2: 400k total — give it more time at full difficulty to grok
CURRICULUM_PHASES_PICO = [
    (1, 3,   20_000),
    (1, 6,   40_000),
    (1, 10, 340_000),
]


# ── Training loop ─────────────────────────────────────────────────────────────
def train(use_pico_schedule=True):
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model      = GPTAddition().to(device)
    total      = sum(p.numel() for p in model.parameters())
    batch_size = 128

    phases  = CURRICULUM_PHASES_PICO if use_pico_schedule else CURRICULUM_PHASES
    n_steps = sum(s for _, _, s in phases)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.1)
    warmup    = int(0.05 * n_steps)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / (n_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Parameters: {total}  |  Device: {device}")
    print(f"Training for {n_steps:,} steps with {len(phases)}-phase curriculum\n")
    for i, (lo, hi, s) in enumerate(phases):
        print(f"  Phase {i+1}: digits {lo}-{hi}, {s:,} steps")
    print()

    # ── Diagnostics: verify data pipeline ──
    print("── Data pipeline verification ──")
    # Trace a specific example
    a_test, b_test = 42, 7
    c_test = a_test + b_test
    a_s = str(a_test).zfill(10)
    b_s = str(b_test).zfill(10)
    c_s = str(c_test).zfill(11)[::-1]
    print(f"  a={a_test} b={b_test} c={c_test}")
    print(f"  a_str='{a_s}' b_str='{b_s}' c_reversed='{c_s}'")
    print(f"  input_str='{a_s}+{b_s}='")
    print(f"  target_str='{c_s}'")

    inp_tok = encode(f"{a_s}+{b_s}=")
    tgt_tok = encode(c_s)
    full_seq = inp_tok + tgt_tok
    full_tgt = inp_tok[1:] + tgt_tok + [EOS_ID]
    eq_pos = len(inp_tok) - 1
    print(f"  input_tokens  len={len(inp_tok)}: {inp_tok}")
    print(f"  target_tokens len={len(tgt_tok)}: {tgt_tok}")
    print(f"  full_seq  len={len(full_seq)}")
    print(f"  full_tgt  len={len(full_tgt)}")
    print(f"  eq_pos={eq_pos}")
    print(f"  At eq_pos, full_seq[{eq_pos}]={full_seq[eq_pos]} "
          f"(should be 11='=')")
    print(f"  At eq_pos, full_tgt[{eq_pos}]={full_tgt[eq_pos]} "
          f"(first answer digit, should be {tgt_tok[0]})")
    print(f"  full_tgt[{eq_pos}:{eq_pos+12}] = {full_tgt[eq_pos:eq_pos+12]}")
    print()

    # Show a few batch examples
    ids, tgts, mask = make_batch(4, min_digits=1, max_digits=3)
    for i in range(2):
        inp = decode(ids[i].tolist())
        tgt_str = "".join(
            str(tgts[i, j].item()) if tgts[i, j].item() < 10 else
            ("+" if tgts[i, j].item() == 10 else
             ("=" if tgts[i, j].item() == 11 else
              ("P" if tgts[i, j].item() == 12 else "E")))
            for j in range(CTX)
        )
        m = mask[i].tolist()
        print(f"Example {i}:")
        print(f"  input : {inp.strip()}")
        print(f"  target: {tgt_str}")
        print(f"  mask  : {''.join(str(int(v)) for v in m)}")
        print(f"  mask 1-count: {int(sum(m))}  "
              f"positions: {[j for j,v in enumerate(m) if v]}")
    print()

    global_step = 0
    for phase_idx, (min_d, max_d, phase_steps) in enumerate(phases):
        print(f"\n=== Phase {phase_idx+1}: digits {min_d}-{max_d}, "
              f"{phase_steps:,} steps ===")

        for step in range(1, phase_steps + 1):
            ids, tgts, mask = make_batch(batch_size,
                                         min_digits=min_d,
                                         max_digits=max_d)
            ids  = ids.to(device)
            tgts = tgts.to(device)
            mask = mask.to(device)

            logits = model(ids)
            loss   = masked_cross_entropy(logits, tgts, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1

            if global_step % 1000 == 0:
                # Evaluate both on current phase difficulty and full 10-digit
                phase_acc = evaluate_phase(model, device,
                                           n_examples=200,
                                           n_digits=max_d)
                full_acc  = evaluate(model, device,
                                     n_examples=200,
                                     n_digits=10)
                # Show a sample from current phase
                nd = random.randint(min_d, max_d)
                a = random.randint(0, 10**nd - 1)
                b = random.randint(0, 10**nd - 1)
                sample = generate(model, a, b, device)
                print(f"  step {global_step:>6}  loss {loss.item():.4f}  "
                      f"phase_acc {phase_acc*100:.1f}%  "
                      f"full_acc {full_acc*100:.1f}%  "
                      f"lr {scheduler.get_last_lr()[0]:.5f}")
                print(f"           {a}+{b}={a+b} got={sample} "
                      f"{'✓' if sample == str(a+b) else '✗'}")

    print("\n── Final evaluation ──")
    for nd in [3, 6, 10]:
        acc = evaluate_phase(model, device, n_examples=500, n_digits=nd)
        print(f"  digits 1-{nd}: {acc*100:.2f}%")
    acc = evaluate(model, device, n_examples=1000, n_digits=10)
    print(f"  Full 10-digit exact-match: {acc*100:.2f}%")
    torch.save(model.state_dict(), "gpt_addition.pt")
    print("Saved to gpt_addition.pt")
    return model


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model  = train(use_pico_schedule=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    add(model, device, "1234567890", "9876543210")
    add(model, device, "9999999999", "1")
    add(model, device, "5000000000", "5000000000")
    add(model, device, "42", "7")
    add(model, device, "999", "1")
    add(model, device, "123456", "654321")
