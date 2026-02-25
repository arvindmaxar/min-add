"""
continue_training.py
────────────────────
Drop this into the same folder as pytorch_model_fixed.py and run:

    python continue_training.py

It loads gpt_addition.pt (saved at the end of the first run) and does a
second-pass phase-3-only training with:
  - Fresh cosine LR schedule (0.005 → ~0.0001)
  - Larger batch size (256)
  - Lower weight decay (0.01)
  - 200k additional steps, all on 10-digit problems
"""

import math
import random
import torch

# Import everything from the original file
from pytorch_model import (
    GPTAddition,
    make_batch,
    masked_cross_entropy,
    evaluate,
    evaluate_phase,
    generate,
    add,
)

CHECKPOINT = "gpt_addition.pt"
SAVE_PATH  = "gpt_addition_v2.pt"


def continue_training(
    checkpoint_path: str = CHECKPOINT,
    n_steps:         int  = 200_000,
    batch_size:      int  = 256,
    peak_lr:         float = 0.005,
    min_lr_frac:     float = 0.02,   # floor = peak_lr * min_lr_frac
    weight_decay:    float = 0.01,
    warmup_frac:     float = 0.02,
    eval_every:      int  = 1_000,
    save_path:       str  = SAVE_PATH,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load checkpoint ────────────────────────────────────────────────────────
    model = GPTAddition().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint from '{checkpoint_path}'")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total}  |  Device: {device}\n")

    # ── Baseline eval before we touch anything ─────────────────────────────────
    print("── Baseline accuracy (before continued training) ──")
    for nd in [3, 6, 10]:
        acc = evaluate_phase(model, device, n_examples=500, n_digits=nd)
        print(f"  digits 1-{nd}: {acc*100:.2f}%")
    acc10 = evaluate(model, device, n_examples=1_000, n_digits=10)
    print(f"  Full 10-digit exact-match: {acc10*100:.2f}%\n")

    # ── Optimizer & scheduler ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        weight_decay=weight_decay,
    )
    warmup = int(warmup_frac * n_steps)

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(n_steps - warmup, 1)
        cosine   = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_frac + (1.0 - min_lr_frac) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Continuing training: {n_steps:,} steps on 10-digit problems")
    print(f"  batch_size={batch_size}, peak_lr={peak_lr}, "
          f"weight_decay={weight_decay}, warmup={warmup}\n")

    # ── Training loop ──────────────────────────────────────────────────────────
    best_acc = acc10
    best_step = 0

    for step in range(1, n_steps + 1):
        ids, tgts, mask = make_batch(batch_size, min_digits=1, max_digits=10)
        ids   = ids.to(device)
        tgts  = tgts.to(device)
        mask  = mask.to(device)

        logits = model(ids)
        loss   = masked_cross_entropy(logits, tgts, mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % eval_every == 0:
            phase_acc = evaluate_phase(model, device, n_examples=200, n_digits=10)
            full_acc  = evaluate(model, device, n_examples=500, n_digits=10)

            nd = random.randint(1, 10)
            a  = random.randint(0, 10**nd - 1)
            b  = random.randint(0, 10**nd - 1)
            sample = generate(model, a, b, device)

            print(f"  step {step:>6}  loss {loss.item():.4f}  "
                  f"phase_acc {phase_acc*100:.1f}%  "
                  f"full_acc {full_acc*100:.1f}%  "
                  f"lr {scheduler.get_last_lr()[0]:.5f}")
            print(f"           {a}+{b}={a+b} got={sample} "
                  f"{'✓' if sample == str(a+b) else '✗'}")

            # Save best checkpoint
            if full_acc > best_acc:
                best_acc  = full_acc
                best_step = step
                torch.save(model.state_dict(), save_path)
                print(f"           ★ New best {best_acc*100:.1f}% — saved to {save_path}")

            # Early-exit if we've grokked
            if full_acc >= 0.99:
                print(f"\n🎉 Reached {full_acc*100:.1f}% at step {step} — stopping early!")
                break

    # ── Final evaluation ───────────────────────────────────────────────────────
    print(f"\n── Final evaluation (best was {best_acc*100:.2f}% at step {best_step}) ──")

    # Load best weights for final eval
    if best_step > 0:
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"   (loaded best checkpoint from step {best_step})")

    for nd in [3, 6, 10]:
        acc = evaluate_phase(model, device, n_examples=1_000, n_digits=nd)
        print(f"  digits 1-{nd}: {acc*100:.2f}%")
    acc = evaluate(model, device, n_examples=2_000, n_digits=10)
    print(f"  Full 10-digit exact-match: {acc*100:.2f}%")

    # Spot-check some specific tricky cases
    print("\n── Spot checks ──")
    for a, b in [
        (9999999999, 1),
        (5000000000, 5000000000),
        (1234567890, 9876543210),
        (9999999999, 9999999999),
        (1000000000, 1000000000),
        (123456789,  987654321),
    ]:
        add(model, device, str(a), str(b))

    torch.save(model.state_dict(), save_path)
    print(f"\nSaved final model to '{save_path}'")
    return model


if __name__ == "__main__":
    model = continue_training()
