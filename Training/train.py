"""Обучение чат-бота.  Запуск:  python3 train.py [--epochs 300] [--hidden 128]"""
import argparse
import time
import numpy as np
from model import GRULM, Adam, PAD, normalize


def load_pairs(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "|" not in line:
                continue
            q, a = line.split("|", 1)
            pairs.append((normalize(q), a.strip()))
    return pairs


def build_batch(texts, bounds, c2i):
    """Готовим батч: X (вход), Y (цель, сдвинута на 1 символ), M (веса потерь)."""
    L = max(len(t) for t in texts) - 1
    B = len(texts)
    X = np.zeros((B, L), dtype=int)
    Y = np.zeros((B, L), dtype=int)
    M = np.zeros((B, L))
    for i, (t, bar) in enumerate(zip(texts, bounds)):
        ids = [c2i[c] for c in t]
        n = len(ids) - 1
        X[i, :n] = ids[:-1]
        Y[i, :n] = ids[1:]
        M[i, :n] = 1.0
        M[i, :bar] = 0.1      # часть с вопросом учим слабее, главное - ответ
    return X, Y, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.txt")
    ap.add_argument("--out", default="chatbot.npz")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--emb", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    pairs = load_pairs(args.data)
    texts = ["\n" + q + "|" + a + "\n" for q, a in pairs]
    bounds = [len(q) + 1 for q, _ in pairs]   # позиция символа "|"
    chars = PAD + "".join(sorted(set("".join(texts))))
    c2i = {c: i for i, c in enumerate(chars)}
    print(f"Примеров: {len(texts)}, символов в словаре: {len(chars)}")

    model = GRULM(len(chars), args.emb, args.hidden)
    n_params = sum(v.size for v in model.p.values())
    print(f"Параметров: {n_params}")
    opt = Adam(model.p, lr=args.lr)
    rng = np.random.default_rng(0)
    nl = c2i["\n"]

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(texts))
        total, count = 0.0, 0
        for s in range(0, len(order), args.batch):
            idx = order[s:s + args.batch]
            X, Y, M = build_batch([texts[i] for i in idx], [bounds[i] for i in idx], c2i)
            loss, grads = model.loss_and_grads(X, Y, M)
            opt.update(model.p, grads)
            total += loss
            count += 1
        # постепенно уменьшаем скорость обучения
        opt.lr = args.lr * (0.5 * (1 + np.cos(np.pi * epoch / args.epochs)) * 0.9 + 0.1)

        if epoch % 10 == 0 or epoch == 1:
            q = normalize("как дела")
            out = model.generate([c2i[c] for c in "\n" + q + "|"], nl, temperature=0.3, rng=rng)
            print(f"эпоха {epoch:4d} | потери {total / count:.3f} | {time.time() - t0:5.0f} c"
                  f" | 'как дела' -> {''.join(chars[i] for i in out)}", flush=True)
            model.save(args.out, chars)

    model.save(args.out, chars)
    print("Готово, модель сохранена в", args.out)


if __name__ == "__main__":
    main()
