"""Разговор с обученной моделью.  Запуск:  python3 chat.py"""
import argparse
import numpy as np
from model import GRULM, normalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="chatbot.npz")
    ap.add_argument("--temp", type=float, default=0.3,
                    help="0.1 = осторожные ответы, 1.0 = более случайные")
    args = ap.parse_args()

    model, chars = GRULM.load(args.model)
    c2i = {c: i for i, c in enumerate(chars)}
    rng = np.random.default_rng()
    print("Нейрон: Привет! Напиши что-нибудь (пустая строка или 'выход' - закончить).")
    while True:
        try:
            text = input("Ты: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or normalize(text) in ("выход", "exit", "quit"):
            break
        q = "".join(c for c in normalize(text) if c in c2i)   # неизвестные символы выбрасываем
        ids = [c2i[c] for c in "\n" + q + "|"]
        out = model.generate(ids, c2i["\n"], temperature=args.temp, rng=rng)
        print("Нейрон:", "".join(chars[i] for i in out))


if __name__ == "__main__":
    main()
