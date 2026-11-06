"""Маленькая символьная GRU-нейросеть на чистом NumPy.

Модель читает текст по одному символу и предсказывает следующий символ.
Диалог оформлен как:  \n вопрос | ответ \n
Чтобы получить ответ, мы подаём "\n вопрос |" и даём сети дописывать
символы, пока она не поставит "\n".
"""
import re
import numpy as np

PAD = "\x00"  # символ-заполнитель для выравнивания длины примеров


def normalize(text):
    """Приводим вопрос к единому виду: строчные буквы, без знаков препинания."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class GRULM:
    def __init__(self, vocab_size, emb=32, hidden=128, seed=0):
        rng = np.random.default_rng(seed)
        V, E, H = vocab_size, emb, hidden
        self.V, self.E, self.H = V, E, H

        def init(shape, fan_in):
            return rng.normal(0.0, 1.0 / np.sqrt(fan_in), shape)

        self.p = {
            "Emb": rng.normal(0.0, 0.1, (V, E)),   # эмбеддинги символов
            "Wx": init((E, 3 * H), E),             # вход -> гейты z, r и кандидат n
            "Uzr": init((H, 2 * H), H),            # состояние -> гейты z, r
            "Un": init((H, H), H),                 # состояние -> кандидат n
            "b": np.zeros(3 * H),
            "Wy": init((H, V), H),                 # состояние -> логиты символов
            "by": np.zeros(V),
        }

    # ------------------------------------------------------------------
    def step(self, ids, h):
        """Один шаг GRU (для батча). Возвращает новое состояние и логиты."""
        p, H = self.p, self.H
        x = p["Emb"][ids]
        a = x @ p["Wx"] + p["b"]
        zr = h @ p["Uzr"]
        z = sigmoid(a[:, :H] + zr[:, :H])
        r = sigmoid(a[:, H:2 * H] + zr[:, H:])
        n = np.tanh(a[:, 2 * H:] + (r * h) @ p["Un"])
        h_new = (1 - z) * n + z * h
        return h_new, h_new @ p["Wy"] + p["by"]

    # ------------------------------------------------------------------
    def loss_and_grads(self, X, Y, M):
        """Прямой проход + обратное распространение ошибки во времени (BPTT).

        X, Y : (B, T) индексы входных и целевых символов
        M    : (B, T) вес каждой позиции в функции потерь (0 для заполнителей)
        """
        p, H = self.p, self.H
        B, T = X.shape
        h = np.zeros((B, H))
        cache = []
        loss = 0.0
        rows = np.arange(B)

        for t in range(T):
            x = p["Emb"][X[:, t]]
            a = x @ p["Wx"] + p["b"]
            zr = h @ p["Uzr"]
            z = sigmoid(a[:, :H] + zr[:, :H])
            r = sigmoid(a[:, H:2 * H] + zr[:, H:])
            rh = r * h
            n = np.tanh(a[:, 2 * H:] + rh @ p["Un"])
            h_new = (1 - z) * n + z * h
            logits = h_new @ p["Wy"] + p["by"]
            logits = logits - logits.max(1, keepdims=True)
            e = np.exp(logits)
            pr = e / e.sum(1, keepdims=True)
            loss -= (np.log(pr[rows, Y[:, t]] + 1e-12) * M[:, t]).sum()
            cache.append((x, h, z, r, rh, n, h_new, pr))
            h = h_new

        norm = M.sum()
        loss /= norm

        g = {k: np.zeros_like(v) for k, v in p.items()}
        dh_next = np.zeros((B, H))
        for t in reversed(range(T)):
            x, h_prev, z, r, rh, n, h_new, pr = cache[t]
            dlog = pr.copy()
            dlog[rows, Y[:, t]] -= 1.0
            dlog *= M[:, t:t + 1] / norm

            g["Wy"] += h_new.T @ dlog
            g["by"] += dlog.sum(0)
            dh = dlog @ p["Wy"].T + dh_next

            dn = dh * (1 - z)
            dz = dh * (h_prev - n)
            dh_prev = dh * z

            dn_pre = dn * (1 - n * n)
            g["Un"] += rh.T @ dn_pre
            drh = dn_pre @ p["Un"].T
            dr = drh * h_prev
            dh_prev += drh * r

            dz_pre = dz * z * (1 - z)
            dr_pre = dr * r * (1 - r)
            dzr = np.concatenate([dz_pre, dr_pre], axis=1)
            g["Uzr"] += h_prev.T @ dzr
            dh_prev += dzr @ p["Uzr"].T

            da = np.concatenate([dz_pre, dr_pre, dn_pre], axis=1)
            g["Wx"] += x.T @ da
            g["b"] += da.sum(0)
            np.add.at(g["Emb"], X[:, t], da @ p["Wx"].T)

            dh_next = dh_prev

        return loss, g

    # ------------------------------------------------------------------
    def generate(self, prompt_ids, stop_id, max_len=160, temperature=0.4, rng=None):
        """Дописывает текст после prompt_ids, пока не встретит stop_id."""
        rng = rng or np.random.default_rng()
        h = np.zeros((1, self.H))
        logits = None
        for i in prompt_ids:
            h, logits = self.step(np.array([i]), h)
        out = []
        for _ in range(max_len):
            l = logits[0] / max(temperature, 1e-3)
            l = l - l.max()
            pr = np.exp(l)
            pr /= pr.sum()
            i = int(rng.choice(len(pr), p=pr))
            if i == stop_id:
                break
            out.append(i)
            h, logits = self.step(np.array([i]), h)
        return out

    # ------------------------------------------------------------------
    def save(self, path, chars):
        np.savez(path, chars=np.array(chars), E=self.E, H=self.H, **self.p)

    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=False)
        chars = str(d["chars"])
        m = GRULM(len(chars), int(d["E"]), int(d["H"]))
        for k in m.p:
            m.p[k] = d[k].copy()
        return m, chars


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def update(self, params, grads, clip=5.0):
        # ограничиваем норму градиента, чтобы обучение не "взрывалось"
        total = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
        scale = min(1.0, clip / (total + 1e-9))
        self.t += 1
        for k in params:
            g = grads[k] * scale
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def gradient_check():
    """Сверяем ручные градиенты с численными — так ловят ошибки в backprop."""
    rng = np.random.default_rng(1)
    m = GRULM(11, emb=5, hidden=7, seed=2)
    X = rng.integers(0, 11, (3, 6))
    Y = rng.integers(0, 11, (3, 6))
    M = (rng.random((3, 6)) > 0.2).astype(float)
    _, g = m.loss_and_grads(X, Y, M)
    worst = 0.0
    for k, w in m.p.items():
        for _ in range(6):
            idx = tuple(rng.integers(0, s) for s in w.shape)
            old = w[idx]
            w[idx] = old + 1e-5
            lp, _ = m.loss_and_grads(X, Y, M)
            w[idx] = old - 1e-5
            lm, _ = m.loss_and_grads(X, Y, M)
            w[idx] = old
            num = (lp - lm) / 2e-5
            err = abs(num - g[k][idx]) / max(1e-8, abs(num) + abs(g[k][idx]))
            worst = max(worst, err)
    return worst


if __name__ == "__main__":
    err = gradient_check()
    print(f"Макс. относительная ошибка градиентов: {err:.2e}",
          "- OK" if err < 1e-4 else "- ОШИБКА")
