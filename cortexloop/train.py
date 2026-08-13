"""Train a tiny INT8 CNN on synthetic warehouse views and export CortexLoop weights."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

from .render import render_centered_class

N_CLASSES = 8
IN = 32


def to_float(buf: bytes) -> np.ndarray:
    # Match native conv1: (u8 - 128) / 128
    return (np.frombuffer(buf, dtype=np.uint8).reshape(IN, IN, 3).astype(np.float32) - 128.0) / 128.0


def im2col(x, kh=3, kw=3, stride=2, pad=1):
    n, h, w, c = x.shape
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w + 2 * pad - kw) // stride + 1
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    cols = np.zeros((n, oh, ow, kh * kw * c), dtype=np.float32)
    idx = 0
    for ky in range(kh):
        for kx in range(kw):
            cols[:, :, :, idx : idx + c] = xp[:, ky : ky + oh * stride : stride, kx : kx + ow * stride : stride, :]
            idx += c
    return cols.reshape(n * oh * ow, kh * kw * c), oh, ow


def col2im(dcol, n, h, w, c, kh=3, kw=3, stride=2, pad=1):
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w + 2 * pad - kw) // stride + 1
    dcol = dcol.reshape(n, oh, ow, kh, kw, c)
    xp = np.zeros((n, h + 2 * pad, w + 2 * pad, c), dtype=np.float32)
    for ky in range(kh):
        for kx in range(kw):
            xp[:, ky : ky + oh * stride : stride, kx : kx + ow * stride : stride, :] += dcol[:, :, :, ky, kx, :]
    if pad:
        return xp[:, pad:-pad, pad:-pad, :]
    return xp


def conv_fw(x, w, b, stride=2, pad=1):
    n = x.shape[0]
    nout = w.shape[0]
    col, oh, ow = im2col(x, w.shape[1], w.shape[2], stride, pad)
    y = col @ w.reshape(nout, -1).T + b
    return y.reshape(n, oh, ow, nout), col


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def make_dataset(n: int, seed: int = 0):
    rng = __import__("random").Random(seed)
    xs, ys = [], []
    for i in range(n):
        cid = i % N_CLASSES
        xs.append(to_float(render_centered_class(cid, IN, IN, rng=rng)))
        ys.append(cid)
    return np.stack(xs), np.array(ys, dtype=np.int64)


def train(epochs: int = 20, n: int = 3200, seed: int = 1):
    x, y = make_dataset(n, seed)
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, 0.08, (8, 3, 3, 3)).astype(np.float32)
    b1 = np.zeros(8, np.float32)
    w2 = rng.normal(0, 0.08, (16, 3, 3, 8)).astype(np.float32)
    b2 = np.zeros(16, np.float32)
    wf = rng.normal(0, 0.08, (N_CLASSES, 16)).astype(np.float32)
    bf = np.zeros(N_CLASSES, np.float32)
    lr = 0.08
    bs = 64
    for ep in range(epochs):
        idx = rng.permutation(len(x))
        total = correct = seen = 0
        for s in range(0, len(x), bs):
            batch = idx[s : s + bs]
            xb, yb = x[batch], y[batch]
            nB, h, w, c = xb.shape
            h1, col1 = conv_fw(xb, w1, b1)
            a1 = np.maximum(h1, 0)
            h2, col2 = conv_fw(a1, w2, b2)
            a2 = np.maximum(h2, 0)
            g = a2.mean(axis=(1, 2))
            logits = g @ wf.T + bf
            p = softmax(logits)
            oh = np.zeros_like(p)
            oh[np.arange(len(yb)), yb] = 1.0
            loss = -np.mean(np.log(np.clip(p[np.arange(len(yb)), yb], 1e-7, 1)))
            dlogits = (p - oh) / len(yb)
            dwf = dlogits.T @ g
            dbf = dlogits.sum(0)
            dg = dlogits @ wf
            n2, hh, ww, c2 = a2.shape
            da2 = np.broadcast_to((dg / (hh * ww))[:, None, None, :], a2.shape).copy()
            dh2 = da2 * (h2 > 0)
            dy2 = dh2.reshape(-1, 16)
            dw2 = (dy2.T @ col2).reshape(w2.shape)
            db2 = dy2.sum(0)
            dcol2 = dy2 @ w2.reshape(16, -1)
            da1 = col2im(dcol2, nB, a1.shape[1], a1.shape[2], a1.shape[3])
            dh1 = da1 * (h1 > 0)
            dy1 = dh1.reshape(-1, 8)
            dw1 = (dy1.T @ col1).reshape(w1.shape)
            db1 = dy1.sum(0)
            wf -= lr * dwf
            bf -= lr * dbf
            w2 -= lr * dw2
            b2 -= lr * db2
            w1 -= lr * dw1
            b1 -= lr * db1
            total += float(loss) * len(yb)
            correct += int((p.argmax(1) == yb).sum())
            seen += len(yb)
        acc = correct / max(1, seen)
        print(f"epoch {ep+1:02d}  loss={total/seen:.4f}  acc={acc:.3f}")
        if acc >= 0.98:
            break
        lr *= 0.94
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "wf": wf, "bf": bf}


def quant_per_out(w: np.ndarray):
    out = w.reshape(w.shape[0], -1)
    scales = np.empty((out.shape[0],), np.float32)
    q = np.empty_like(out, dtype=np.int8)
    for i in range(out.shape[0]):
        amax = float(np.max(np.abs(out[i])))
        scale = (amax / 127.0) if amax > 0 else 1.0
        scales[i] = scale
        q[i] = np.clip(np.round(out[i] / scale), -127, 127).astype(np.int8)
    return q.reshape(w.shape), scales


def export(weights: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    w1q, s1 = quant_per_out(weights["w1"])
    w2q, s2 = quant_per_out(weights["w2"])
    wfq, sf = quant_per_out(weights["wf"])
    with path.open("wb") as f:
        f.write(b"CL01")
        f.write(struct.pack("<iiii", N_CLASSES, IN, IN, 3))
        f.write(struct.pack("<iiiiii", 8, 3, 3, 3, 2, 1))
        f.write(w1q.reshape(8, -1).astype(np.int8).tobytes())
        f.write(s1.tobytes())
        f.write(weights["b1"].astype(np.float32).tobytes())
        f.write(struct.pack("<iiiiii", 16, 8, 3, 3, 2, 1))
        f.write(w2q.reshape(16, -1).astype(np.int8).tobytes())
        f.write(s2.tobytes())
        f.write(weights["b2"].astype(np.float32).tobytes())
        f.write(struct.pack("<ii", N_CLASSES, 16))
        f.write(wfq.reshape(8, 16).astype(np.int8).tobytes())
        f.write(sf.tobytes())
        f.write(weights["bf"].astype(np.float32).tobytes())
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--samples", type=int, default=3200)
    p.add_argument("--out", default="models/warehouse_int8.bin")
    args = p.parse_args()
    w = train(args.epochs, args.samples)
    export(w, Path(args.out))


if __name__ == "__main__":
    main()
