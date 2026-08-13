from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="cortexloop", description="CortexLoop Physical AI on Arm")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="launch the live warehouse dashboard")
    b = sub.add_parser("bench", help="run Arm kernel microbenchmarks")
    b.add_argument("--m", type=int, default=256)
    b.add_argument("--n", type=int, default=256)
    b.add_argument("--k", type=int, default=256)
    t = sub.add_parser("train", help="train and export the INT8 perception CNN")
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--samples", type=int, default=2400)
    t.add_argument("--out", default="models/warehouse_int8.bin")
    args = p.parse_args(argv)

    if args.cmd == "demo":
        from .server import main as demo

        demo()
        return 0
    if args.cmd == "bench":
        from .native import bench_json, cpu_features, load

        load()
        print(cpu_features())
        print(bench_json(args.m, args.n, args.k))
        return 0
    if args.cmd == "train":
        from .train import export, train

        from pathlib import Path

        w = train(args.epochs, args.samples)
        export(w, Path(args.out))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
