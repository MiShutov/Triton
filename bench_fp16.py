#!/usr/bin/env python3
"""
benchmark_matmul.py

Бенчмарк скорости матричных умножений в torch для float32 и float16.
Поддерживает CPU и CUDA (если доступен). Считает GFLOPS и относительную ошибку
результата fp16 по отношению к fp32.

Пример:
    python benchmark_matmul.py --device cuda --sizes 1024 1024 2048 --iters 50
"""

import argparse
import time
import math
import torch
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark matrix multiply FP32 vs FP16 (PyTorch)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device to run on: 'cuda' or 'cpu'")
    p.add_argument("--shapes", nargs="+", type=int, default=[1024, 1024, 1024],
                   help="Shape triple M N P (you can pass multiple of 3 values; script will use them as triples)")
    p.add_argument("--batch", type=int, default=1, help="Batch size (1 = plain 2D matmul)")
    p.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    p.add_argument("--iters", type=int, default=50, help="Number of measured iterations")
    p.add_argument("--check", action="store_true", help="Check fp16 result against fp32 (relative error)")
    p.add_argument("--no-cuda-sync", action="store_true", help="Don't call torch.cuda.synchronize() (unsafe for timing)")
    return p.parse_args()

def make_tensor(shape, dtype, device):
    # Нормальное распределение — подходит для тестов. Можно менять.
    return torch.randn(shape, dtype=dtype, device=device)

def measure(matmul_fn, a, b, warmup, iters, device, sync=True):
    # warmup
    for _ in range(warmup):
        r = matmul_fn(a, b)
        # ensure computation occurs on GPU before continuing
        if device.type == "cuda" and sync:
            torch.cuda.synchronize()

    # measured runs
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        r = matmul_fn(a, b)
        if device.type == "cuda" and sync:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = sorted(times)
    mean = sum(times) / len(times)
    median = times[len(times)//2]
    # also report best time
    best = times[0]
    return {"times": times, "mean": mean, "median": median, "best": best, "last_result": r}

def flops_for_matmul(m, n, p, batch=1):
    # стандартное число FLOPs для матмул: 2 * m * n * p (multiply-adds)
    return 2.0 * m * n * p * batch

def human_gflops(flops, seconds):
    return (flops / seconds) / 1e9

def main():
    args = parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA не доступна, переключаюсь на CPU.", file=sys.stderr)
        device = torch.device("cpu")

    if len(args.shapes) % 3 != 0:
        print("Ошибка: shapes должен состоять из трёх чисел (M N P) или нескольких триплетов.", file=sys.stderr)
        return

    triples = []
    s = args.shapes
    for i in range(0, len(s), 3):
        triples.append((s[i], s[i+1], s[i+2]))

    # настройки производительности
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        # Разрешить/запретить TF32 можно настроить здесь при необходимости:
        # torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Device: {device} | cuda available: {torch.cuda.is_available()} | batch: {args.batch}")
    print(f"Warmup: {args.warmup}, iters: {args.iters}")
    print("-" * 72)

    results = []

    for (M, N, P) in triples:
        shape_a = (args.batch, M, N) if args.batch > 1 else (M, N)
        shape_b = (args.batch, N, P) if args.batch > 1 else (N, P)
        flops = flops_for_matmul(M, N, P, batch=args.batch)

        row = {"M": M, "N": N, "P": P, "batch": args.batch}

        # 1) FP32
        a32 = make_tensor(shape_a, torch.float32, device)
        b32 = make_tensor(shape_b, torch.float32, device)

        def matmul_float32(x, y): return torch.matmul(x, y)

        res32 = measure(matmul_float32, a32, b32, args.warmup, args.iters, device,
                        sync=(device.type=="cuda" and not args.no_cuda_sync))
        gflops32 = human_gflops(flops, res32["best"])
        row.update({
            "time_fp32_best_s": res32["best"],
            "time_fp32_mean_s": res32["mean"],
            "gflops_fp32_best": gflops32
        })
        # free memory
        del a32, b32

        # 2) FP16
        # Note: на CPU float16 матмулы часто исполняются на float32 или очень медленно.
        try:
            a16 = make_tensor(shape_a, torch.float16, device)
            b16 = make_tensor(shape_b, torch.float16, device)
        except Exception as e:
            print(f"Не удалось создать float16 тензоры на {device}: {e}", file=sys.stderr)
            row.update({
                "time_fp16_best_s": None,
                "time_fp16_mean_s": None,
                "gflops_fp16_best": None,
                "fp16_ok": False
            })
            results.append(row)
            continue

        def matmul_float16(x, y): 
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                return torch.matmul(x, y)

        res16 = measure(matmul_float16, a16, b16, args.warmup, args.iters, device,
                        sync=(device.type=="cuda" and not args.no_cuda_sync))
        gflops16 = human_gflops(flops, res16["best"])
        row.update({
            "time_fp16_best_s": res16["best"],
            "time_fp16_mean_s": res16["mean"],
            "gflops_fp16_best": gflops16,
            "fp16_ok": True
        })

        # Проверка корректности: сравнить fp16 результат с fp32 (по относительной ошибке)
        if args.check:
            # Recreate reference fp32 on same device (to avoid host<->cuda copies)
            a32_ref = make_tensor(shape_a, torch.float32, device)
            b32_ref = make_tensor(shape_b, torch.float32, device)
            ref = torch.matmul(a32_ref, b32_ref)
            # cast fp16 result к fp32 для сравнения
            approx = res16["last_result"].to(dtype=torch.float32)
            # compute relative error: ||ref - approx||_fro / ||ref||_fro
            num = torch.norm(ref - approx).item()
            den = torch.norm(ref).item()
            rel_err = float("inf") if den == 0.0 else num / den
            row["fp16_rel_err"] = rel_err
            # small informative message
            if rel_err > 1e-2:
                row["fp16_rel_err_warning"] = f"Относительная ошибка {rel_err:.3e} (>1e-2)"
            else:
                row["fp16_rel_err_warning"] = None
            del a32_ref, b32_ref, ref, approx

        # cleanup
        del a16, b16
        results.append(row)

    # печать результатов в табличном виде
    print("{:>6} {:>6} {:>6} {:>5} | {:>10} {:>10} | {:>10} {:>10} | {:>10}".format(
        "M", "N", "P", "B", "fp32_best(s)", "fp32_mean(s)", "fp32_GF/s", "fp16_best(s)", "fp16_GF/s")
    )
    for r in results:
        print("{M:6d} {N:6d} {P:6d} {batch:5d} | {time_fp32_best_s:10.6f} {time_fp32_mean_s:10.6f} | {gflops_fp32_best:10.2f} {time_fp16_best_s:10.6f} {gflops_fp16_best:10.2f}".format(
            M=r["M"], N=r["N"], P=r["P"], batch=r["batch"],
            time_fp32_best_s=(r.get("time_fp32_best_s") or float("nan")),
            time_fp32_mean_s=(r.get("time_fp32_mean_s") or float("nan")),
            gflops_fp32_best=(r.get("gflops_fp32_best") or float("nan")),
            time_fp16_best_s=(r.get("time_fp16_best_s") or float("nan")),
            gflops_fp16_best=(r.get("gflops_fp16_best") or float("nan")),
        ))

    # если проверка включена, распечатать ошибки
    if args.check:
        print("\nПроверка точности fp16 vs fp32 (относительная ошибка):")
        for r in results:
            if "fp16_rel_err" in r:
                print(f"  {r['M']}x{r['N']} x {r['P']} batch={r['batch']}  rel_err={r['fp16_rel_err']:.3e}  {r.get('fp16_rel_err_warning','')}")
    print("\nГотово.")

if __name__ == "__main__":
    main()
