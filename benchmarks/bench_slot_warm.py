"""Bench: slot-warm cudaHostRegister amortization.

Preloads the same snapshot into the same slot 5 times. Load 0 pays the
one-time cudaHostRegister pin cost. Loads 1-4 should skip registration
and hit PCIe Gen4/5 throughput (15+ GB/s expected on H100 SXM).
"""

import time

from thaw_vllm.pool import EnginePool


def main():
    pool = EnginePool()
    pool.init_pool(
        "meta-llama/Meta-Llama-3-8B",
        pool_size=1,
        tensor_parallel_size=1,
        dtype="float16",
        enforce_eager=True,
    )
    pool.register("llama", "/tmp/llama3-8b.thaw")

    print("--- slot-warm amortization bench ---")
    for i in range(5):
        t0 = time.perf_counter()
        stats = pool.preload("llama", slot_id=0)
        elapsed = time.perf_counter() - t0
        backend = stats.get("backend", "?")
        thr = stats.get("throughput_gb_s", 0.0)
        print(
            f"load {i}: {elapsed:6.2f}s  "
            f"backend={backend:30s}  "
            f"{thr:5.1f} GB/s"
        )


if __name__ == "__main__":
    main()
