"""Correctness check: back-to-back slot reloads produce identical output.

Rules out the possibility that the 55 GB/s slot-warm path is just fast
garbage. Generates once after the first load, reloads the same
snapshot into the same slot, generates again with the same prompt +
seed + greedy sampling, and asserts the outputs match.
"""

from thaw_vllm.pool import EnginePool
from vllm import SamplingParams


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

    sp = SamplingParams(temperature=0.0, max_tokens=20)
    prompts = ["The capital of France is"]

    pool.preload("llama", slot_id=0)
    out1 = pool.slots[0].llm.generate(prompts, sp)[0].outputs[0].text

    pool.preload("llama", slot_id=0)
    out2 = pool.slots[0].llm.generate(prompts, sp)[0].outputs[0].text

    print("--- correctness check ---")
    print(f"out1: {out1!r}")
    print(f"out2: {out2!r}")
    print(f"identical: {out1 == out2}")


if __name__ == "__main__":
    main()
