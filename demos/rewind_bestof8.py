import os, json, itertools
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
from vllm import LLM, SamplingParams
import thaw_vllm
from thaw_vllm import rewind
from thaw_vllm.rewind import _first_divergence, summarize_rollout

MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT = "/workspace/rollouts"
os.system(f"rm -rf {OUT}")

llm = LLM(model=MODEL, enforce_eager=True, gpu_memory_utilization=0.85,
          max_model_len=2048, dtype="bfloat16")
tok = llm.get_tokenizer()
problem = ("A train leaves city A heading east at 60 mph. Two hours later, a second "
           "train leaves city A on the same track heading east at 90 mph. How many "
           "hours after the second train departs will it catch up to the first train? "
           "Reason step by step, then give the final answer.")
prompt = tok.apply_chat_template(
    [{"role": "user", "content": problem}], tokenize=False, add_generation_prompt=True)

# best-of-8 at a moderate temp: chains share the setup, then fork at a real step
sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=400, seed=7, logprobs=5)
labels = [f"rollout-{i}" for i in range(8)]
paths = thaw_vllm.capture_rollouts(llm, prompt, sp, out_dir=OUT, n=8, logprobs=5, labels=labels)
print("WROTE", len(paths), "rollouts from", MODEL)

dirs = [os.path.dirname(p) for p in paths]
S = {d: summarize_rollout(d) for d in dirs}
# deepest-pivot pair = the most interesting diff (longest shared reasoning prefix)
pairs = sorted(((_first_divergence(S[a]["tokens"], S[b]["tokens"]), a, b)
                for a, b in itertools.combinations(dirs, 2)), reverse=True)
deep, da, db = pairs[0]

print("\n==== pivot (all 8) ====")
print(rewind.pivot_rollouts(OUT))
print("\n==== deepest-pivot diff: %s vs %s (split at token %d) ====" %
      (os.path.basename(da), os.path.basename(db), deep))
print(rewind.diff_rollouts(da, db))
print("\nDEEP_PAIR %s %s %d" % (os.path.basename(da), os.path.basename(db), deep))
print("VALIDATION_OK")
