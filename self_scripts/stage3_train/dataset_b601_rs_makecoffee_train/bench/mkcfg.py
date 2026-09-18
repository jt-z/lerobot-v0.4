import json, sys
n = int(sys.argv[1]); bs = int(sys.argv[2]); steps = int(sys.argv[3])
cfg = {
  "dataset": {
    "repo_id": "hellozjt/b601_20260910_164106",
    "root": "/data/share/b601_20260910_164106",
    "revision": "main",
    "streaming": False
  },
  "policy": {"type": "act", "device": "cuda", "push_to_hub": False, "optimizer_lr": 8e-5},
  "output_dir": f"/tmp/act_bench/out_n{n}_bs{bs}",
  "job_name": f"bench_n{n}_bs{bs}",
  "steps": steps,
  "eval_freq": 0,
  "batch_size": bs,
  "num_workers": 6,
  "log_freq": 50,
  "save_freq": 10**9,
  "save_checkpoint": False,
  "seed": 1000,
}
json.dump(cfg, open(f"/tmp/act_bench/n{n}_bs{bs}.json","w"), indent=2)
print(f"wrote n{n}_bs{bs}.json  (steps={steps}, per-GPU bs={bs}, global={n*bs})")
