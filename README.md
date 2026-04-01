# Federated Continual Learning Benchmark

This branch provides a clean Federated Continual Learning (FCL) benchmark with ready-to-run baselines, reproducible data preparation, colored training progress, and two data-partition modes, Tuan and Lucaz.

# General Information

## 1, Algorithms

| Algorithm                                       | Venue                      | Paper                                                                                                                       | Original code base                                             | Newly added | Verified |
| ----------------------------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- | ----------: | -------: |
| **FedAvg**                                      | AISTATS 2017               | [Communication-Efficient Learning of Deep Networks from Decentralized Data] ([Proceedings of Machine Learning Research][1]) | —                                                              |             |        ⏳ pending |
| **GLFC** (Federated Class-Incremental Learning) | CVPR 2022                  | [Federated Class-Incremental Learning] ([CVF Open Access][2])                                                               | [conditionWang/FCIL] ([GitHub][3])                             |          ✔️ |        ⏳ pending |
| **LANDER**                                      | CVPR 2024                  | [Text-Enhanced Data-free Approach for FCIL] ([CVF Open Access][4])                                                          | [tmtuan1307/LANDER] ([GitHub][5])                              |          ✔️ |        ⏳ pending |
| **FedWeIT**                                     | ICML 2021                  | [Federated Continual Learning with Weighted Inter-client Transfer] ([Proceedings of Machine Learning Research][6])          | [wyjeong/FedWeIT] ([GitHub][7])                                |             |        ⏳ pending |
| **TARGET**                                      | ICCV 2023                  | [TARGET, Federated Class-Continual Learning via Exemplar-Free Distillation] ([CVF Open Access][8])                          | [zj-jayzhang/Federated-Class-Continual-Learning] ([GitHub][9]) |             |        ⏳ pending |
| **FedALA**                                      | AAAI 2023                  | [Adaptive Local Aggregation for Personalized FL] ([AAAI Open Access Journal][10])                                           | [TsingZ0/FedALA] ([GitHub][11])                                |             |        ⏳ pending |
| **FedAS**                                       | CVPR 2024                  | [Bridging Inconsistency in Personalized FL] ([CVF Open Access][12])                                                         | [xiyuanyang45/FedAS] ([GitHub][13])                            |             |        ⏳ pending |
| **FedDBE**                                      | NeurIPS 2023               | [Eliminating Domain Bias for FL in Representation Space] ([NeurIPS Proceedings][14])                                        | [TsingZ0/DBE] ([GitHub][15])                                   |             |        ⏳ pending |
| **FedL2P**                                      | NeurIPS 2023               | [Federated Learning to Personalize] ([NeurIPS Proceedings][16])                                                             | —                                                              |             |        ⏳ pending |

[1]: https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf "Communication-Efficient Learning of Deep Networks from ..."
[2]: https://openaccess.thecvf.com/content/CVPR2022/html/Dong_Federated_Class-Incremental_Learning_CVPR_2022_paper.html "[CVPR-2022] Federated Class-Incremental Learning"
[3]: https://github.com/conditionWang/FCIL "conditionWang/FCIL: This is the formal code ..."
[4]: https://openaccess.thecvf.com/content/CVPR2024/html/Tran_Text-Enhanced_Data-free_Approach_for_Federated_Class-Incremental_Learning_CVPR_2024_paper.html "CVPR 2024 Open Access Repository"
[5]: https://github.com/tmtuan1307/LANDER "tmtuan1307/LANDER: [CVPR-2024] Text-Enhanced Data- ..."
[6]: https://proceedings.mlr.press/v139/yoon21b/yoon21b.pdf "Federated Continual Learning with Weighted Inter-client ..."
[7]: https://github.com/wyjeong/FedWeIT "wyjeong/FedWeIT"
[8]: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_TARGET_Federated_Class-Continual_Learning_via_Exemplar-Free_Distillation_ICCV_2023_paper.pdf "TARGET: Federated Class-Continual Learning via Exemplar ..."
[9]: https://github.com/zj-jayzhang/Federated-Class-Continual-Learning "zj-jayzhang/Federated-Class-Continual-Learning: This is ..."
[10]: https://ojs.aaai.org/index.php/AAAI/article/view/26330 "FedALA: Adaptive Local Aggregation for Personalized ..."
[11]: https://github.com/TsingZ0/FedALA "TsingZ0/FedALA: AAAI 2023 accepted paper ..."
[12]: https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_FedAS_Bridging_Inconsistency_in_Personalized_Federated_Learning_CVPR_2024_paper.pdf "Bridging Inconsistency in Personalized Federated Learning"
[13]: https://github.com/xiyuanyang45/FedAS "Code Implementation and Informations about FedAS"
[14]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/2e0d3c6ad1a4d85bef3cfe63af58bc76-Abstract-Conference.html "Eliminating Domain Bias for Federated Learning in ..."
[15]: https://github.com/TsingZ0/DBE "TsingZ0/DBE: NeurIPS 2023 accepted paper, Eliminating ..."
[16]: https://proceedings.neurips.cc/paper_files/paper/2023/file/2fb57276bfbaf1b832d7bfcba36bb41c-Paper-Conference.pdf "FedL2P: Federated Learning to Personalize"

## 2, Dataset

| Dataset         | Modality   | #Classes | Image size (input)                              | Train / Test       | Per-class `.npy` folder       | Prep command                                                        | Default `cpt` | Default `#tasks` (`nt`) | Partition modes supported | Normalization (mean / std)                                  | Notes                                                                                    |
| --------------- | ---------- | -------: | ----------------------------------------------- | ------------------ | ----------------------------- | ------------------------------------------------------------------- | ------------: | ----------------------: | ------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **CIFAR-10**    | RGB images |       10 | 32×32                                           | 50,000 / 10,000    | `dataset/cifar10-classes/`    | `python dataset/cifar10_npy.py`                                     |             2 |                       5 | `current`, `mine`         | mean [0.4914, 0.4822, 0.4465], std [0.2023, 0.1994, 0.2010] | Label names available (airplane…truck). Uses all samples of selected classes by default. |
| **CIFAR-100**   | RGB images |      100 | 32×32                                           | 50,000 / 10,000    | `dataset/cifar100-classes/`   | `python dataset/cifar100_npy.py`                                    |             2 |                      50 | `current`, `mine`         | mean [0.4914, 0.4822, 0.4465], std [0.2023, 0.1994, 0.2010] | Class IDs 0–99. Common alternative is `cpt=10` (10 tasks).                               |
| **ImageNet-1K** | RGB images |    1,000 | variable (typically resized/cropped to 224×224) | ~1.28M / 50k (val) | `dataset/imagenet1k-classes/` | *(prepare `.npy` per class; follow your local script/instructions)* |             2 |                     500 | `current`, `mine`         | mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]       | Ensure storage available for per-class dumps. Transforms/resize set in your loaders.     |



# Experimental Setup

## 1, Installation

### 1.1, Environment

```bash
git clone https://github.com/chickbong221/FCL.git
cd FCL
git checkout Lucaz-FCL-PFLlib

# Python 3.9+ recommended
python -m venv .env
source .env/bin/activate

pip install -r requirements.txt
# optional, for pretty console tables and progress bars
pip install rich
# optional, for Weights & Biases logging
pip install wandb
```

### 1.2, Repo layout, important paths

```
FCL/
  system/
    main.py                      , entry point
    flcore/
      servers/                   , FedAvg, etc.
      clients/                   , client logic
      trainmodel/                , models
    utils/
      data_utils.py              , CURRENT partition (codebase split)
      data_utils_mine.py         , MINE partition (HeteroScope, all knobs in this file)
      partition_viz.py           , optional, partition visualization (if you added it)
  dataset/
    cifar10_npy.py               , builds per-class .npy
    cifar100_npy.py              , builds per-class .npy
    cifar10-classes/             , 0.npy … 9.npy
    cifar100-classes/            , 0.npy … 99.npy
    imagenet1k-classes/          , 0.npy … 999.npy (if you prepare it)
  hparams/
    FedAvg.json                  , example config, see below
```

## 2, Data preparation

### 2.1, CIFAR-10 and CIFAR-100

```bash
# CIFAR-10
cd dataset
python cifar10_npy.py        # creates dataset/cifar10-classes/{0..9}.npy

# CIFAR-100
python cifar100_npy.py       # creates dataset/cifar100-classes/{0..99}.npy
cd ..
```

### 2.2, ImageNet-1K (optional)

Place per-class `.npy` files under `dataset/imagenet1k-classes/0.npy ... 999.npy` following the same format as CIFAR. The loader expects HWC uint8 or CHW float that can be standardized.

## 3, Configuration

A minimal `hparams/FedAvg.json` example you can copy and edit:

```json
{
  "optimizer": "sgd",
  "device": "cuda",
  "device_id": "0",
  "dataset": "CIFAR100",
  "num_classes": 100,
  "model": "CNN",
  "model_str": "CNN",
  "batch_size": 128,
  "local_learning_rate": 0.01,
  "global_rounds": 100,
  "local_epochs": 5,
  "algorithm": "FedAvg",
  "join_ratio": 1.0,
  "random_join_ratio": false,
  "num_clients": 10,
  "eval_gap": 1,
  "out_folder": "out",
  "client_drop_rate": 0.0,
  "time_threthold": 10000
}
```

Notes, set `device="cuda"` only if you have a GPU, otherwise use `"cpu"`. The server keeps the global model on CPU by default, and each client moves to GPU only during its local `train()`.

## 4, Running one experiment

### 4.1, The quick run

```bash
# codebase/current partition, CIFAR-100, FedAvg
python system/main.py --cfp ./hparams/FedAvg.json --partition_options tuan
```

Useful optional flags:

* `--wandb True`, log to Weights & Biases
* `--log True`, print key metrics
* `--offlog True`, save CSVs under `out/<exp>/...`
* `--cpt <int>`, classes per task, for example `--cpt 10`
* `--nt <int>`, number of tasks (if not set, defaults by dataset or computed)

Mine-specific flags (new):

* `--alpha <float>`, Dirichlet α for per-class split across clients, smaller => more skew, for example `--alpha 0.3`
* `--task_disorder <0..1>`, task-order disorder, 0 => same as master order, 1 => fully permuted per client
* `--client_disorder "<csv or [list]>"`, per-client overrides, for example `"0,0.2,0.8,0.4"` (client 0 is forced to 0.0)
* `--seed <int>`, RNG seed for the partitioner

Colored progress and per-round tables appear automatically when `rich` is installed.

### 4.2, Expected outputs

* Console shows round headers, selected clients, per-client loss/acc/time/samples, and round summaries.
* Files saved under `out/<DATASET>_<ALGO>_<MODEL>_...`
* If you turned on offline logging, per-client and global CSVs are created in the run folder.

## 5, Data partition modes

You can pick between the **current** (codebase) split and **mine** (HeteroScope) split with one flag:

```bash
--partition_options tuan    , use utils/data_utils.py
--partition_options hetero       , use utils/data_utils_mine.py
```

### 5.1, Current partition, how it works

* There is a precomputed **class order** per client, for example arrays under `dataset/class_order/*.npy`.
* For a given `client` and `task`, the loader slices `cpt` consecutive class IDs from that client’s class order and builds the task dataset from per-class `.npy`.
* All images of the selected classes are used by default.

Example, CIFAR-10 with `cpt=2`, five clients, ten classes. Suppose a client’s class order is `[6, 5, 3, 2, 9, 7, 8, 1, 4, 0]`. Then:

* Task 0 → classes `[6, 5]`
* Task 1 → `[3, 2]`
* Task 2 → `[9, 7]`
* Task 3 → `[8, 1]`
* Task 4 → `[4, 0]`

Clients usually have different class orders, so Task 0 class pairs differ across clients.

### 5.2, Mine partition (HeteroScope), how it works

All knobs are now CLI arguments, no file edits needed:

```bash
# examples
--alpha 0.3
--task_disorder 0.6
--client_disorder "0,0.2,0.8,0.4,0.6"
--seed 42
```

What each does:

* `--task_disorder` controls how each client **permutes the global task pool**. 0 keeps all clients aligned with the master order, 1 produces a **random permutation per client**. Classes are not mixed across tasks.
* `--alpha` controls the **Dirichlet** split of each class’s samples across clients (standard FL skew). Smaller α gives heavier skew. If you later enable class-level sub-sampling, α shapes the label skew; if you use all samples per class, α mainly affects class-to-client ownership.
* `--client_disorder` optionally overrides the disorder per client. Client 0 is always 0.0 (master order).
* `--seed` fixes the global task pool and the per-client permutations.

Example, CIFAR-10 with `cpt=2`, `--task_disorder 1.0`:

* Build the global task pool `Y₀, Y₁, …, Y₄` from the master class order `[0..9]` as `[0,1]`, `[2,3]`, `[4,5]`, `[6,7]`, `[8,9]`.
* Each client draws a **random permutation** $\pi^c$ of `[0,1,2,3,4]`.
* Client c’s Task 0 uses the labels from $Y_{\pi^c(0)}$, so Task 0 labels differ across clients while each task remains a fixed class pair.

### 5.3, Switching between modes at runtime

No code changes are needed once your server’s `set_clients(...)` imports by `partition_options`. Just run:

```bash
# tuan
python system/main.py --cfp ./hparams/FedAvg.json --partition_options tuan

# hetero
python system/main.py --cfp ./hparams/FedAvg.json --partition_options hetero --alpha 0.3 --task_disorder 0.6 --seed 42
```

### 5.4, Quick sanity print (optional)

To verify the split before training, you can print Task 0 labels for the first few clients. Drop this right after `server.set_clients(...)` in your server code, or temporarily in `main.py` after server creation:

```python
print("\n[Partition sanity] Task 0 labels per client:")
for cid, u in enumerate(server.clients[:10]):
    print(f"  client {cid}: {u.task_dict.get(0)}")
```

## 6, Training progress UI

The server uses a colored progress bar and tables when `rich` is installed. You’ll see, per round, a header with the current task and selected clients, a per-client table with loss, accuracy, time and samples, and a summary panel. If `rich` is not installed, it falls back to simple prints.

Install once:

```bash
pip install rich
```

## 7, Tips and troubleshooting

* If **mine** looks identical to **current**, check:
  1, run with `--partition_options hetero`,
  2, ensure the server imports `utils/data_utils_mine.py` in `set_clients`,
  3, set `--task_disorder > 0` to misalign task orders across clients,
  4, optionally use `--client_disorder` to force per-client differences.
* CUDA out-of-memory at startup usually means the global model was moved to GPU at construction. Keep the global model on CPU, move only the selected client to GPU during `train()`.
* If evaluation raises a device mismatch, move the batch to the **model’s device** inside `test_metrics` and `train_metrics`:

  ```python
  dev = next(self.model.parameters()).device
  x = x.to(dev)
  y = y.to(dev)
  ```
* Check dataset paths. `dataset/cifar10-classes/0.npy` and `dataset/cifar100-classes/0.npy` should exist.

## 8, Reproducibility

* The codebase split is driven by fixed class-order arrays and will be deterministic given identical seeds in your training loop.
* The HeteroScope split now uses `--seed` to fix the global task pool and per-client permutations.

## 9, Minimal end-to-end recipe

```bash
# 1, environment
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
pip install rich

# 2, data
cd dataset
python cifar10_npy.py
python cifar100_npy.py
cd ..

# 3, config
# edit hparams/FedAvg.json with your dataset, num_clients, cpt, etc.

# 4a, run with codebase partition
python system/main.py --cfp ./hparams/FedAvg.json --partition_options tuan --log True --offlog True

# 4b, run with mine partition via CLI knobs
python system/main.py --cfp ./hparams/FedAvg.json \
  --partition_options hetero --log True --offlog True \
  --cpt 2 --nt 5 --alpha 0.3 --task_disorder 0.6 --seed 42
```

## 10, Suggested Experiments (what to run & why)

### Metrics used throughout

* **Average Accuracy**: up to the current task.
* **Average Forgetting** (already implemented as `metric_average_forgetting`) — core FCL metric.
* **Per-client accuracy distribution** (mean/±std/percentiles) — fairness/personalization.
* **Compute / time**: round time, total time, communication (number of rounds × participating clients).
* *(Optional)* **Gradient diagnostics**: when `--seval/--teval/pca_eval` are enabled.

### Tier 0 – Sanity & reproducibility (quick, ~minutes)

These ensure the pipeline, partitions, and logging behave as expected.

| ID   | Aim                          | Command (template)                                                                                                                              | Why                                                                                      |
| ---- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| S0.1 | Smoke-test CIFAR-10, 5 tasks | `python system/main.py --cfp ./hparams/FedAvg.json --partition_options tuan --cpt 2 --nt 5 --log True --offlog True`                         | Confirms training loop, saving, CSV logging work end-to-end.                             |
| S0.2 | Partition printout           | *(already in your server: prints task 0 classes per client)*                                                                                    | Verifies **current** partition class sets differ by client as expected.                  |
| S0.3 | Switch to **mine**           | `python system/main.py --cfp ./hparams/FedAvg.json --partition_options hetero --cpt 2 --nt 5 --task_disorder 0.6 --seed 123 --log True` | Confirms **mine** is actually used (Task-0 labels differ when `--task_disorder>0`). |
| S0.4 | Partition visualization      | *(your `partition_viz` helper)*                                                                                                                 | Produces heatmaps/CSV to visually confirm client × task × class layout.                  |

### Tier 1 – Core baselines matrix (CIFAR-10/100; **current** vs **mine**)

Establish reference numbers you can cite. Use **FedAvg**, **GLFC**, **LANDER** (or your available algorithms).

> **Setup**: in `hparams/FedAvg.json` (or per-algo JSON), set: `dataset`, `num_clients`, `global_rounds`, `local_epochs`, `batch_size`, `join_ratio`, etc.

#### CIFAR-10 (5 tasks, cpt=2)

| ID    | Algo   | Partition | Command                                                                                                   | Why                                                         |
| ----- | ------ | --------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| C10.1 | FedAvg | current   | `... --partition_options tuan --cpt 2 --nt 5`                                                          | Baseline for small-K datasets.                              |
| C10.2 | GLFC   | current   | `... --partition_options tuan --cpt 2 --nt 5`                                                          | FCL-aware baseline; establishes forgetting baseline.        |
| C10.3 | LANDER | current   | `... --partition_options tuan --cpt 2 --nt 5`                                                          | Text-enhanced FCIL; sanity on small dataset.                |
| C10.4 | FedAvg | mine      | `... --partition_options hetero --cpt 2 --nt 5 --task_disorder 0.6 --seed 123`                    | Tests sensitivity to **task-order disorder**.               |
| C10.5 | GLFC   | mine      | `... --partition_options hetero --cpt 2 --nt 5 --task_disorder 0.6 --alpha 0.3`                   | Same as above for advanced method; also tests label skew α. |
| C10.6 | LANDER | mine      | `... --partition_options hetero --cpt 2 --nt 5 --task_disorder 1.0 --client_disorder "0,0.2,0.8"` | Strong disorder and per-client overrides for robustness.    |

#### CIFAR-100 (50 tasks, cpt=2)

| ID     | Algo   | Partition | Command                                                                                                    | Why                                                          |
| ------ | ------ | --------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| C100.1 | FedAvg | current   | `... --partition_options tuan --cpt 2 --nt 50`                                                          | Standard FCIL stress test; many tasks exacerbate forgetting. |
| C100.2 | GLFC   | current   | `... --partition_options tuan --cpt 2 --nt 50`                                                          | Benchmarks FCL-aware method under long curricula.            |
| C100.3 | LANDER | current   | `... --partition_options tuan --cpt 2 --nt 50`                                                          | Compare with text-enhanced FCIL across many tasks.           |
| C100.4 | FedAvg | mine      | `... --partition_options hetero --cpt 2 --nt 50 --task_disorder 0.5 --seed 123`                    | Robustness to heterogeneous, asynchronous curricula.         |
| C100.5 | GLFC   | mine      | `... --partition_options hetero --cpt 2 --nt 50 --task_disorder 0.8 --alpha 0.3`                   | See if method benefits or degrades under stronger disorder.  |
| C100.6 | LANDER | mine      | `... --partition_options hetero --cpt 2 --nt 50 --task_disorder 1.0 --client_disorder "0,0.4,0.7"` | Same as above with client-specific permutations.             |

**Why Tier 1:** Gives you **headline numbers** (accuracy & forgetting) on standard datasets in both partition styles. These are the reference points for all ablations and claims.

### Tier 2 – Mine sweeps (CLI, only for **mine**)

Use CLI flags instead of editing Python.

* **Task-order disorder `--task_disorder`**: `{0.0, 0.2, 0.5, 1.0}`
  *Why:* quantify how asynchronous curricula across clients affect convergence & forgetting.
* **Per-client overrides `--client_disorder`**: e.g., `"0,0.2,0.8,0.4"`
  *Why:* evaluate heterogeneity where some clients deviate more than others.
* **Dirichlet label skew `--alpha`**: `{0.1, 0.3, 1.0, 10.0}`
  *Why:* test resilience to class-wise sample imbalance across clients.

**Template command (CIFAR-100):**

```bash
python system/main.py --cfp ./hparams/FedAvg.json \
  --partition_options hetero --cpt 2 --nt 50 \
  --task_disorder 0.5 --alpha 0.3 --seed 123 \
  --log True --offlog True
```

**What to collect:**
Per setting, log **Global Top-1**, **Average Forgetting**, and a **per-client accuracy histogram**.
**Why:** demonstrates your algorithm’s **robustness** to realistic heterogeneity (asynchrony, label skew).

### Tier 3 – Task & client scaling

Stress test scalability and participation dynamics.

| ID   | Factor             | Values                                               | Command example                   | Why                                                                          |
| ---- | ------------------ | ---------------------------------------------------- | --------------------------------- | ---------------------------------------------------------------------------- |
| SC.1 | `num_clients`      | 10 → 50 (same total data; split across more clients) | set in JSON, re-run C100.1/C100.4 | Tests communication & aggregation stability; variance across clients.        |
| SC.2 | `join_ratio`       | 1.0 → 0.2                                            | set in JSON, keep others          | Partial participation; real FL.                                              |
| SC.3 | `client_drop_rate` | 0.0 → 0.3                                            | set in JSON                       | Robustness to stragglers/dropouts.                                           |
| SC.4 | `local_epochs`     | 1, 5, 10                                             | set in JSON                       | Compute/accuracy trade-off; too many local steps can hurt stability (drift). |

**Why Tier 3:** Validates method **at scale** and under **practical constraints** (partial participation, drops).

### Tier 4 – Class-per-task & curriculum granularity

Change **cpt** (classes per task). Keep total classes fixed.

| ID  | Dataset   | cpt | nt | Command            | Why                                    |
| --- | --------- | --: | -: | ------------------ | -------------------------------------- |
| G.1 | CIFAR-100 |   2 | 50 | `--cpt 2 --nt 50`  | Long sequences (hardest forgetting).   |
| G.2 | CIFAR-100 |   5 | 20 | `--cpt 5 --nt 20`  | Coarser tasks, less severe forgetting. |
| G.3 | CIFAR-100 |  10 | 10 | `--cpt 10 --nt 10` | Typical in literature.                 |

**Why Tier 4:** Shows sensitivity to **task granularity** — important when comparing to other papers.

### Tier 5 – Diagnostics (optional but insightful)

Turn on your built-ins to interpret learning dynamics.

| Switch             | Add to command | Why                                                             |
| ------------------ | -------------- | --------------------------------------------------------------- |
| Spatio-grad eval   | `--seval`      | Logs angles/distances between clients/rounds; interpretational. |
| Temporal-grad eval | `--teval`      | Temporal dynamics across rounds.                                |
| PCA eval           | `--pca_eval`   | Stores models to inspect representation evolution offline.      |

## Recommended reporting sheet (what to log per run)

| Field                                | Description                                                                                                                     |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset / Partition**              | CIFAR-10/100 / current or mine (with `--task_disorder`, `--alpha`, `--seed`, optional `--client_disorder`). |
| **Algo**                             | FedAvg, GLFC, LANDER, …                                                                                                         |
| **Clients / join_ratio / drop_rate** | e.g., 10 / 1.0 / 0.0                                                                                                            |
| **cpt / nt**                         | e.g., 2 / 50                                                                                                                    |
| **local_epochs / batch_size / lr**   | e.g., 5 / 128 / 0.01                                                                                                            |
| **Global Top-1 @ last**              | Final accuracy.                                                                                                                 |
| **Avg Forgetting**                   | At task end (from your `eval_task`).                                                                                            |
| **Per-client mean±std**              | Distribution and fairness.                                                                                                      |
| **Round time (median)**              | Efficiency.                                                                                                                     |
| **Notes**                            | Any anomalies or hardware notes.                                                                                                |

## Why these experiments (summary)

* **Core baselines (Tier 1)** establish **reproducible references** on both partitions. Everything else compares against them.
* **Mine sweeps (Tier 2)** validate **robustness** to real heterogeneity: asynchronous curricula via task-order disorder and client-specific permutations, and class-wise skew via α.
* **Scaling (Tier 3)** tests **practical FL constraints** (partial participation, drops), ensuring methods are deployable.
* **Granularity (Tier 4)** shows sensitivity to **task design** (cpt/nt) so results aren’t over-fit to a single configuration.
* **Diagnostics (Tier 5)** help **explain** wins/losses (not just report numbers), which strengthens your paper or report.

## Copy-paste command snippets

**CIFAR-100, FedAvg, current:**

```bash
python system/main.py --cfp ./hparams/FedAvg.json \
  --partition_options tuan --cpt 2 --nt 50 --log True --offlog True
```

**CIFAR-100, GLFC, mine (task disorder 1.0, α=0.3):**

```bash
python system/main.py --cfp ./hparams/FedAvg.json \
  --partition_options hetero --cpt 2 --nt 50 \
  --task_disorder 1.0 --alpha 0.3 --seed 123 \
  --log True --offlog True
```

**CIFAR-10, LANDER, mine (disorder sweep):**

```bash
# disorder ∈ {0.0, 0.2, 0.5, 1.0}
python system/main.py --cfp ./hparams/FedAvg.json \
  --partition_options hetero --cpt 2 --nt 5 \
  --task_disorder 0.5 --seed 123 --log True
```

## 11, Ready-to-paste experiments — CIFAR-10 (no GLFC, no LANDER)

All commands assume your hparams JSON for each algorithm lives under `./hparams/cifar10/<Algo>.json`. If a file is missing, just copy `FedAvg.json` and change `"algorithm"` inside to the target name. CIFAR-10 uses 10 classes, so common settings are `--cpt 2 --nt 5` or `--cpt 5 --nt 2`.

```bash
# FedAvg
python system/main.py --cfp ./hparams/cifar10/FedAvg.json \
  --partition_options hetero --cpt 2 --nt 5 \
  --task_disorder 0.6 --alpha 0.3 --seed 123 \
  --log True --offlog True

# FedWeIT
python system/main.py --cfp ./hparams/cifar10/FedWeIT.json \
  --partition_options hetero --cpt 2 --nt 5 \
  --task_disorder 0.6 --alpha 0.3 --seed 123 \
  --log True --offlog True

# TARGET
python system/main.py --cfp ./hparams/cifar10/TARGET.json \
  --partition_options hetero --cpt 2 --nt 5 \
  --task_disorder 0.6 --alpha 0.3 --seed 123 \
  --log True --offlog True

# FedALA
python system/main.py --cfp ./hparams/cifar10/FedALA.json \
  --partition_options hetero --cpt 2 --nt 5 \
  --task_disorder 0.6 --alpha 0.3 --seed 123 \
  --log True --offlog True

# FedAS
python system/main.py --cfp ./hparams/cifar10/FedAS.json \
  --partition_options hetero --cpt 2 --nt 5 \
  --task_disorder 0.6 --alpha 0.3 --seed 123 \
  --log True --offlog True

# FedDBE
python system.main.py --cfp ./hparams/cifar10/FedDBE.json \
  --partition_options hetero --cpt 2 --nt 5 \
  --task_disorder 0.6 --alpha 0.3 --seed 123 \
  --log True --offlog True

# FedL2P
python system/main.py --cfp ./hparams/cifar10/FedL2P.json \
  --partition_options hetero --cpt 2 --nt 5 \
  --task_disorder 0.6 --alpha 0.3 --seed 123 \
  --log True --offlog True

# FedSTGM
python system/main.py --cfp ./hparams/cifar10/FedSTGM.json \
  --partition_options hetero --cpt 2 --nt 5 \
  --task_disorder 0.6 --alpha 0.3 --seed 123 \
  --log True --offlog True
```