# Waste-Aware Calibrated Early Exit (WACEE)

**Problem.** A conventional waste classifier sends every image through the full network, even when an easy class can be recognized by an earlier layer. This spends the same computation on easy and difficult images.

**Solution.** Waste-Aware Calibrated Early Exit (WACEE) adds two intermediate exits to ResNet-18, calibrates their confidence, estimates predicted-class difficulty on held-out data, and applies stricter exit thresholds to harder waste classes.

**Outcome.** Across three seeds, WACEE sends 92.58% of locked-test images through the first exit and reduces theoretical average FLOPs by 42.75% relative to full ResNet-18, while mean Macro F1 remains within 0.27 percentage points. The experiment therefore validates calibrated early exit as a feasible same-backbone computation-reduction strategy for waste classification.

The default run is a smoke test. It checks the full pipeline, but it is not final research evidence.

# Project structure

```text
code/
├── Waste_Aware_Early_Exit_Experiment.ipynb  # Run this file
├── configs/                                  # Smoke and full settings
├── scripts/                                  # Environment and power helpers
├── waste_early_exit/                         # Reusable experiment code
├── tests/                                    # Unit and integration tests
├── garbage_classification/                   # Locally downloaded image dataset
└── artifacts/                                # Generated outputs
```

The notebook keeps the story readable. Training, calibration, routing, metrics, profiling, and plotting live in small Python modules.

# Installation, configuration, and usage

Follow these steps in order. Restoring the published v1.0 artifacts in Step 3 is optional; skip it when you want to train everything locally.

## 1. Download the dataset

This experiment uses Mostafa Abla's [Garbage Classification (12 classes) dataset on Kaggle](https://www.kaggle.com/datasets/mostafaabla/garbage-classification/data). Dataset images are intentionally excluded from Git; the repository tracks only `garbage_classification/.gitkeep` so that the destination directory exists after cloning.

Download the archive in a browser from the Kaggle page, or use the official [Kaggle CLI](https://github.com/Kaggle/kaggle-cli):

```bash
python -m pip install kaggle
kaggle datasets download mostafaabla/garbage-classification --path . --unzip
```

The CLI requires Kaggle authentication; follow the official [authentication instructions](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication). After extraction, ensure that the 12 class directories sit directly under the repository's `garbage_classification/` directory:

```text
garbage_classification/
├── battery/
├── biological/
├── brown-glass/
├── cardboard/
├── clothes/
├── green-glass/
├── metal/
├── paper/
├── plastic/
├── shoes/
├── trash/
└── white-glass/
```

From the repository root, this command should report `15515` for the dataset version used by this project:

```bash
find garbage_classification -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l
```

The dataset is maintained outside this repository. Review the current license and usage terms on its Kaggle page before redistributing it; the repository's software terms do not automatically replace the dataset's terms.

## 2. Create the environment

The helper script creates the Conda environment, registers the Jupyter kernel, and runs a real MPS forward/backward check.

```bash
chmod +x scripts/create_env.sh
./scripts/create_env.sh
conda activate waste-early-exit
```

The equivalent manual commands are:

```bash
conda env create -f environment.yml
conda activate waste-early-exit
python -m ipykernel install --user --name waste-early-exit --display-name "Python (waste-early-exit)"
```

Check Apple MPS directly:

```bash
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_built()); print(torch.backends.mps.is_available())"
```

Expected output on the target Mac ends with `True` and `True`. Run the check in a native macOS terminal. A sandboxed process can hide MPS even when the Mac supports it.

## 3. Optionally restore the v1.0 checkpoints and results

Large checkpoints and generated results are distributed through GitHub Releases rather than the Git repository:

| Item | Value |
|---|---|
| Release title | `Checkpoint and result v1.0` |
| Git tag | `v1.0` |
| Archive | `artifacts-v1.0.zip` |
| Release page | [Checkpoint and result v1.0](https://github.com/blue1018/code/releases/tag/v1.0) |
| Archive layout | One top-level `artifacts/` directory |

The link becomes available after the Release and archive have been published. The archive does not contain the Kaggle dataset.

Download it in a browser from the Release page, or use either command:

```bash
curl -L -o artifacts-v1.0.zip \
  https://github.com/blue1018/code/releases/download/v1.0/artifacts-v1.0.zip

# Alternative for users with the GitHub CLI installed:
gh release download v1.0 \
  --repo blue1018/code \
  --pattern 'artifacts-v1.0.zip'
```

Extract the archive from the repository root:

```bash
unzip artifacts-v1.0.zip -d .
```

Use a clean clone or preserve an existing local `artifacts/` directory before extraction so that your own results are not overwritten. A correctly packaged archive restores `artifacts/seed_42/`, `artifacts/seed_123/`, `artifacts/seed_2026/`, and `artifacts/aggregate/`.

The package contains the full-run outputs, checkpoints, cached logits, frozen manifests and splits, tables, figures, and logs present when v1.0 was created. Results can be inspected immediately under `artifacts/aggregate/results/`; plots are under directories such as `artifacts/seed_42/figures/`.

## 4. Configure the notebook

The first parameter cell in `Waste_Aware_Early_Exit_Experiment.ipynb` contains:

```python
RUN_MODE = "smoke"
FORCE_REBUILD = False
```

Choose values according to the task:

| Goal | `RUN_MODE` | `FORCE_REBUILD` | Behavior |
|---|---|---|---|
| First pipeline check | `"smoke"` | `False` | Runs the small end-to-end validation and reuses valid caches |
| Full experiment | `"full"` | `False` | Runs seeds 42, 123, and 2026 and trains only missing or invalid stages |
| Reuse the v1.0 release | `"full"` | `False` | Reuses release artifacts when the code, data inventory, and configuration match |
| Rebuild data audit and splits | `"smoke"` or `"full"` | `True` | Recomputes the data preparation state and invalidates dependent work as needed |

Use `smoke` first. Change to `full` only after checking the smoke outputs. For the best release-cache compatibility, use the `v1.0` Git tag with the same dataset version and directory layout.

## 5. Run the notebook

```bash
conda activate waste-early-exit
jupyter lab Waste_Aware_Early_Exit_Experiment.ipynb
```

Choose the `Python (waste-early-exit)` kernel, confirm the parameter cell, then use **Run All**.

Every main table and chart appears directly in the notebook. The same results are saved under `artifacts/` for the report.

In `full` mode, **Run All** completes seeds 42, 123, and 2026 automatically. Each seed uses an isolated folder such as `artifacts/seed_42/`. The notebook then displays and saves the raw seed results, mean, standard deviation, and 95% confidence interval under `artifacts/aggregate/results/`.

## 6. Resume or reuse cached work

Each stage checks the normalized configuration and data inventory before accepting a cached artifact. Changes to the dataset, seed, model, or training configuration invalidate dependent work. Checkpoints preserve the model, optimizer, scheduler, epoch, best score, random state, and effective batch size.

Keep `FORCE_REBUILD = False` to reuse compatible checkpoints and caches. Set it to `True` when the data audit and splits must be rebuilt. To deliberately repeat training with the same configuration, first preserve or move the matching checkpoint and cached-logit files out of `artifacts/`. Only load PyTorch checkpoints obtained from a release you trust.

# Dataset details

The local dataset version used for this project contains 15,515 images in 12 class folders:

| Class | Images |
|---|---:|
| clothes | 5,325 |
| shoes | 1,977 |
| paper | 1,050 |
| biological | 985 |
| battery | 945 |
| cardboard | 891 |
| plastic | 865 |
| white-glass | 775 |
| metal | 769 |
| trash | 697 |
| green-glass | 629 |
| brown-glass | 607 |

The pipeline verifies every image, records dimensions, finds exact duplicates with SHA-256, groups near duplicates with a perceptual hash, and blocks duplicate groups from crossing data splits.

The outer data policy is:

- 70% development data;
- 10% calibration data for exit temperatures;
- 10% threshold-validation data for class difficulty and routing search;
- 10% locked test data for one final evaluation.

Inside the 70% development block, one fold is used only for model validation and early stopping. This keeps calibration and routing data clean.

# Full-run findings (v1.0)

The full experiment completed all 34 stages for seeds 42, 123, and 2026. The recorded orchestration time was 58,610 seconds (16 hours, 16 minutes, and 50 seconds) on the local Apple M1 Pro/MPS system. The table reports means across the three locked test splits:

| Method | Accuracy | Macro F1 | Average FLOPs |
|---|---:|---:|---:|
| WACEE | 93.43% | 91.28% | 1.041B |
| ResNet-18 Final-only | 93.66% | 91.55% | 1.819B |
| Global-Calibrated | 93.38% | 91.22% | 1.041B |
| EfficientNet-B0 | 95.83% | 94.42% | 0.400B |
| MobileNetV3-Small | 94.36% | 92.46% | 0.059B |

The main findings are:

- WACEE reduces theoretical average FLOPs by **42.75%** relative to full ResNet-18 while its mean Macro F1 differs by only **0.27 percentage points**. The paired-bootstrap interval crosses zero, supporting comparable predictive quality within this experiment.
- WACEE routes **92.58%** of test images through Exit 1, **3.81%** through Exit 2, and only **3.60%** through the final exit on average. Most images therefore avoid the deepest computation.
- Accuracy reaches **93.43%** and Macro F1 reaches **91.28%** across three locked-test seeds. The Macro-F1 standard deviation is below one percentage point.
- The data pipeline found no corrupt images and prevented duplicate groups from crossing splits, so the reported locked-test results are not explained by detected split leakage.

**Conclusion:** WACEE preserves same-backbone classification quality while substantially reducing theoretical computation. These results establish a working foundation for resource-aware waste classification and motivate the deployment improvements described under Future work.

The machine-readable evidence is stored in `artifacts/aggregate/results/locked_test_seed_summary.csv`, `latency_seed_summary.csv`, `paired_bootstrap_seed_summary.csv`, and `conclusion_assessment.json`. These files are included in the [Checkpoint and result v1.0](https://github.com/blue1018/code/releases/tag/v1.0) archive rather than Git.

# Expected runtime

These estimates target the local Apple M1 Pro with 16 GB unified memory.

| Step | Smoke | Full |
|---|---:|---:|
| Environment and first model-weight download | 10-25 min | same one-time setup |
| Data audit and duplicate grouping | 2-6 min | 6-18 min across three seeds |
| Training, calibration, routing, and plots | 15-35 min | 24-72 hours across three seeds |

Runtime depends on thermal state, free memory, internet speed, batch-size fallback, and early stopping. After the first epoch, the notebook reports an observed time estimate instead of relying only on this planning range.

# Progress and logs

The notebook shows two levels of progress:

- overall progress across every seed and pipeline stage;
- local progress for the current model, epoch, training or validation batch, prediction batch, loss, image throughput, elapsed time, and ETA.

An active stage refreshes its display and writes a heartbeat at least once every 30 seconds, even when no batch has completed. This also covers data audit, threshold search, profiling, and other long non-training stages.

Batch progress is transient so notebook output stays bounded. Durable JSON-line logs record stage, model, epoch, metrics, cache hits, early stopping, MPS batch-size fallback, heartbeat, elapsed time, ETA, and exceptions:

```text
artifacts/logs/run.log                         # smoke seed log
artifacts/seed_42/logs/run.log                 # one full seed
artifacts/aggregate/logs/smoke_run.log         # smoke orchestration
artifacts/aggregate/logs/full_run.log          # three-seed orchestration
```

Each record includes a `run_id`, timestamp, seed, and event kind, so appended runs can be separated reliably.

# Experiment matrix

Static models:

- MobileNetV3-Small;
- EfficientNet-B0;
- ResNet-18 Final-only;
- WACEE dynamic ResNet-18.

Same-backbone routing methods:

- Fixed Exit 1;
- Fixed Exit 2;
- Global-Raw;
- Global-Calibrated;
- Waste-Aware-Calibrated;
- Oracle as a diagnostic upper bound only.

Ablations:

- no temperature calibration;
- no class awareness (`lambda=0`);
- no self-distillation;
- Exit 2 only;
- no difficulty shrinkage (`rho=0`);
- no per-class recall guardrails.

# Main outputs

The notebook shows and saves:

- data-quality and split tables;
- class distribution, image size, and sample plots;
- training curves;
- confusion matrices and per-class metrics;
- temperatures, ECE, NLL, and reliability diagrams;
- class difficulty and threshold heatmaps;
- exit-share charts;
- Macro F1 versus FLOPs, latency, and energy plots;
- model comparison and ablation tables;
- qualitative early-exit successes and failures.

Generated files are grouped here:

```text
artifacts/
├── seed_42/         # Full-mode outputs for one seed
├── seed_123/        # Full-mode outputs for one seed
├── seed_2026/       # Full-mode outputs for one seed
├── aggregate/       # Cross-seed tables
├── manifests/       # Image audit and hashes
├── splits/          # Frozen split CSV files
├── checkpoints/     # Best model states
├── cached_logits/   # Calibration, routing, and test logits
├── results/         # CSV and JSON results
├── figures/         # Report-ready plots
└── logs/            # Environment and energy logs
```

# MPS memory behavior

The project prefers MPS, then records the exact device in each run. It does not silently move a benchmark to CPU.

If MPS runs out of memory during training, the trainer restores the start-of-epoch state, halves the batch size, and retries. It stops with a clear error when the configured minimum batch size is reached.

# Energy reporting

The default CodeCarbon value is a software estimate. It can miss part of Apple GPU power, so the notebook labels it as estimated.

For optional whole-system measurement, open a second terminal and run:

```bash
./scripts/record_powermetrics.sh artifacts/logs/powermetrics.txt
```

This command asks for administrator permission. Stop it when the benchmark ends. The raw log is kept, and the parser reports the result as measured. Whole-system power can include unrelated background activity.

## Recorded MPS training power sample

A 60-second `powermetrics` sample was recorded on 2026-08-11 while the full experiment was training the Seed 2026 EfficientNet-B0 baseline on MPS (epoch 24). The Mac was a `MacBookPro18,1` running OS build `25F84`. Sampling ran from 11:57:50 to 11:58:50 Europe/Dublin at one-second intervals. The experiment log continued from approximately batch 129 to batch 134 during the measurement, so the power sampler did not interrupt training.

| Quantity | Mean | Minimum | Maximum |
|---|---:|---:|---:|
| CPU power | 7.471 W | 5.885 W | 14.429 W |
| GPU power | 1.018 W | 0.610 W | 4.929 W |
| Combined CPU + GPU + ANE power | 8.494 W | 6.703 W | 19.381 W |

The combined 60-second energy was **509.653 J**, equivalent to **0.14157 Wh** or **0.000141570 kWh**, across 60 valid samples. The raw `powermetrics` log is retained locally at `artifacts/logs/powermetrics_training_sample_2026-08-11.txt`; generated artifacts are intentionally excluded from Git.

This is a whole-system training snapshot, not a process-isolated or paired baseline-versus-WACEE measurement. It includes unrelated background activity and therefore does not by itself demonstrate that the early-exit method saves energy.

FLOPs, latency, and energy are reported separately. A lower FLOP count does not automatically prove lower real energy use.

# Literature evidence matrix

The table below is based on the actual local PDFs in `../paper`. `DIRECTLY USED` means a method or reporting rule appears in this code. `INDIRECT SUPPORT` means the paper supports feasibility or an alternative design, but that method is not implemented. `BACKGROUND` provides framing or survey coverage.

| Local paper | Year and source | What it studies | Evidence and result used here | Role in this project | Usage | Code link |
|---|---|---|---|---|---|---|
| [Adaptive Neural Networks for Efficient Inference](../paper/bolukbasi17a.pdf) | 2017, ICML | Per-sample selection of cheaper or deeper computation. | Easy samples can use early components while expensive processing is reserved for harder samples. | Supports adaptive depth and cost-aware evaluation. | **DIRECTLY USED** | `models.py`, `routing.py` |
| [Multi-Scale Dense Networks for Resource Efficient Image Classification](../paper/229_multi_scale_dense_networks_for.pdf) | 2018, ICLR | One network with multiple classifiers for anytime and budgeted prediction. | Joint intermediate classifiers can reuse computation and improve the accuracy-budget trade-off; the appendix also shows that side heads add real cost. | Supports the shared-backbone multi-exit design and counting head FLOPs. We do not copy MSDNet's dense multi-scale backbone. | **DIRECTLY USED** | `models.py`, `profiling.py` |
| [Knowledge Distillation by On-the-Fly Native Ensemble](../paper/NeurIPS-2018-knowledge-distillation-by-on-the-fly-native-ensemble-Paper.pdf) | 2018, NeurIPS | One-stage online distillation inside a multi-branch network. | A jointly trained internal teacher can improve branch generalization without a separate pretrained teacher. | Supports final-to-early self-distillation. Our teacher is the detached final exit, not ONE's gated ensemble teacher. | **DIRECTLY USED** | `losses.py` |
| [Shallow-Deep Networks: Understanding and Mitigating Network Overthinking](../paper/kaya19a.pdf) | 2019, ICML | Internal classifiers, early exits, and the overthinking problem. | Confidence-based exits reduce average inference cost while preserving original performance, but later layers can sometimes turn an early correct answer into an error. | Supports internal heads, fixed-depth baselines, and explicit error analysis by exit. | **DIRECTLY USED** | `models.py`, `experiments.py` |
| [On Calibration of Modern Neural Networks](../paper/guo17a.pdf) | 2017, ICML | Post-hoc confidence calibration for modern classifiers. | Scalar temperature scaling is simple, fast, and often highly effective; it is fitted on held-out validation logits. | Provides the exact per-exit temperature-scaling method. | **DIRECTLY USED** | `calibration.py` |
| [Class Based Thresholding in Early Exit Semantic Segmentation Networks](../paper/2210.15621v1.pdf) | 2022 preprint, later IEEE SPL | Different exit thresholds for different predicted classes. | Class-dependent thresholds let easy classes leave earlier; experiments report lower compute with limited mIoU loss, including a stated 23% cost reduction in one comparison. | Directly motivates predicted-class-specific thresholds. Our difficulty statistic and image-classification task differ. | **DIRECTLY USED** | `routing.py` |
| [Adaptive Deep Neural Network Inference Optimization with EENet](../paper/Ilhan_Adaptive_Deep_Neural_Network_Inference_Optimization_With_EENet_WACV_2024_paper.pdf) | 2024, WACV | Learning exit scores and assigning samples under an inference budget. | EENet combines confidence with class-wise scores and optimizes exit allocation/thresholds across vision and NLP tasks. | Supports validation-based, class-aware threshold optimization and joint multi-exit training. We use a transparent grid search rather than EENet's learned scheduler. | **DIRECTLY USED** | `training.py`, `routing.py` |
| [Green AI](../paper/1907.10597v3.pdf) | 2020, Communications of the ACM version | Efficiency as a research objective alongside predictive quality. | Recommends reporting computational cost and the development effort behind results instead of publishing accuracy alone. | Defines the sustainability framing and the accuracy-efficiency comparison. | **DIRECTLY USED** | `profiling.py`, notebook results |
| [Towards the Systematic Reporting of the Energy and Carbon Footprints of Machine Learning](../paper/20-312.pdf) | 2020, JMLR | Consistent energy and carbon accounting for ML experiments. | Shows that FLOPs can be uncorrelated with measured energy and that carbon depends on grid intensity and location. | Drives separate FLOPs, latency, energy-source, and CO2e reporting with raw logs. | **DIRECTLY USED** | `profiling.py` |
| [Early-Exit with Class Exclusion for Efficient Inference of Neural Networks](../paper/2309.13443v2.pdf) | 2024, IEEE AICAS | Removing irrelevant classes at intermediate layers instead of relying only on confidence. | The paper reports up to 33.06% FLOP reduction and observes that some classes exit earlier while cat/dog-like classes need deeper features. | Supports class difficulty as a meaningful routing signal. Class exclusion itself is not implemented. | **INDIRECT SUPPORT** | `routing.py` design rationale |
| [Jointly-Learned Exit and Inference for a Dynamic Neural Network: JEI-DNN](../paper/ICLR-2024-jointly-learned-exit-and-inference-for-a-dynamic-neural-network-Paper-Conference.pdf) | 2024, ICLR | Joint learning of inference modules and exit gates through a cost-aware bilevel objective. | Identifies train-test mismatch in decoupled gates and reports a better cost-performance trade-off with improved uncertainty characterization. | Strong alternative to threshold routing and a limitation of this simpler project. | **INDIRECT SUPPORT** | README limitations |
| [BEEM: Boosting Performance of Early Exit DNNs Using Multi-Exit Classifiers as Experts](../paper/ICLR-2025-beem-boosting-performance-of-early-exit-dnns-using-multi-exit-classifiers-as-experts-Paper-Conference.pdf) | 2025, ICLR | Aggregating confidence across consistent neighboring exits. | Treating exits as weighted experts yields reported speedups of 1.5x-2.1x on language and image-captioning tasks while keeping competitive accuracy. | Motivates multi-exit consistency as a future routing feature. BEEM aggregation is not implemented. | **INDIRECT SUPPORT** | README future work |
| [Beyond Greedy Exits: Improved Early Exit Decisions for Risk Control and Reliability](../paper/NeurIPS-2025-beyond-greedy-exits-improved-early-exit-decisions-for-risk-control-and-reliability-Paper-Conference.pdf) | 2025, NeurIPS | Online threshold adaptation under distribution shift using a bandit and reliability-aware reward. | Reports 1.70x-2.10x speedup with under 2% performance drop across several tasks and gives risk guarantees. | Motivates recall guardrails and highlights the deployment risk of static thresholds. Online bandit adaptation is not implemented. | **INDIRECT SUPPORT** | `routing.py`, README limitations |
| [Early Exit Ensembles for Uncertainty Quantification](../paper/qendro21a.pdf) | 2021, PMLR ML4H | Treating exits as a weight-sharing ensemble for uncertainty. | Across medical imaging tasks, early-exit ensembles improve calibration metrics by up to 2x in reported comparisons and use one forward pass. | Supports exit-level calibration and uncertainty analysis. This project routes one prediction instead of ensembling exits. | **INDIRECT SUPPORT** | `calibration.py`, `metrics.py` |
| [Revisiting the Calibration of Modern Neural Networks](../paper/NeurIPS-2021-revisiting-the-calibration-of-modern-neural-networks-Paper.pdf) | 2021, NeurIPS | Calibration trends across newer image architectures, pretraining, size, and distribution shift. | Calibration varies with architecture and can degrade under shift; accuracy alone does not guarantee trustworthy confidence. | Supports measuring ECE/NLL separately at every exit. | **INDIRECT SUPPORT** | `metrics.py`, notebook calibration section |
| [SelectiveNet: A Deep Neural Network with an Integrated Reject Option](../paper/geifman19a.pdf) | 2019, ICML | Joint classification and selective rejection at a target coverage. | End-to-end selection improves risk-coverage trade-offs over confidence rejection in the paper's experiments. | Provides a risk-coverage comparison point. Rejection is not the same as continuing to a deeper exit. | **INDIRECT SUPPORT** | README related methods |
| [Early-Exit Deep Neural Network - A Comprehensive Survey](../paper/43343_Early_Exit_Deep_Neural_N.pdf) | 2025, ACM Computing Surveys 57(3) | Architectures, training strategies, exit policies, deployment, and open problems. | The survey stresses that branch placement, branch design, training strategy, and exit policy all affect performance and still lack a universal optimum. | Organizes the background and helps state the limits of a two-exit ResNet-18 study. | **BACKGROUND** | README scope |

# Future work

- **Translate FLOPs into runtime gains.** The current Python/MPS implementation records 10.376 ms median batch-1 latency for WACEE versus 4.475 ms for full ResNet-18. Vectorized routing, compiled execution, static exit graphs, and deployment-specific batching should reduce control-flow overhead and make theoretical savings observable in latency.
- **Strengthen class-aware routing.** WACEE improves mean Macro F1 by 0.05 percentage points over Global-Calibrated at similar FLOPs, but the paired-bootstrap interval crosses zero. Larger routing-validation sets, richer class-difficulty features, and adaptive thresholds can target a clearer class-aware advantage.
- **Combine WACEE with efficient backbones.** EfficientNet-B0 and MobileNetV3-Small are stronger static accuracy-compute baselines in this experiment. Adding calibrated waste-aware exits to these backbones may combine architectural efficiency with per-image adaptive computation.
- **Expand statistical and dataset coverage.** Future experiments should use more than three seeds, replace the small-sample normal confidence approximation, curate the 1,653 cross-class duplicate-conflict rows, and evaluate additional waste datasets and real camera images.
- **Measure deployment energy directly.** CodeCarbon has incomplete MPS coverage, and the current `powermetrics` record is a whole-system training snapshot. Paired baseline-versus-WACEE inference measurements are needed to establish an energy benefit.
- **Adapt under distribution shift.** Online threshold adjustment, stronger recall guardrails, jointly learned gates, class exclusion, and exit-ensemble consistency should be tested when class frequencies or image conditions change.
