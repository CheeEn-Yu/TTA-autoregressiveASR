# Test-Time Adaptation for Autoregressive ASR

Official code for [**Rethinking Entropy Minimization in Test-Time Adaptation for Autoregressive Models**](https://arxiv.org/abs/2605.08186).

## Methods and implementation

The project performs **episodic test-time adaptation (TTA)** independently for each utterance. It first obtains the original transcription, repeatedly generates candidate transcriptions and minimizes entropy, and then evaluates the adapted model using greedy decoding. Only LayerNorm parameters in the Whisper encoder and decoder are updated. The pretrained model and optimizer are reinitialized for each utterance. Reference transcripts are used only for evaluation, not for the adaptation loss.

The paper shows that the gradient of the token-level entropy minimization objective for autoregressive models contains both a policy gradient (selection) term and a direct entropy (pathwise) term. Entropy minimization with teacher forcing on pseudo-labels captures only the latter. `RBLoss` combines both terms and uses a leave-one-out baseline to reduce policy gradient variance.

| Paper method / launcher name | `tta` | `suta_method` | `sampling_strategy` | `num_samples` | Implementation |
| --- | --- | --- | --- | --- | --- |
| Source / `source` | false | aed | random | 0 | Evaluate the original model |
| Greedy-EM / `greedy_em` | true | aed | random | 0 | `AED_suta`: entropy loss on a greedy pseudo-label |
| EM-seq / `em_seq` | true | mc | random | 16 | `MC_suta` + `MCLoss`: sequence-level policy gradient |
| EM-tok / `em_tok` | true | rb | random | 16 | `RB_suta` + `RBLoss`: selection + pathwise |
| EM-tok-b / `em_tok_b` | true | rb | beam | 16 | The same `RBLoss`, with candidates from beam search |

EM-tok and EM-tok-b use `rb_mode=none`, which **enables both loss terms**. EM-tok-b uses beam search during adaptation; final evaluation still uses greedy decoding. `strategy=baseline` is the shared execution strategy for all methods above. To disable adaptation, set `tta=false`.

## 1. Installation

Run the following commands from the repository root. Use a separate Python environment and an NVIDIA GPU; the paper used a single RTX 3090. Python 3.10 below is an example environment choice, not an author-provided version lock.

```bash
conda create -n tta-asr python=3.10 -y
conda activate tta-asr
python -m pip install -r requirements.txt
python -m pip check
```

Install compatible versions of `torch` and `torchaudio` with a CUDA build suitable for your machine.

Only `datasets==2.16.1` is pinned in `requirements.txt`. In particular, `generation_whisper.py` relies on internal Transformers Whisper generation APIs, so compatibility with arbitrary newer releases is not guaranteed. Import errors or generation argument errors may require a compatible Transformers version.

The first run downloads the `openai/whisper-base` model and processor from Hugging Face. MLS also requires a dataset download. Allow network access and sufficient space for model and dataset caches. You can also set `asr` to a local directory containing a downloaded model.

## 2. Data preparation

The main experiments use the following three loaders. Obtain the datasets separately; `res/` contains noise assets, not a complete evaluation dataset.

### LibriSpeech-C (Table 2)

The paper adds MS-SNSD noise to LibriSpeech `test-other` at an SNR of 10 dB. This loader expects **audio that has already been corrupted**, organized as follows:

```text
/data/LibriSpeech-C/
├── AA/snr=10/
│   ├── data_info.json
│   └── wav/
│       └── 123-456-0001.wav
├── AC/snr=10/...
└── GS/snr=10/...
```

Each `data_info.json` is a JSON array with at least these fields:

```json
[
  {
    "basename": "123-456-0001",
    "text": "THE REFERENCE TRANSCRIPTION",
    "length": 80000
  }
]
```

`basename` excludes the `.wav` extension. `length` is the number of audio samples, not seconds; the example represents 5 seconds at 16 kHz. Audio must be **16 kHz and mono**. This loader does not resample audio and uses `length` to filter out utterances longer than 20 seconds.

Noise codes are `AA` (airport announcement), `AC` (air conditioner), `BA` (babble), `CM` (copy machine), `GS` (Gaussian), `MU` (munching), `NB` (neighbors), `SD` (shutting door), `TP` (typing), and `VC` (vacuum cleaner).

For an individual run, `dataset_dir` points to a specific `NOISE/snr=10` directory. For the batch launcher, `LS_C_ROOT` points to the root containing the noise directories. Setting `snr=10` or `noise_dir` does not modify audio in this loader. The additional tool `corpus/noisyspeech_synthesizer.py` does not directly produce the required `data_info.json`; if you use it, you must also prepare the metadata and directory structure.

### L2-Arctic (Table 3)

Organize the data by accent and speaker, with matching audio and transcript files:

```text
/data/L2Arctic/
├── Arabic/
│   └── ABA/
│       ├── wav/arctic_a0001.wav
│       └── transcript/arctic_a0001.txt
├── Chinese/...
├── Hindi/...
├── Korean/...
├── Spanish/...
└── Vietnamese/...
```

Audio and transcript basenames must match. The Mandarin group in the paper uses the directory name `Chinese` in the launcher. `dataset_dir` can point to one speaker, one accent, or the entire dataset root; the Table 3 launcher runs each accent separately. The loader resamples to 16 kHz, preprocesses transcripts, excludes the corrupted files listed in the code, and sorts utterances by transcript length in descending order.

### Multilingual LibriSpeech (Table 4)

With `dataset_name=multilibri`, the loader calls:

```python
load_dataset("facebook/multilingual_librispeech", lang, split="test")
```

The positional `lang` argument here comes from **`dataset_dir`**. Set `dataset_dir` to the language name in English, rather than a local path.

| `dataset_dir` / `LANGUAGES` | `lang` and `asr_lang` |
| --- | --- |
| dutch | nl |
| french | fr |
| german | de |
| italian | it |
| polish | pl |
| portuguese | pt |
| spanish | es |

The main experiments use `noise_dir=null` without additional noise. `asr_lang` controls Whisper's decoding language; the launcher sets the language codes automatically.

## 3. Run a small experiment first

The root [config.yaml](config.yaml) provides the main paper settings: Whisper-base, 10 adaptation steps, 16 candidates, AdamW, and a learning rate of `1e-3`. The default method is EM-tok. `dataset_dir` is required; replace the example paths below with your actual paths.

```bash
# Print the composed configuration without loading data or models
python main.py --cfg job 'dataset_dir="/data/LibriSpeech-C/GS/snr=10"'

# Source: check data loading, model loading, and decoding
CUDA_VISIBLE_DEVICES=0 python main.py \
  'dataset_dir="/data/LibriSpeech-C/GS/snr=10"' \
  tta=false num_data=2 exp_name=results/smoke_source

# EM-tok: adapt for 10 steps on each of the first two utterances
CUDA_VISIBLE_DEVICES=0 python main.py \
  'dataset_dir="/data/LibriSpeech-C/GS/snr=10"' \
  num_data=2 exp_name=results/smoke_em_tok
```

Preserve the Hydra quoting shown above for paths containing `=` or spaces. Use a new `exp_name` for each run: direct calls to `main.py` append to existing logs, so reusing an output directory mixes results from different runs.

Examples with other datasets and methods:

```bash
# L2-Arctic, EM-tok-b
CUDA_VISIBLE_DEVICES=0 python main.py \
  dataset_name=l2arctic dataset_dir=/data/L2Arctic/Arabic \
  sampling_strategy=beam num_samples=16 \
  num_data=2 exp_name=results/smoke_arabic_em_tok_b

# German MLS, EM-seq (the first run may download the entire split)
CUDA_VISIBLE_DEVICES=0 python main.py \
  dataset_name=multilibri dataset_dir=german lang=de asr_lang=de \
  suta_method=mc num_samples=16 \
  num_data=2 exp_name=results/smoke_german_em_seq

# Greedy-EM
CUDA_VISIBLE_DEVICES=0 python main.py \
  'dataset_dir="/data/LibriSpeech-C/GS/snr=10"' \
  suta_method=aed sampling_strategy=random num_samples=0 \
  num_data=2 exp_name=results/smoke_greedy_em
```

`num_data=2` limits processing to the first two loader entries, but does not guarantee two evaluated utterances: the strategy skips audio longer than 20 seconds. Set `num_data=false` to process the full loader. This limit does not reduce the initial MLS dataset download.

## 4. Run the main experiments

The three launchers share [scripts/main_tables.sh](scripts/main_tables.sh). They run each domain and method combination sequentially using the main paper settings. The default methods are `source greedy_em em_seq em_tok em_tok_b`.

```bash
# Preview commands without loading models or checking dataset files
LS_C_ROOT=/data/LibriSpeech-C \
  bash scripts/run_table2_librispeechc.sh --dry-run

# Start with two entries from one domain and one method
LS_C_ROOT=/data/LibriSpeech-C NOISES=GS METHODS=em_tok_b NUM_DATA=2 \
  bash scripts/run_table2_librispeechc.sh

# Table 2: 10 noise domains at 10 dB
LS_C_ROOT=/data/LibriSpeech-C CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_table2_librispeechc.sh

# Table 3: 6 accents
L2ARCTIC_ROOT=/data/L2Arctic CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_table3_l2arctic.sh

# Table 4: 7 languages
CUDA_VISIBLE_DEVICES=0 bash scripts/run_table4_mls.sh
```

| Environment variable | Purpose |
| --- | --- |
| `METHODS` | Space-separated method names, e.g. `"source em_tok_b"` |
| `NOISES` / `ACCENTS` / `LANGUAGES` | Restrict domains for Table 2 / 3 / 4, respectively |
| `NUM_DATA` | A positive integer, or the default `false` for all entries |
| `CUDA_VISIBLE_DEVICES` | Defaults to `0`; use one GPU per process |
| `PYTHON` | Python executable; defaults to `python` |
| `RESULTS_ROOT` | Defaults to `results/` under the repository root |
| `RUN_ID` | Batch name; defaults to a timestamp and process ID |
| `DRY_RUN` | Set to `1` for the same behavior as `--dry-run` |

For example: `LANGUAGES="german french" METHODS="source em_tok_b" bash scripts/run_table4_mls.sh`. The launchers reject existing experiment output directories to prevent mixing results. Use `--help` for all options. To change the learning rate or number of steps, run `main.py` directly; the launchers explicitly override the main experiment parameters from the config.

## 5. Outputs and evaluation

Batch runs produce the following structure:

```text
results/<RUN_ID>/table2/GS/em_tok_b/
├── log.txt       # Configuration, timing, and Original / Step9 corpus WER
├── result.txt    # Per-utterance reference, original/adapted transcripts, or SKIPPED
└── figs/         # Usually empty; plotting is currently disabled
```

In `result.txt`, `ori(...)` and `step9(...)` contain per-utterance WER. The final `Original corpus WER` and `Step9 corpus WER` in `log.txt` are recomputed from all evaluated transcripts, rather than averaging per-utterance WER.

Steps are indexed from zero, so the result after 10 updates is `step9`. The current implementation records only the final adapted transcription, does not report WER at every intermediate step, and does not save adapted model checkpoints. Source runs report only original results. Hydra may also create run metadata under `outputs/`; the primary evaluation files are under `exp_name`.

To compare with the tables' Avg. columns, compute corpus WER for each domain and then take the arithmetic mean across domains. Pooling all transcripts across domains gives a different aggregate. The launchers generate individual experiment logs; they do not automatically assemble the paper tables.

## 6. Code structure and parameters

| File | Responsibility |
| --- | --- |
| `main.py` | Hydra entry point, data loading, per-utterance model/optimizer initialization, and metric aggregation |
| `config.yaml` | Default parameters and required dataset path |
| `strategies/registry.py` | Strategy registry; currently registers only `baseline` |
| `strategies/baseline.py` | Original decoding → candidate generation → TTA updates → final decoding |
| `suta.py` | `WhisperTTADecoder`, the three update methods, and LayerNorm parameter selection |
| `loss.py` | `MCLoss`, `RBLoss`, entropy, and log-probability calculations |
| `generation_whisper.py` | Simplified Whisper generation for short audio |
| `utils/data.py`, `corpus/` | Dataset dispatch, audio loading, and collation |
| `utils/model_utils.py` | Optimizer and scheduler setup |
| `utils/metrics.py`, `utils/logging_utils.py` | Corpus WER and experiment logging |
| `scripts/run_table*.sh` | Entry points for the three main experiments |

Common adjustments:

- `steps`, `lr`: number of updates and learning rate; the main settings are `10` and `0.001`.
- `num_samples`: number of candidates G. In the `random` branch, `0` generates one greedy candidate; **it does not disable adaptation**.
- `rb_mode=selection` / `pathwise`: retain only the policy gradient / entropy term for ablations. Full EM-tok uses `none`.
- `temp`, `topk`: temperature and vocabulary top-k for AED / RB entropy computation; defaults are `1.0` and `false`. `temp` does not control the current random sampler's sampling temperature.
- `scheduler=CosineAnnealingLR t_max=10 lr_min=0.0`: optional scheduler; the main experiments use `null`.

Documentation checks covered the paper, configuration keys, and launcher commands. Model inference, full Table 2–4 runs, and compatibility of the unpinned dependencies have not been verified in this environment. Complete the import checks and a small run before launching full experiments.

## Acknowledgments

We thank the authors of [Test-time-adaptation-ASR-SUTA](https://github.com/DanielLin94144/Test-time-adaptation-ASR-SUTA.git) and [SGEM](https://github.com/drumpt/SGEM.git) for making their codebase available. This repository builds on their work.

## Citation

If you use this codebase, please cite:

```bibtex
@article{huang2026rethinking,
  title={Rethinking Entropy Minimization in Test-Time Adaptation for Autoregressive Models},
  author={Huang, Wei-Ping and Yu, Chee-En and Lin, Guan-Ting and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2605.08186},
  year={2026}
}
```
