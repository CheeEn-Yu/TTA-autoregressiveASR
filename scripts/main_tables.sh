#!/usr/bin/env bash
# Shared implementation; source from a run_table*.sh entry point.
set -euo pipefail

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }

run_main_table() {
    local table="$1"
    shift
    local dry_run="${DRY_RUN:-0}"
    while (( $# )); do
        case "$1" in
            --dry-run) dry_run=1 ;;
            --help|-h)
                cat <<'HELP'
Usage: bash scripts/run_table{2_librispeechc,3_l2arctic,4_mls}.sh [--dry-run]
Activate your Python environment before running.

Environment variables:
  METHODS       Space-separated: source greedy_em em_seq em_tok em_tok_b (default: all)
  LS_C_ROOT     Table 2: LibriSpeech-C root containing NOISE/snr=10/{data_info.json,wav/}
  NOISES        Table 2: AA AC BA CM GS MU NB SD TP VC (default: all)
  L2ARCTIC_ROOT Table 3: root containing ACCENT/SPEAKER/{wav,transcript}/
  ACCENTS       Table 3: Arabic Chinese Hindi Korean Spanish Vietnamese (default: all)
  LANGUAGES     Table 4: dutch french german italian polish portuguese spanish (default: all)
  CUDA_VISIBLE_DEVICES  GPU selection (default: 0)
  PYTHON        Python executable, without additional arguments (default: python)
  NUM_DATA      false for all data, or a positive integer for a smoke run
  RESULTS_ROOT  Output root (default: REPO/results)
  RUN_ID        Output batch name (default: timestamp and process ID)
  DRY_RUN       1 prints commands without loading models or checking dataset files

Paper settings: whisper-base, 10 steps, G=16, AdamW lr=1e-3, seed=42.
Each output directory must be new; existing results are never appended by this launcher.
HELP
                return 0 ;;
            *) die "Unknown argument: $1 (use --help)" ;;
        esac
        shift
    done
    [[ "$dry_run" == 0 || "$dry_run" == 1 ]] || die 'DRY_RUN must be 0 or 1'
    local repo_root
    repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
    local python_bin="${PYTHON:-python}" num_data="${NUM_DATA:-false}"
    [[ "$num_data" == false || "$num_data" =~ ^[1-9][0-9]*$ ]] || die 'NUM_DATA must be false or a positive integer'
    local run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
    [[ "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || die 'RUN_ID must contain only letters, digits, _, . or -'
    local results_root="${RESULTS_ROOT:-$repo_root/results}"
    # Resolve user-supplied relative paths before changing to the repository.
    [[ "$results_root" == /* ]] || results_root="$PWD/$results_root"
    local methods_text="${METHODS-source greedy_em em_seq em_tok em_tok_b}"
    local domains_text allowed dataset root='' domain method language dataset_dir output
    case "$table" in
        2)
            dataset=librispeechc
            root="${LS_C_ROOT:-}"
            [[ -n "$root" ]] || die 'Set LS_C_ROOT to your LibriSpeech-C root'
            allowed='AA AC BA CM GS MU NB SD TP VC'
            domains_text="${NOISES-$allowed}" ;;
        3)
            dataset=l2arctic
            root="${L2ARCTIC_ROOT:-}"
            [[ -n "$root" ]] || die 'Set L2ARCTIC_ROOT to your L2Arctic root'
            allowed='Arabic Chinese Hindi Korean Spanish Vietnamese'
            domains_text="${ACCENTS-$allowed}" ;;
        4)
            dataset=multilibri
            allowed='dutch french german italian polish portuguese spanish'
            domains_text="${LANGUAGES-$allowed}" ;;
        *) die "Unknown table: $table" ;;
    esac
    if [[ -n "$root" && "$root" != /* ]]; then root="$PWD/$root"; fi
    local -a methods domains cmd method_args
    read -r -a methods <<< "$methods_text"
    read -r -a domains <<< "$domains_text"
    (( ${#methods[@]} && ${#domains[@]} )) || die 'Method and domain lists must not be empty'
    for method in "${methods[@]}"; do
        case "$method" in source|greedy_em|em_seq|em_tok|em_tok_b) ;; *) die "Unknown method: $method" ;; esac
    done
    # Validate all inputs before starting any experiments.
    for domain in "${domains[@]}"; do
        [[ " $allowed " == *" $domain "* ]] || die "Unknown domain: $domain"
        if [[ "$dry_run" == 0 ]]; then
            case "$table" in
                2) [[ -f "$root/$domain/snr=10/data_info.json" && -d "$root/$domain/snr=10/wav" ]] || die "Missing LibriSpeech-C data: $root/$domain/snr=10" ;;
                3) [[ -d "$root/$domain" ]] || die "Missing accent directory: $root/$domain" ;;
            esac
        fi
        for method in "${methods[@]}"; do
            output="$results_root/$run_id/table$table/$domain/$method"
            [[ ! -e "$output" ]] || die "Output already exists: $output; choose a new RUN_ID"
        done
    done
    if [[ "$dry_run" == 0 ]]; then command -v "$python_bin" >/dev/null || die "Python not found: $python_bin"; fi
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    cd -- "$repo_root"
    for domain in "${domains[@]}"; do
        language=en
        case "$table" in
            2) dataset_dir="$root/$domain/snr=10" ;;
            3) dataset_dir="$root/$domain" ;;
            4)
                dataset_dir="$domain"
                case "$domain" in
                    dutch) language=nl ;; french) language=fr ;; german) language=de ;;
                    italian) language=it ;; polish) language=pl ;; portuguese) language=pt ;; spanish) language=es ;;
                esac ;;
        esac
        # Hydra quoted string values preserve spaces and '=' in dataset/output paths.
        [[ "$dataset_dir" != *'"'* && "$dataset_dir" != *'\'* ]] || die 'Dataset paths cannot contain double quotes or backslashes'
        for method in "${methods[@]}"; do
            output="$results_root/$run_id/table$table/$domain/$method"
            [[ "$output" != *'"'* && "$output" != *'\'* ]] || die 'Output paths cannot contain double quotes or backslashes'
            case "$method" in
                source) method_args=(tta=false suta_method=aed sampling_strategy=random num_samples=0) ;;
                greedy_em) method_args=(tta=true suta_method=aed sampling_strategy=random num_samples=0) ;;
                em_seq) method_args=(tta=true suta_method=mc sampling_strategy=random num_samples=16) ;;
                em_tok) method_args=(tta=true suta_method=rb sampling_strategy=random num_samples=16) ;;
                em_tok_b) method_args=(tta=true suta_method=rb sampling_strategy=beam num_samples=16) ;;
            esac
            cmd=("$python_bin" main.py "exp_name=\"$output\""
                strategy=baseline task=transcribe asr=openai/whisper-base
                "dataset_name=$dataset" "dataset_dir=\"$dataset_dir\""
                "lang=$language" "asr_lang=$language" noise_dir=null snr=10
                steps=10 opt=AdamW lr=0.001 scheduler=null 'objective_f=[e_loss]'
                temp=1.0 topk=false rb_mode=none batch_size=1 "num_data=$num_data"
                seed=42 device=cuda use_wandb=false "${method_args[@]}")
            printf '\nTable %s | %s | %s\n' "$table" "$domain" "$method"
            printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
            printf '%q ' "${cmd[@]}"
            printf '\n'
            if [[ "$dry_run" == 0 ]]; then
                [[ ! -e "$output" ]] || die "Output already exists: $output"
                "${cmd[@]}"
            fi
        done
    done
}
