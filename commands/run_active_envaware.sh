#!/usr/bin/env bash

# ─── HF model cards used across alignment methods ──────────────────────────────
# OnlineDPO:
# • policy LLM ........ RLHFlow/LLaMA3-SFT
# • reward model ...... RLHFlow/pair-preference-model-LLaMA3-8B
#
# APL:
# • IMDb policy ....... openai-community/gpt2-large
# • TL;DR policy ...... EleutherAI/pythia-1b
#
# SEA:
# • policies .......... trl-lib/pythia-{1b,2.8b,6.9b}-deduped-tldr-sft EleutherAI/pythia-{1b,2.8b,6.9b}
# • reward oracle ..... Skywork/Skywork-Reward-Llama-3.1-8B
#
# XPO:
# • policy LLM ........ RLHFlow/LLaMA3-SFT
# • reward model ...... RLHFlow/pair-preference-model-LLaMA3-8B
#
# DivReward:
# • HH policy ........ jtatman/gpt2-open-instruct-v1-Anthropic-hh-rlhf
# • IMDB ........ edbeeching/gpt2-large-imdb
# • TL;DR ........ pvduy/pythia-1B-sft-summarize-tldr
# ──────────────────────────────────────────────────────────────

# ─── HF dataset cards used across alignment methods ──────────────────────────────
# OnlineDPO:
# • HH ........ Anthropic/hh-rlhf - train: 161k, 8.55k
# • TL;DR ........ UCL-DARK/openai-tldr-summarisation-preferences - train: 92.9k, val: 33.1k, test: 50.7k
#
# APL:
# • IMDB ........ stanfordnlp/imdb (No need to use this; it is weird)
# • TL;DR ........ (No need to use this; it is weird)
#
# SEA:
# • TL;DR ........ lkevinzc/tldr-with-sft-reference - train: 117k, val: 6.45k, 6.55k
#
# DivReward:  
# • HH ........ Anthropic/hh-rlhf - train: 161k, 8.55k
# • IMDB ........ yuasosnin/imdb-dpo - train: 9k, test 1k
# • TL;DR ........ UCL-DARK/openai-tldr-summarisation-preferences - train: 92.9k, val: 33.1k, test: 50.7k
# ──────────────────────────────────────────────────────────────

# ─── Model and data combinations ──────────────────────────────
# • HH ........ data: won-bae/bpo_preference_hh_data - train: 10k, 0.5k 
#               model: MohamadBazzi/gemma-bpo-sft
# • IMDB ........ data: yuasosnin/imdb-dpo - not conversational, train: 9k, test 1k 
#                 model: edbeeching/gpt2-large-imdb
# • TL;DR ........ data: trl-lib/tldr - train: 117k, val: 6.45k, test: 6.55k  
#                  model: trl-lib/pythia-2.8b-deduped-tldr-sft
# • Ultrafeedback ........ data: trl-lib/ultrafeedback_binarized - train: 62.1k, test: 1k 
#                          model: Qwen/Qwen2-1.5B-Instruct or RichardErkhov/sahandrez_-_sft-Qwen2.5-1.5B-ultrafeedback-gguf
# ──────────────────────────────────────────────────────────────


# ========================================================================================

append_args() {
    local value="$1"
    [[ -z "$value" ]] && return
    local tmp=()
    read -r -a tmp <<<"$value"
    CMD+=("${tmp[@]}")
}

sanitize_path_component() {
    python - <<'PY' "$1"
import re, sys
value = sys.argv[1].strip() if len(sys.argv) > 1 else ""
value = re.sub(r'[^0-9A-Za-z._+-]+', '-', value)
value = re.sub(r'-{2,}', '-', value)
value = value.strip('-')
if value.startswith("activeDap-"):
    trimmed = value[len("activeDap-") :]
    value = trimmed or value
value = value.strip('-') or 'unknown'
print(value)
PY
}

# Fix it for each dataset
DATASET=${DATASET:-activeDap/sft-hh-data} # tl;dr: trl-lib/tldr, hh: won-bae/bpo_preference_hh_data
MODEL=${MODEL:-MohamadBazzi/gemma-bpo-sft} # tl;dr: trl-lib/pythia-2.8b-deduped-tldr-sft, hh: MohamadBazzi/gemma-bpo-sft

JUDGE=${JUDGE:-} # meta-llama/Meta-Llama-3-70B-Instruct
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-OpenAssistant/reward-model-deberta-v3-large-v2} # tl;dr: trl-lib/pythia-2.8b-deduped-tldr-rm, hh: OpenAssistant/reward-model-deberta-v3-large-v2
EVAL_JUDGE=${EVAL_JUDGE:-OpenAssistant/reward-model-deberta-v3-large-v2} # tl;dr: trl-lib/pythia-2.8b-deduped-tldr-rm, hh: OpenAssistant/reward-model-deberta-v3-large-v2

SEQ_LEN=${SEQ_LEN:-256} # tl;dr: 256, hh: 256
# Do not set a default for MAX_STEPS; prefer epoch-centric configs by default.
# If MAX_STEPS is provided explicitly, it will override epochs.
MAX_STEPS=${MAX_STEPS:-}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-16}
UPDATES_PER_SAMPLE=${UPDATES_PER_SAMPLE:-4} # OnlineDPO: UPDATES_PER_SAMPLE = 1, HybridDPO: UPDATES_PER_SAMPLE > 1
SCALE_UPDATES_PER_SAMPLE=${SCALE_UPDATES_PER_SAMPLE:-true} # If true, scale UPS by number of GPUs

# different by dataset
LEARNING_RATE=${LEARNING_RATE:-5.0e-5} # tl;dr: 5.0e-7, hh: 1.0e-5
LORA_LEARNING_RATE=${LORA_LEARNING_RATE:-5.0e-5}
NUM_EVAL_PROMPTS=${NUM_EVAL_PROMPTS:-500} # tl;dr: 1000, hh: 500
SEED=${SEED:-1}
FILTER_OUT=${FILTER_OUT:-none} # wrong, right, none
QUERY_BATCH_SIZE=${QUERY_BATCH_SIZE:-}
APL_N=${APL_N:-}
# ========================================================================================


# Set up paths
WANDB_MODE=${WANDB_MODE:-disabled}
export WANDB_MODE
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
HF_XET_CACHE=${HF_XET_CACHE:-$HF_HOME/xet}
HF_ASSETS_CACHE=${HF_ASSETS_CACHE:-$HF_HOME/assets}
HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HF_HOME/token}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
HF_DATASETS_DISABLE_PROGRESS_BARS=${HF_DATASETS_DISABLE_PROGRESS_BARS:-1}

export HF_HOME
export HF_HUB_CACHE
export HF_XET_CACHE
export HF_ASSETS_CACHE
export HF_TOKEN_PATH
export HF_DATASETS_CACHE
export HF_DATASETS_DISABLE_PROGRESS_BARS

# garbage_collection_threshold=<float> A value between 0.0 and 1.0. The GPU memory cleanup threshold. 
# The higher the value, the more frequently it is cleaned. 
# expandable_segments=True Introduced in PyTorch 2.1+. Very effective in solving memory fragmentation. 
# Makes dynamic memory segments expandable.
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.7
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.7}
export PYTORCH_CUDA_ALLOC_CONF

TRL_ACCELERATE_CONFIG=${TRL_ACCELERATE_CONFIG:-$PWD/trl/accelerate_configs/ddp.yaml}
TRITON_DISABLE_AUTOTUNE=${TRITON_DISABLE_AUTOTUNE:-1}
VLLM_USE_FLASH_ATTENTION=${VLLM_USE_FLASH_ATTENTION:-0}

export TRL_ACCELERATE_CONFIG
export TRITON_DISABLE_AUTOTUNE
export VLLM_USE_FLASH_ATTENTION

# Determine GPU count
if [[ -n "$SLURM_GPUS" ]]; then
    NUM_GPUS=$SLURM_GPUS
elif [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
else
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [[ -z "$NUM_GPUS" || "$NUM_GPUS" -lt 1 ]]; then
        NUM_GPUS=1
    fi
fi

# Scale updates_per_sample by world size so that "per-sample updates" stays
# invariant across number of GPUs. With N GPUs, a value of U will become U*N.
if [[ "$SCALE_UPDATES_PER_SAMPLE" == "true" || "$SCALE_UPDATES_PER_SAMPLE" == "True" || "$SCALE_UPDATES_PER_SAMPLE" == "1" || "$SCALE_UPDATES_PER_SAMPLE" == "yes" || "$SCALE_UPDATES_PER_SAMPLE" == "on" ]]; then
  if [[ -n "$NUM_GPUS" && "$NUM_GPUS" -gt 1 ]]; then
    EFFECTIVE_UPDATES_PER_SAMPLE=$(( UPDATES_PER_SAMPLE * NUM_GPUS ))
  else
    EFFECTIVE_UPDATES_PER_SAMPLE=$UPDATES_PER_SAMPLE
  fi
else
  EFFECTIVE_UPDATES_PER_SAMPLE=$UPDATES_PER_SAMPLE
fi
    
HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} # offline mode by default


# Eval and log steps
EVAL_STRATEGY=${EVAL_STRATEGY:-steps} # always no since we will generate responses and compare their win rates using judges
EVAL_STEPS=${EVAL_STEPS:-0.05}
LOGGING_STEPS=${LOGGING_STEPS:-$EVAL_STEPS}
SAVE_STEPS=${SAVE_STEPS:-0.05}

# Training parameters
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}

# Keep global batch constant across GPUs (by default)
FIX_GLOBAL_BATCH=${FIX_GLOBAL_BATCH:-true}
BASE_NUM_GPUS=${BASE_NUM_GPUS:-1}
if [[ "$FIX_GLOBAL_BATCH" == "true" || "$FIX_GLOBAL_BATCH" == "True" || "$FIX_GLOBAL_BATCH" == "1" || "$FIX_GLOBAL_BATCH" == "yes" || "$FIX_GLOBAL_BATCH" == "on" ]]; then
  if [[ -n "$NUM_GPUS" && "$NUM_GPUS" -gt 0 ]]; then
    BASE_GLOBAL_BATCH=$(( PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * BASE_NUM_GPUS ))
    NEW_PER_DEVICE=$(( (PER_DEVICE_BATCH_SIZE * BASE_NUM_GPUS) / NUM_GPUS ))
    if [[ "$NEW_PER_DEVICE" -lt 1 ]]; then NEW_PER_DEVICE=1; fi
    DENOM=$(( NEW_PER_DEVICE * NUM_GPUS ))
    if [[ "$DENOM" -lt 1 ]]; then DENOM=1; fi
    # ceil division for new GA to match base global batch
    NEW_GA=$(( (BASE_GLOBAL_BATCH + DENOM - 1) / DENOM ))
    if [[ "$NEW_GA" -lt 1 ]]; then NEW_GA=1; fi
    PER_DEVICE_BATCH_SIZE=$NEW_PER_DEVICE
    GRADIENT_ACCUMULATION_STEPS=$NEW_GA
  fi
fi

BATCH_RATIO=$(awk -v bs=$PER_DEVICE_BATCH_SIZE -v gpus=$NUM_GPUS -v ga=$GRADIENT_ACCUMULATION_STEPS 'BEGIN { printf "%.4f", bs * gpus * ga / 128 }')

# Default eval batch to 8x train batch (can be overridden explicitly).
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-$((PER_DEVICE_BATCH_SIZE * 8))}

# Unless explicitly provided, use the eval batch size for reward scoring as well.
# This keeps RM throughput high by default while still allowing manual overrides
# via the REWARD_BATCH_SIZE environment variable.
REWARD_BATCH_SIZE=${REWARD_BATCH_SIZE:-$PER_DEVICE_EVAL_BATCH_SIZE}

# Warmup: prefer ratio by default; allow explicit WARM_STEPS to override
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
# If user explicitly set WARM_STEPS keep it, otherwise leave it unset to use ratio
if [[ -z "${WARM_STEPS+x}" ]]; then
  unset WARM_STEPS
fi


# Set up active learning parameters
QUERY_FREQ_FACTOR=${QUERY_FREQ_FACTOR:-1} # OnlineDPO: UPDATES_PER_SAMPLE = 1, HybridDPO: UPDATES_PER_SAMPLE >= 1
QUERY_STRATEGY=${QUERY_STRATEGY:-random}
if [[ -z "${NUM_QUERY+x}" ]]; then
    NUM_QUERY=$(( PER_DEVICE_BATCH_SIZE * NUM_GPUS * QUERY_FREQ_FACTOR * GRADIENT_ACCUMULATION_STEPS ))
fi
EXTRACTOR_TYPE=${EXTRACTOR_TYPE:-roberta} # roberta, modernberta, sentence_transformer, llm
EMBEDDING_INPUT_TYPE=${EMBEDDING_INPUT_TYPE:-template} # prompt, concat, template
RADIUS=${RADIUS:-1.0}
NORMALIZE=${NORMALIZE:-true}

# Base output directory (allows redirecting to outputs_crossover, etc.)
OUTPUT_ROOT_DIR=${OUTPUT_ROOT_DIR:-outputs}
OUTPUT_ROOT_DIR=${OUTPUT_ROOT_DIR%/}

# Set up model parameters
ALIGNMENT=${ALIGNMENT:-online_dpo} # [dpo, online_dpo, xpo, sea]
LOSS_TYPE=${LOSS_TYPE:-sigmoid} # [sigmoid, ipo]
ENABLE_LORA=${ENABLE_LORA:-true}
SAVE_ONLY_MODEL=${SAVE_ONLY_MODEL:-true}

if [[ "$ENABLE_LORA" == "true" || "$ENABLE_LORA" == "True" || "$ENABLE_LORA" == "1" ]]; then
  EFFECTIVE_LEARNING_RATE=${LORA_LEARNING_RATE:-$LEARNING_RATE}
else
  EFFECTIVE_LEARNING_RATE=$LEARNING_RATE
fi

# Handle extra arguments in case one passes accelerate configs.
if [[ -z "${EXTRA_TRAINING_ARGS+x}" ]]; then
  if [[ "$ENABLE_LORA" == "true" || "$ENABLE_LORA" == "True" || "$ENABLE_LORA" == "1" ]]; then
    EXTRA_TRAINING_ARGS="--use_peft --torch_dtype bfloat16 --lora_r 32 --lora_alpha 64 --lora_dropout 0.05"
  else
    EXTRA_TRAINING_ARGS="--torch_dtype bfloat16"
  fi
fi

OPTIMAL_BATCH=${OPTIMAL_BATCH:-0}

if [[ -n "${BETA+x}" ]]; then
  BETA_SET=1
else
  BETA_SET=0
fi
if [[ -n "${SIMPO_GAMMA+x}" ]]; then
  SIMPO_GAMMA_SET=1
else
  SIMPO_GAMMA_SET=0
fi

while getopts ":f:a:m:j:J:K:w:e:d:b:o:r:q:n:x:t:h:z:N:s:F:u:l:G:B:" flag; do
  case "${flag}" in
    f) HF_HUB_OFFLINE=${OPTARG};;
    a) ALIGNMENT=${OPTARG};;
    b) OPTIMAL_BATCH=${OPTARG};;
    l) LOSS_TYPE=${OPTARG};;
    m) MODEL=${OPTARG};;
    j) JUDGE=${OPTARG};;
    J) EVAL_JUDGE=${OPTARG};; # New flag for eval_judge
    w) REWARD_MODEL_PATH=${OPTARG};;
    e) EMBED_MODEL=${OPTARG};;
    d) DATASET=${OPTARG};;
    b) PER_DEVICE_BATCH_SIZE=${OPTARG};;
    o) OUTPUT_DIR=${OPTARG};;
    u) UPDATES_PER_SAMPLE=${OPTARG};;
    q) QUERY_STRATEGY=${OPTARG};;
    n) NUM_QUERY=${OPTARG};;
    x) EXTRACTOR_TYPE=${OPTARG};;
    t) EMBEDDING_INPUT_TYPE=${OPTARG};;
    h) RADIUS=${OPTARG};;
    z) NORMALIZE=${OPTARG};;
    N) NUM_EVAL_PROMPTS=${OPTARG};;
    s) SEED=${OPTARG};;
    F) FILTER_OUT=${OPTARG};;
    G) SIMPO_GAMMA=${OPTARG}; SIMPO_GAMMA_SET=1;;
    B) BETA=${OPTARG}; BETA_SET=1;;
    :)                                         # If expected argument omitted:
        echo "Error: -${OPTARG} requires an argument."
        exit_abnormal;;                          # Exit abnormally.
    *)                                         # If unknown (any other) option:
        exit_abnormal;;                          # Exit abnormally.
  esac
done

if [[ "$LOSS_TYPE" == "sigmoid" ]]; then
    if [[ $BETA_SET -eq 0 ]]; then
        BETA=0.1
    fi
elif [[ "$LOSS_TYPE" == "ipo" ]]; then
    if [[ $BETA_SET -eq 0 ]]; then
        BETA=1.0
    fi
elif [[ "$ALIGNMENT" == "dpo" && "$LOSS_TYPE" == "hinge" ]]; then
    if [[ $BETA_SET -eq 0 ]]; then
        BETA=0.002 # online_dpo do not support hinge (SLiC)
    fi
elif [[ "$LOSS_TYPE" == "simpo" ]]; then
    if [[ $BETA_SET -eq 0 ]]; then
        BETA=2.0
    fi
    if [[ $SIMPO_GAMMA_SET -eq 0 ]]; then
        SIMPO_GAMMA=1.0
        SIMPO_GAMMA_SET=1
    fi
else
    echo "Unknown ALIGNMENT or LOSS_TYPE value: ALIGNMENT=$ALIGNMENT, LOSS_TYPE=$LOSS_TYPE"
    exit 1
fi

export HF_HUB_OFFLINE=$HF_HUB_OFFLINE

if [[ -n "$SIMPO_GAMMA" ]]; then
    EXTRA_TRAINING_ARGS+=" --simpo_gamma $SIMPO_GAMMA"
fi

if [ "$NORMALIZE" = "true" ] || [ "$NORMALIZE" = "True" ]; then
    EXTRA_ACTIVE_ARGS="--normalize"
else
    EXTRA_ACTIVE_ARGS=""
fi

DATASET_ABBV=""
case "$DATASET" in
  "Anthropic/hh-rlhf") DATASET_ABBV="hh" ;;
  "won-bae/bpo_preference_hh_data") DATASET_ABBV="hh" ;;
  "activeDap/sft-hh-data") DATASET_ABBV="help-bpo" ;;
  "activeDap/sft-harm-data") DATASET_ABBV="harm-bpo" ;;
  "activeDap/ultrafeedback_chosen") DATASET_ABBV="ultrafeedback" ;;
  "yuasosnin/imdb-dpo") DATASET_ABBV="imdb" ;;
  "UCL-DARK/openai-tldr-summarisation-preferences") DATASET_ABBV="tldr" ;;
  "trl-lib/tldr") DATASET_ABBV="tldr" ;;
  "trl-lib/ultrafeedback_binarized") DATASET_ABBV="ultrafeedback" ;;
esac

DATASET_ADDR=$(sanitize_path_component "$DATASET")
if [[ -z "$DATASET_ABBV" || "$DATASET_ABBV" == "unknown" ]]; then
    DATASET_ABBV="$DATASET_ADDR"
fi
echo "Dataset abbreviation is: $DATASET_ABBV"
echo "Dataset identifier is: $DATASET_ADDR"

MODEL_ABBV=""
case "$MODEL" in
  "Qwen/Qwen2.5-3B-Instruct") MODEL_ABBV="qwen3b" ;;
  "edbeeching/gpt2-large-imdb") MODEL_ABBV="gpt2" ;;
  "trl-lib/pythia-1b-deduped-tldr-sft") MODEL_ABBV="pythia1b" ;;
  "trl-lib/pythia-2.8b-deduped-tldr-sft") MODEL_ABBV="pythia2.8b" ;;
  "sahandrez/sft-Qwen2.5-1.5B-ultrafeedback") MODEL_ABBV="qwen1.5b" ;;
  "MohamadBazzi/gemma-bpo-sft") MODEL_ABBV="gemma2b" ;;
  "google/gemma-2b") MODEL_ABBV="gemma2b-vanilla" ;;
esac

MODEL_ADDR=$(sanitize_path_component "$MODEL")
if [[ -z "$MODEL_ABBV" || "$MODEL_ABBV" == "unknown" ]]; then
    MODEL_ABBV="$MODEL_ADDR"
fi

echo "Model abbreviation is: $MODEL_ABBV"
echo "Model identifier is: $MODEL_ADDR"

case "$EVAL_JUDGE" in
  "gpt-4o-mini") eJUDGE="openai" ;;
  "pair_rm") eJUDGE="rm" ;;
  "hf") eJUDGE="hf" ;;
  "trl-lib/pythia-1b-deduped-tldr-rm") eJUDGE="pythia1b" ;;
  "trl-lib/pythia-2.8b-deduped-tldr-rm") eJUDGE="pythia2.8b" ;;
  "OpenAssistant/reward-model-deberta-v3-large-v2") eJUDGE="deberta" ;;
  "Skywork/Skywork-Reward-V2-Qwen3-8B") eJUDGE="skywork-v2-qwen3" ;;
  "Skywork/Skywork-Reward-V2-Llama-3.1-8B") eJUDGE="skywork-v2-llama31" ;;
  Skywork/Skywork-Reward-V2-*) eJUDGE="skywork-v2" ;;
  "PKU-Alignment/beaver-7b-v1.0-reward") eJUDGE="beaver7b" ;;
  "RLHFlow/pair-preference-model-LLaMA3-8B") eJUDGE="llama3-pairrm" ;;
  "nvidia/Qwen-3-Nemotron-32B-Reward") eJUDGE="nemotron32" ;;
  *) eJUDGE="unknown" ;;
esac

echo "Judge abbreviation is: $eJUDGE"

if [[ "$EXTRA_TRAINING_ARGS" == *"--use_peft"* ]]; then
  LORA_TAG="lora"
else
  LORA_TAG="nolora"
fi

# Build a default output dir reflecting steps or epochs (steps take precedence)
_train_len_component=""
if [[ -n "$MAX_STEPS" ]]; then
  _train_len_component="steps${MAX_STEPS}"
elif [[ -n "$NUM_TRAIN_EPOCHS" ]]; then
  _train_len_component="epochs${NUM_TRAIN_EPOCHS}"
else
  # Reflect HF default (3) if neither is explicitly set, purely for naming consistency
  _train_len_component="epochs3"
fi
DEFAULT_OUTPUT_DIR=${OUTPUT_ROOT_DIR}/${DATASET_ADDR}/${MODEL_ADDR}_${eJUDGE}_${ALIGNMENT}_${QUERY_STRATEGY}_query${NUM_QUERY}_update_per_sample${EFFECTIVE_UPDATES_PER_SAMPLE}_${_train_len_component}_batch${PER_DEVICE_BATCH_SIZE}_gpu${NUM_GPUS}_loss_${LOSS_TYPE}_${LORA_TAG}/
if [[ -z "${OUTPUT_DIR+x}" ]]; then
  OUTPUT_DIR=$DEFAULT_OUTPUT_DIR
fi
echo "Output directory resolved to: $OUTPUT_DIR"

SAVE_ONLY_MODEL_ARG=""
if [[ "$EXTRA_TRAINING_ARGS" != *"--save_only_model"* && "$EXTRA_TRAINING_ARGS" != *"--no_save_only_model"* ]]; then
  case "$SAVE_ONLY_MODEL" in
    true|True|1|yes|Yes|YES|on|On|ON)
      SAVE_ONLY_MODEL_ARG="--save_only_model"
      ;;
    false|False|0|no|No|NO|off|Off|OFF)
      SAVE_ONLY_MODEL_ARG=""
      ;;
    *)
      echo "Warning: Unrecognized SAVE_ONLY_MODEL value '$SAVE_ONLY_MODEL'. Expected true/false. Defaulting to saving optimizer state."
      ;;
  esac
fi
# Handle dpo or online dpo specific args
if [[ "$ALIGNMENT" == "dpo" ]]; then
  ONLINE_OR_OFFLINE_ARGS=""
else
  # ONLINE_OR_OFFLINE_ARGS="--use_vllm" # TODO: switch to vllm once an error (vLLM + peft) is fixed
  ONLINE_OR_OFFLINE_ARGS=""
  # Handle judge or reward model for online dpo
  if [[ -n "$JUDGE" && -z "$REWARD_MODEL_PATH" ]]; then
    JUDGE_OR_REWARD_MODEL=" --judge $JUDGE"
  elif [[ -z "$JUDGE" && -n "$REWARD_MODEL_PATH" ]]; then
    JUDGE_OR_REWARD_MODEL=" --reward_model_path $REWARD_MODEL_PATH --missing_eos_penalty 1.0"
  else
    JUDGE_OR_REWARD_MODEL=""
  fi
  ONLINE_OR_OFFLINE_ARGS="$ONLINE_OR_OFFLINE_ARGS $JUDGE_OR_REWARD_MODEL"
fi

# Handle eval_judge parameter
EVAL_JUDGE_ARGS=""
if [[ -n "$EVAL_JUDGE" ]]; then
  EVAL_JUDGE_ARGS=" --eval_judge $EVAL_JUDGE"
fi

# Handle evaluation strategy and steps
EVAL_ARGS=" --num_eval_prompts $NUM_EVAL_PROMPTS"
if [[ "$EVAL_STRATEGY" != "no" ]]; then
  EVAL_ARGS="$EVAL_ARGS --eval_strategy $EVAL_STRATEGY"
  if [[ -n "$EVAL_STEPS" ]]; then
    EVAL_ARGS="$EVAL_ARGS --eval_steps $EVAL_STEPS"
  fi
fi

if [[ "${TRL_ACCELERATE_CONFIG}" == "" ]]; then
  EXTRA_ACCELERATE_ARGS=""
else
  EXTRA_ACCELERATE_ARGS="--config_file $TRL_ACCELERATE_CONFIG"
  # For DeepSpeed configs we need to set the `--fp16` flag to comply with our configs exposed
  # on `examples/accelerate_configs` and our runners do not support bf16 mixed precision training.
  if [[ $TRL_ACCELERATE_CONFIG == *"deepspeed"* ]]; then
    EXTRA_TRAINING_ARGS="--fp16"
  fi
fi
echo "Using accelerate config: $TRL_ACCELERATE_CONFIG"

if [[ "$JUDGE" == *"gpt"* ]] || [[ "$EVAL_JUDGE" == *"gpt"* ]]; then
    if [ -f .env ]; then
        export $(cat .env | xargs)
        echo "Loaded environment variables for GPT judges - JUDGE: $JUDGE, EVAL_JUDGE: $EVAL_JUDGE"
    else
        echo "Warning: Using GPT judge but no .env file found"
    fi
fi
CMD=(accelerate launch)
append_args "$EXTRA_ACCELERATE_ARGS"
CMD+=(--num_processes "$NUM_GPUS" --mixed_precision bf16)
if [[ "$NUM_GPUS" -ge 2 ]]; then
  CMD+=(--multi_gpu)
fi
CMD+=(--num_machines 1 --dynamo_backend no "$(pwd)/trl/scripts/active.py")
CMD+=(--alignment "$ALIGNMENT" --loss_type "$LOSS_TYPE" --beta "$BETA" --learning_rate "$EFFECTIVE_LEARNING_RATE" --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS")
# Apply warmup via ratio unless explicit steps provided
if [[ -n "${WARM_STEPS+x}" && -n "$WARM_STEPS" ]]; then
  CMD+=(--warmup_steps "$WARM_STEPS")
else
  CMD+=(--warmup_ratio "$WARMUP_RATIO")
fi
append_args "$ONLINE_OR_OFFLINE_ARGS"
append_args "$EVAL_JUDGE_ARGS"
append_args "$EVAL_ARGS"
# Core training arguments
CMD+=(--model_name_or_path "$MODEL" --dataset_name "$DATASET" --output_dir "$OUTPUT_DIR" --logging_steps "$LOGGING_STEPS" --save_steps "$SAVE_STEPS")

# Prefer steps if provided; otherwise allow epoch-based control
if [[ -n "$MAX_STEPS" ]]; then
  CMD+=(--max_steps "$MAX_STEPS")
fi
if [[ -n "$NUM_TRAIN_EPOCHS" ]]; then
  CMD+=(--num_train_epochs "$NUM_TRAIN_EPOCHS")
fi
if [[ -n "$SAVE_ONLY_MODEL_ARG" ]]; then
  CMD+=("$SAVE_ONLY_MODEL_ARG")
fi
CMD+=(--per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" --max_length "$SEQ_LEN" --updates_per_sample "$EFFECTIVE_UPDATES_PER_SAMPLE" --query_strategy "$QUERY_STRATEGY" --num_query "$NUM_QUERY" --filter_out "$FILTER_OUT" --extractor_type "$EXTRACTOR_TYPE" --embedding_input_type "$EMBEDDING_INPUT_TYPE" --radius "$RADIUS")
append_args "$EXTRA_ACTIVE_ARGS"
CMD+=(--optimal_batch "$OPTIMAL_BATCH" --seed "$SEED" --report_to wandb)
append_args "$EXTRA_TRAINING_ARGS"
if [[ -n "$REWARD_BATCH_SIZE" ]]; then
  CMD+=(--reward_batch_size "$REWARD_BATCH_SIZE")
fi
if [[ -n "$QUERY_BATCH_SIZE" ]]; then
  CMD+=(--query_batch_size "$QUERY_BATCH_SIZE")
fi
if [[ -n "$APL_N" ]]; then
  CMD+=(--apl_n "$APL_N")
fi

echo "Starting program..."
echo "Configuration:"
echo "  NUM_GPUS: $NUM_GPUS"
# echo "  DETECTED GPUs index for training: $TRAINING_INDEX"
# echo "  DETECTED GPUs index for judge: $JUDGE_INDEX"
echo "  ALIGNMENT: $ALIGNMENT"
echo "  MODEL: $MODEL"
echo "  DATASET: $DATASET"
echo "  JUDGE: $JUDGE"
echo "  EVAL_JUDGE: $EVAL_JUDGE"
echo "  EVAL_STRATEGY: $EVAL_STRATEGY"
echo "  EVAL_STEPS: $EVAL_STEPS"
echo "  ROUNDS: $ROUND"
echo "  BASE_LEARNING_RATE: $LEARNING_RATE"
echo "  LORA_LEARNING_RATE: $LORA_LEARNING_RATE"
echo "  EFFECTIVE_LEARNING_RATE: $EFFECTIVE_LEARNING_RATE"
echo "  QUERY_STRATEGY: $QUERY_STRATEGY"
echo "  NUM_QUERY: $NUM_QUERY"
echo "  UPDATES_PER_SAMPLE (effective): $EFFECTIVE_UPDATES_PER_SAMPLE (base=$UPDATES_PER_SAMPLE, gpus=$NUM_GPUS)"
echo "  OPTIMAL_BATCH: $OPTIMAL_BATCH"
echo "  PER_DEVICE_BATCH_SIZE: $PER_DEVICE_BATCH_SIZE"
echo "  GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS"
echo ""

{ # try
    printf 'Executing command:\n  '
    printf '%q ' "${CMD[@]}"
    printf '\n'
    "${CMD[@]}"
    TRAINING_SUCCESS=$?
} || { # catch
    # save log for exception 
    echo "Operation Failed!"
    exit 1
}

if [ $TRAINING_SUCCESS -eq 0 ]; then
    echo "Training completed successfully. Output directory: $OUTPUT_DIR"
fi

exit $TRAINING_SUCCESS
    
