# DINO v2 VLAD Ablations
#
# Usage: bash ./scripts/dino_v2_vlad_ablations.sh
#
# 

declare -A model_dict
model_dict=(
    ["depth"]="/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc/pretraining/checkpoints_ce/dinov2_ms2_checkpoints_thermal_global_bigger_denser_no_night/tartanair/rgb_depth/2025-04-28_15-06-05/model1.pth"
)

# ---- Program arguments for user (after setting up datasets) ----
# Directory for storing experiment cache
# Dataset directory
data_dir="/ocean/projects/cis220039p/shared/datasets/tartanair_v2"
# data_dir=""
# Cache directory (where images and model cache will be stored)
cache_dir="$(dirname "${BASH_SOURCE[0]}"..)/../multiloc_cache"
# Datasets
datasets=("tartanair")
# datasets=("cart")
# Modalities
db_modality=("rgb")
q_modality=("depth")
#Sequences
seq_list=( 'ForestEnv/Data_easy/P000' )
# GPU
gpu=${1:-0}
export CUDA_VISIBLE_DEVICES=$gpu
# WandB parameters

# wandb_entity="jkarhade"
wandb_project="MultiLoc_inference"
# ----------- Main Experiment Code -----------
curr_run=0
start_time=$(date)
start_time_secs=$SECONDS
echo "Start time: $start_time"
# For each dataset
for dataset in ${datasets[*]}; do
for db_modality in ${db_modality[*]}; do
for seq in ${seq_list[*]}; do
    # Header
    echo -ne "\e[1;93m"
    echo "---"
    curr_run=$((curr_run+1))
    echo -ne "\e[0m"
    # Variables for experiment
    wandb_group="$dataset"
    wandb_name="${db_modality}_${q_modality}_${seq}"
    exp_id="ablations/$wandb_name"
    python_cmd="python3 dino_v2_plot_qual.py"
    python_cmd+=" --exp-id $exp_id"
    python_cmd+=" --db-modality $db_modality"
    python_cmd+=" --q-modality $q_modality"
    # if db_model is not rgb, then use the model_dict
    if [[ $db_modality != "rgb" ]]; then
        db_model=${model_dict[$db_modality]}
        python_cmd+=" --db-model $db_model"
    fi
    q_model=${model_dict[$q_modality]}
    python_cmd+=" --q-model $q_model"
    python_cmd+=" --prog.cache-dir ${cache_dir}"
    python_cmd+=" --prog.data-dir ${data_dir}"
    python_cmd+=" --prog.ms2_seq ${seq}"
    python_cmd+=" --prog.dataset-name ${dataset}"
    python_cmd+=" --prog.use-wandb"
    python_cmd+=" --prog.wandb-proj ${wandb_project}"
    # python_cmd+=" --prog.wandb-entity ${wandb_entity}"
    python_cmd+=" --prog.wandb-group ${wandb_group}"
    python_cmd+=" --prog.wandb-run-name ${wandb_name}"
    echo -ne "\e[0;36m"
    echo $python_cmd
    echo -ne "\e[0m"
    # run_start_time=$(date)
    run_start_secs=$SECONDS
    $python_cmd
    # run_end_time=$(date)
    run_end_secs=$SECONDS
    # run_dur=$(echo $(date -d "$run_end_time" +%s) \
    #         - $(date -d "$run_start_time" +%s) | bc -l)
    run_dur=$(( $run_end_secs - $run_start_secs ))
    echo -n "---- Run finished in (HH:MM:SS): "
    echo "`date -d@$run_dur -u +%H:%M:%S` ----"
done
done
done
end_time=$(date)
end_time_secs=$SECONDS
# dur=$(echo $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) | bc -l)
dur=$(( $end_time_secs - $start_time_secs ))
_d=$(( dur/3600/24 ))
echo "---- Ablation took (d-HH:MM:SS): $_d-`date -d@$dur -u +%H:%M:%S` ----"
echo "Starting time: $start_time"
echo "Ending time: $end_time"

