# DINO v2 VLAD Ablations
#
# Usage: bash ./scripts/dino_v2_vlad_ablations.sh
#
# 

# ---- Program arguments for user (after setting up datasets) ----
# Directory for storing experiment cache
# Dataset directory
data_dir="/storage2/datasets/ms2_full"
# data_dir=""
# Cache directory (where images and model cache will be stored)
cache_dir="/storage2/datasets/jkarhade/multiloc_cache_thesis"
# Datasets
datasets=("thermal_day_night")
# datasets=("cart")
# Modalities
db_modality=("rgb")
q_modality=("lidar")
#Sequences
# seq_list=( '_2021-08-06-16-45-28' '_2021-08-06-11-37-46' '_2021-08-06-17-21-04') #for rainy unseen seq # '_2021-08-06-16-45-28' '_2021-08-06-16-19-00' '_2021-08-13-15-46-56' '_2021-08-13-17-06-04' '_2021-08-13-21-18-04' '_2021-08-13-21-36-10')
# seq_list=('_2021-08-13-22-03-03' '_2021-08-13-21-58-13')
# seq_list=( '_2021-08-06-16-45-28' '_2021-08-06-11-37-46' '_2021-08-06-17-21-04' '_2021-08-13-16-08-46' '_2021-08-13-22-03-03' '_2021-08-13-21-58-13')
seq_list=('_2021-08-06-16-45-28' '_2021-08-13-22-03-03')
# seq_list=("Idyll_wild" "big_bear") # "ocean_duck") #('_2021-08-13-16-08-46')
# GPU
gpu=${1:-0}
export CUDA_VISIBLE_DEVICES=$gpu
# WandB parameters

wandb_entity="jkarhade"
wandb_project="MultiLoc"
wandb_group="cart_eval"

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
    python_cmd="python dino_v2_plot_qual.py"
    python_cmd+=" --exp-id $exp_id"
    python_cmd+=" --db-modality $db_modality"
    python_cmd+=" --q-modality $q_modality"

    python_cmd+=" --prog.cache-dir ${cache_dir}"
    python_cmd+=" --prog.data-dir ${data_dir}"
    python_cmd+=" --prog.ms2_seq ${seq}"
    python_cmd+=" --prog.dataset-name ${dataset}"
    # python_cmd+=" --prog.use-wandb"
    python_cmd+=" --prog.wandb-proj ${wandb_project}"
    python_cmd+=" --prog.wandb-entity ${wandb_entity}"
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

