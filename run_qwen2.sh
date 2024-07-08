export CUDA_VISIBLE_DEVICES=2
export NCCL_IGNORE_DISABLED_P2P=1
export RAY_DEDUP_LOGS=0

DATA_DIR="/home/jye/huggingface/datasets/legalbench/data"
MODEL_NAME_OR_PATH="/home/jye/huggingface/pretrained_model/qwen/Qwen2-7B-Instruct"
MODEL_TYPE="qwen2_7b_instruct"
OUTPUT_PATH="output/logs"
NUM_GPUS=1

for TASK_TYPE in conclusion rhetoric rule issue
do
    echo "Running task $TASK_TYPE"
    python usinglegalbench.py run_batch_eval \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --model_type $MODEL_TYPE \
        --data_dir $DATA_DIR \
        --task_type $TASK_TYPE \
        --output_path $OUTPUT_PATH \
        --num_gpus $NUM_GPUS 
done