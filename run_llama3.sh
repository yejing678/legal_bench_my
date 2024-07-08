export CUDA_VISIBLE_DEVICES=7
export NCCL_IGNORE_DISABLED_P2P=1

DATA_DIR="/home/jye/huggingface/datasets/legalbench/data"
MODEL_NAME_OR_PATH="/home/jye/huggingface/pretrained_model/Meta-Llama-3-8B-Instruct/"
MODEL_TYPE="llama_3_8b_instruct"
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