## LegalBenchmark

This project is utilization for legalbench, please refer to [legalbench](https://github.com/HazyResearch/legalbench) for more detials.

## Usage

```
DATA_DIR=`<legalbench dataset dir>` # download for huggingface
MODEL_NAME_OR_PATH=`<your test model>`
MODEL_TYPE=`<define your model name>`
OUTPUT_PATH="output/logs"
NUM_GPUS=1

for TASK_TYPE in conclusion rhetoric rule issue
do
    echo "Running task $TASK_TYPE"
    python usinglegalbench.py run_batch_eval
    --model_name_or_path $MODEL_NAME_OR_PATH
    --model_type $MODEL_TYPE
    --data_dir $DATA_DIR
    --task_type $TASK_TYPE
    --output_path $OUTPUT_PATH
    --num_gpus $NUM_GPUS
done
```