import os
import sys
from tqdm.auto import tqdm
import datasets
import fire
from vllm import LLM, SamplingParams

from tasks import TASKS, ISSUE_TASKS, RULE_TASKS, CONCLUSION_TASKS, INTERPRETATION_TASKS, RHETORIC_TASKS
from utils import generate_prompts
import pandas as pd
import numpy as np
from loguru import logger

from evaluation import evaluate

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_tasks_by_type(task_type):
    if task_type == 'issue':
        return ISSUE_TASKS
    elif task_type == 'rule':
        return RULE_TASKS
    elif task_type == 'conclusion':
        return CONCLUSION_TASKS
    elif task_type == 'interpretation':
        return INTERPRETATION_TASKS
    elif task_type == 'rhetoric':
        return RHETORIC_TASKS
    elif task_type == 'all':
        return TASKS
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def load_prompts(data_dir, dataset_name, file_type='test'):
    if file_type not in ["train", "test"]:
        raise ValueError("file_type most be 'train' or 'test'")
    
    file_name = f"{file_type}.tsv"
    file_path = os.path.join(data_dir, dataset_name, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not exsit!")
    
    df = pd.read_csv(file_path, sep='\t')
    test_df = df

    prompt_path=f"tasks/{dataset_name}/base_prompt.txt"
    with open(prompt_path) as in_file:
        prompt_template = in_file.read()
    
    prompts = generate_prompts(prompt_template=prompt_template, data_df=test_df)

    return prompts, test_df

def load_model(model_name_or_path, model_type):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token  
    return tokenizer, model.to("cuda")


def run_batch_eval(task_type, data_dir, model_name_or_path, model_type, output_path, num_gpus=1, batch_size=2):
    from torch.utils.data import DataLoader
    logger.add(f"{output_path}/{task_type}_{model_type}.log")
    task_list = get_tasks_by_type(task_type)
    logger.info(f"{task_type} has {len(task_list)} tasks: {task_list}")
    logger.info(f"Using model: {model_name_or_path}")
    results = {}
    markdown_table = "Dataset | Result\n--- | ---\n"
    tokenizer, model = load_model(model_name_or_path, model_type)

    for dataset_name in tqdm(task_list):
        logger.info(f"Evaluating {task_type} tasks for dataset: {dataset_name}")
        result_save_path = f"{output_path}/{task_type}_{dataset_name}_{model_type}.jsonl"
        result = []

        prompts, test_df = load_prompts(data_dir, dataset_name)
        prompt_data_loader = DataLoader(prompts, batch_size=batch_size)

        generations = []
        for batch in tqdm(prompt_data_loader):
            batch_inputs = tokenizer(batch, padding=True, return_tensors="pt").to("cuda")
            batch_generations = model.generate(**batch_inputs, max_new_tokens=512)
            batch_outputs = [tokenizer.decode(gen[len(inp):], skip_special_tokens=True) for gen, inp in zip(batch_generations, batch_inputs['input_ids'])]
            generations.extend(batch_outputs)
            result.extend([{"prompt": p, "generation": g} for p, g in zip(batch, batch_outputs)])

        eval_results = evaluate(dataset_name, generations, test_df["answer"].tolist())
        results[dataset_name] = eval_results
        result_summary = eval_results  # Modify as needed to extract summary from results
        markdown_table += f"{dataset_name} | {result_summary:.4f}\n"
        logger.info(f"Result for dataset {dataset_name}: {result_summary:.4f}")

        with open(result_save_path, "w") as f:
            for line in result:
                f.write(f"{line}\n")
        logger.info(f"Results saved to: {result_save_path}")
    logger.info("## Evaluation Results:\n" + markdown_table)
    return results

def run_eval(task_type, data_dir, model_name_or_path, model_type, output_path, num_gpus=1):
    logger.add(f"{output_path}/{task_type}_{model_type}.log")
    task_list = get_tasks_by_type(task_type)
    logger.info(f"{task_type} has {len(task_list)} tasks: {task_list}")
    logger.info(f"Using model: {model_name_or_path}")
    results = {}
    markdown_table = "Dataset | Result\n--- | ---\n"

    tokenizer, model = load_model(model_name_or_path, model_type)

    for dataset_name in tqdm(task_list):
        # torch.cuda.empty_cache()
        logger.info(f"Evaluating {task_type} tasks for dataset: {dataset_name}")
        result_save_path = f"{output_path}/{task_type}_{dataset_name}_{model_type}.jsonl"
        result = []

        prompts, test_df = load_prompts(data_dir, dataset_name)
        generations = []

        for prompt in tqdm(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            input_length = inputs.input_ids.shape[1]
            generation = model.generate(**inputs, max_new_tokens=512)[0][input_length:]
            generated_text = tokenizer.decode(generation, skip_special_tokens=True)
            generations.append(generated_text)

            result.append({
                "prompt": prompt,
                "generation": generated_text
            })
        
        eval_results = evaluate(dataset_name, generations, test_df["answer"].tolist())
        results[dataset_name] = eval_results
        markdown_table += f"{dataset_name} | {eval_results:.4f}\n"
        logger.info(f"Result for {datasett_name}: {eval_results:.4f}")

        with open(result_save_path, "w") as f:
            for line in result:
                f.write(f"{line}\n")
        logger.info(f"Results saved to: {result_save_path}")
    logger.info("## Evaluation Results:\n" + markdown_table)

    return results

def vllm_eval(task_type, data_dir, model_name_or_path, model_type, output_path, num_gpus=1):
    logger.add(f"{output_path}/{task_type}_{model_type}.log")
    task_list = get_tasks_by_type(task_type)
    logger.info(f"{task_type} has {len(task_list)} tasks: {task_list}")
    logger.info(f"Using model: {model_name_or_path}")
    results = {}
    markdown_table = "Dataset | Result\n--- | ---\n"

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
    llm = LLM(model=model_name_or_path, trust_remote_code=True, dtype=torch.float16, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.5)

    for dataset_name in tqdm(task_list):
        logger.info(f"Evaluating tasks for dataset: {dataset_name}")
        result_save_path = f"{output_path}/{task_type}_{dataset_name}_{model_type}.jsonl"
        result = []

        prompts, test_df = load_prompts(data_dir, dataset_name)
        generations = []

        for prompt in tqdm(prompts):
            output = llm.generate(prompt, sampling_params=sampling_params)
            generated_text = output[0].outputs[0].text
            generations.append(generated_text)

            result.append({
                "prompt": prompt,
                "generation": generated_text
            })
                  
        eval_results = evaluate(dataset_name, generations, test_df["answer"].tolist())
        logger.info(f"{dataset_name}: {eval_results}")
        results[dataset_name] = eval_results
        markdown_table += f"{dataset_name} | {eval_results:.4f}\n"

        with open(result_save_path, "w") as f:
            for line in result:
                f.write(f"{line}\n")
        logger.info(f"Results saved to: {result_save_path}")
    logger.info("## Evaluation Results:\n" + markdown_table)
    return results

if __name__ == "__main__":
    fire.Fire(
        {
            "run_eval": run_eval,
            "run_batch_eval": run_batch_eval,
            "vllm_eval": vllm_eval
        }
    )