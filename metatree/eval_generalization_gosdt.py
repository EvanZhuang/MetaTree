import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import datetime

import datasets
from datasets import Dataset
import torch
from accelerate import Accelerator, DistributedType
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, concatenate_datasets
import evaluate
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
from metatree.model_metatree import LlamaForMetaTree

from metatree.decision_tree_class import DecisionTree, DecisionTreeForest
from metatree.run_train import parse_args, preprocess_dimension_patch

from gosdt import GOSDT
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import pathlib
from metatree.eval_generalization import tree_eval_process

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    #ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    ddp_kwargs = DistributedDataParallelKwargs(static_graph=True)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs], **accelerator_log_kwargs)
    #accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    ## Here we load the generalization dataset
    if args.dataset_name is not None:
        if args.load_from_local:
            raw_datasets = datasets.load_from_disk(args.dataset_name)
        else:
            raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, num_proc=args.preprocessing_num_workers)
        
        if "input_x" not in raw_datasets.column_names:
            # X, Y formatting
            #raw_datasets = raw_datasets.rename_column("X", "input_x")
            raw_datasets = raw_datasets.map(tree_eval_process, num_proc=os.cpu_count()-1, batched=False, fn_kwargs={"num_classes": args.n_class})
        raw_datasets.set_format(type='torch', columns=['input_x', 'input_y'])

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    ## Custom Configs
    if not args.inference_only:
        config.max_position_embeddings = args.block_size
        config.max_sequence_length = args.block_size
        config.n_feature = args.n_feature
        config.n_class = args.n_class
        config.depth = args.depth
        config.normalize = args.normalize
        config.gradient_checkpointing = args.gradient_checkpointing
        config.backward_window = args.backward_window
        config.sigma = args.sigma

        # Small LLAMA
        config.hidden_size = args.hidden_size
        config.intermediate_size = 4*config.hidden_size
        config.num_hidden_layers = args.num_hidden_layers
        config.num_attention_heads = args.num_heads
    else:
        # Generation Depth Specification
        config.depth = args.depth
    generation_config = GenerationConfig.from_model_config(config)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    block_size = args.block_size    

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    lm_datasets = raw_datasets.map(preprocess_dimension_patch, batched=False, num_proc=args.preprocessing_num_workers, fn_kwargs={"n_feature": args.n_feature, "n_class": args.n_class})
    train_dataset = lm_datasets["train"]
    
    if args.eval_on_train:
        eval_dataset = lm_datasets["train"]
    else:
        eval_dataset = lm_datasets["validation"]
    
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size, num_workers=args.preprocessing_num_workers
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size, num_workers=args.preprocessing_num_workers
    )


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("treegen_stats_gosdt", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    if args.inference_only:
        # Override training arguments to only do inference
        starting_epoch = 0
        args.num_train_epochs = 1

    # metrics
    if args.n_class == 2:
        metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
        metatree_eval_metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    else:
        metric = evaluate.combine(["accuracy"])
        metatree_eval_metric = evaluate.combine(["accuracy"])
    epoch = 0
    decision_tree_forest = []
    total_loss = 0
    losses = []
    train_acc = []
    # For each train batch, we generate a decision tree that fits this batch
    # The decision tree is later evaluated on the entire eval dataset
    while len(decision_tree_forest) < args.max_train_steps:
        for step, batch in enumerate(train_dataloader):
            if args.deepspeed:
                for key in batch.keys():
                    if batch[key].dtype == torch.float32: batch[key] = batch[key].to(dtype=torch.float16)
            X = batch["input_x"].detach().cpu().numpy()
            Y = batch["input_y"].argmax(dim=-1).detach().cpu().numpy()

            # GBDT parameters for threshold and lower bound guesses
            n_est = 128
            max_depth = args.depth + 1 # Cause the standard depth is 1 less than the actual depth

            # guess lower bound
            clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
            clf.fit(X, Y)
            yl = clf.predict(X)
            warm_labels = clf.predict(X)

            # save the labels from lower bound guesses as a tmp file and return the path to it.
            labelsdir = pathlib.Path('/tmp/warm_lb_labels')
            labelsdir.mkdir(exist_ok=True, parents=True)
            labelpath = labelsdir / 'warm_label.tmp'
            labelpath = str(labelpath)
            pd.DataFrame(warm_labels, columns=["class_labels"]).to_csv(labelpath, header="class_labels",index=None)

            # train GOSDT model
            config = {
                        "regularization": 0.0001,
                        "allow_small_reg": True,
                        "depth_budget": max_depth,
                        "warm_LB": True,
                        "path_to_labels": labelpath,
                        "time_limit": 100,
                        "similar_support": True,
                        "worker_limit": 60,
                        "look_ahead": True,
                    }

            model = GOSDT(config)

            model.fit(pd.DataFrame(X), pd.DataFrame(Y))
            y_pred = model.predict(pd.DataFrame(X))
            train_acc.append(accuracy_score(Y, y_pred))
            decision_tree_forest.append(model)
            if len(decision_tree_forest) >= args.max_train_steps:
                break

    for step, batch in enumerate(train_dataloader):
        with torch.no_grad():
            forest_pred = torch.zeros_like(batch["input_y"]).float()
            for tree_model in decision_tree_forest:
                tree_pred = tree_model.predict(pd.DataFrame(batch["input_x"].detach().cpu().numpy()))
                tree_pred = torch.from_numpy(tree_pred).to(device=batch["input_x"].device)
                forest_pred += torch.nn.functional.one_hot(tree_pred, num_classes=batch["input_y"].shape[-1]).float()
            forest_pred /= len(decision_tree_forest)
            forest_pred = forest_pred.argmax(dim=-1)
            metric.add_batch(predictions=forest_pred.view(-1), references=batch["input_y"].argmax(dim=-1).view(-1))
    
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            forest_pred = torch.zeros_like(batch["input_y"]).float()
            for tree_model in decision_tree_forest:
                tree_pred = tree_model.predict(pd.DataFrame(batch["input_x"].detach().cpu().numpy()))
                tree_pred = torch.from_numpy(tree_pred).to(device=batch["input_x"].device)
                forest_pred += torch.nn.functional.one_hot(tree_pred, num_classes=batch["input_y"].shape[-1]).float()
            forest_pred /= len(decision_tree_forest)
            forest_pred = forest_pred.argmax(dim=-1)
            metatree_eval_metric.add_batch(predictions=forest_pred.view(-1), references=batch["input_y"].argmax(dim=-1).view(-1))
        
    train_output = metric.compute()
    eval_output = metatree_eval_metric.compute()
    #no_sample_eval_output = no_sample_metric.compute()

    #logger.info(f"epoch {epoch}: eval_acc: {no_sample_eval_output['accuracy']} eval_loss: {eval_loss:.4f}")
    logger.info(f"epoch {epoch}: eval_acc: {eval_output['accuracy']} num_of_trees: {len(decision_tree_forest)}")

    if args.with_tracking:
        accelerator.log(
            {
                "train": train_output,
                "eval": eval_output,
                "train_loss": total_loss,
                "epoch": epoch,
                "step": completed_steps,
            },
            step=completed_steps,
        )

    if args.push_to_hub and epoch < args.num_train_epochs - 1:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            #tokenizer.save_pretrained(args.output_dir)
            repo.push_to_hub(
                commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
            )

    if args.checkpointing_steps == "epoch":
        output_dir = f"epoch_{epoch}"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)
        accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()