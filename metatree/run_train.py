import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import datetime, glob

import datasets
from datasets import Dataset
import torch
from accelerate import Accelerator, DistributedType
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DummyOptim, DummyScheduler, GradScalerKwargs

from datasets import load_dataset, concatenate_datasets
import evaluate
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch._dynamo
torch._dynamo.config.suppress_errors = True

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

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_eval_steps",
        type=int,
        default=None,
        help="Total number of eval steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=4,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--n_feature",
        type=int,
        default=16,
        help="Max Number of Data dimensionality",
    )
    parser.add_argument(
        "--n_class",
        type=int,
        default=10,
        help="Maximum number of classes",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Tree Depth.",
    )
    parser.add_argument(
        "--n_episode",
        type=int,
        default=1000,
        help="episode size in each epoch",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="Gaussian Width",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help=(
            "Apply Power Transform Normalization"
        ),
    )
    parser.add_argument(
        "--deepspeed",
        action="store_true",
        help=(
            "Deepspeed flag"
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=(
            "Gradient checkpointing flag"
        ),
    )
    parser.add_argument(
        "--load_from_local",
        action="store_true",
        help=(
            "Are we loading from disk?"
        ),
    )
    parser.add_argument(
        "--inference_only",
        action="store_true",
        help=(
            "Are we only doing inference?"
        ),
    )
    parser.add_argument(
        "--backward_window",
        type=int,
        default=0,
        help="memory length, default 0 means no limit on backward window",
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=2,
        help="How many layers are we using for the transformer",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=12,
        help="How many attention heads are we using for the transformer",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="Hidden size of the transformer",
    )
    parser.add_argument(
        "--eval_on_train",
        action="store_true",
        help=(
            "Evaluation on train set, test model capacity"
        ),
    )
    parser.add_argument(
        "--clean_label",
        action="store_true",
        help=(
            "Evaluate on Clean/Real Label"
        ),
    )
    parser.add_argument(
        "--no_mapping",
        action="store_true",
        help=(
            "Indicates we don't need mapping the dimensions"
        ),
    )
    parser.add_argument(
        "--inference_topk_sample",
        action="store_true",
        help=(
            "In tree inference, determines whether we use topk sampling"
        ),
    )
    parser.add_argument(
        "--data_filter_threshold",
        type=float,
        default=0.0,
        help="Filter out training acc below this threshold.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args

def preprocess_dimension_patch(data, n_feature, n_class=2):
    input_x, input_y, input_y_clean = data["input_x"], data["input_y"], data["input_y_clean"]
    input_x = torch.nn.functional.pad(input_x, (0, n_feature - input_x.shape[-1]), mode='constant', value=0)
    input_y = torch.nn.functional.pad(input_y, (0, n_class - input_y.shape[-1]), mode='constant', value=0)
    input_y_clean = torch.nn.functional.pad(input_y_clean, (0, n_class - input_y_clean.shape[-1]), mode='constant', value=0)
    data["input_x"] = input_x
    data["input_y"] = input_y
    data["input_y_clean"] = input_y_clean
    return data



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
    ddp_kwargs = DistributedDataParallelKwargs(static_graph=False, find_unused_parameters=False)
    kwargs = GradScalerKwargs(init_scale=2**16, growth_factor=2.0, backoff_factor=0.5, growth_interval=1000, enabled=True)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs, kwargs], **accelerator_log_kwargs)
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
            create_repo(repo_name, exist_ok=True, token=args.hub_token, private=True)
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

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if len(args.dataset_name.split()) == 1:
            # Only One Dataset
            if args.load_from_local:
                raw_datasets = datasets.load_from_disk(args.dataset_name)
            else:
                raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, num_proc=args.preprocessing_num_workers, streaming=True)
        else:
            # Concating the Dataset list together
            dataset_list = args.dataset_name.split()
            raw_datasets = None
            if args.load_from_local:
                dataset_list = args.dataset_name.split()
                train_files, validation_files = [], []
                for dataset_name in dataset_list:
                    train_files += list(glob.glob("{}/train/*.arrow".format(dataset_name)))
                    validation_files += list(glob.glob("{}/validation/*.arrow".format(dataset_name)))
                data_files = {"train": train_files, "validation": validation_files}
                raw_datasets = datasets.load_dataset("arrow", data_files=data_files, num_proc=os.cpu_count()-1, streaming=False)
            else:
                for dataset_name in dataset_list:
                    if raw_datasets is None:
                        raw_datasets = load_dataset(dataset_name, args.dataset_config_name, num_proc=args.preprocessing_num_workers)
                        if not args.no_mapping:
                            raw_datasets['train'] = raw_datasets['train'].cast_column("input_x", datasets.Sequence(feature=datasets.Sequence(feature=datasets.Value(dtype='float32'))))
                            raw_datasets['train'] = raw_datasets['train'].cast_column("split_threshold", datasets.Sequence(feature=datasets.Sequence(feature=datasets.Value(dtype='float32'))))
                            raw_datasets['validation'] = raw_datasets['validation'].cast_column("input_x", datasets.Sequence(feature=datasets.Sequence(feature=datasets.Value(dtype='float32'))))
                            raw_datasets['validation'] = raw_datasets['validation'].cast_column("split_threshold", datasets.Sequence(feature=datasets.Sequence(feature=datasets.Value(dtype='float32'))))

                        raw_datasets.set_format(type='torch', columns=['input_x', 'input_y', 'input_y_clean', "status", "split_threshold", "split_dimension"])
                    else:
                        dataset_cache = load_dataset(dataset_name, args.dataset_config_name, num_proc=args.preprocessing_num_workers)
                        if not args.no_mapping:
                            dataset_cache['train'] = dataset_cache['train'].cast_column("input_x", datasets.Sequence(feature=datasets.Sequence(feature=datasets.Value(dtype='float32'))))
                            dataset_cache['train'] = dataset_cache['train'].cast_column("split_threshold", datasets.Sequence(feature=datasets.Sequence(feature=datasets.Value(dtype='float32'))))
                            dataset_cache['validation'] = dataset_cache['validation'].cast_column("input_x", datasets.Sequence(feature=datasets.Sequence(feature=datasets.Value(dtype='float32'))))
                            dataset_cache['validation'] = dataset_cache['validation'].cast_column("split_threshold", datasets.Sequence(feature=datasets.Sequence(feature=datasets.Value(dtype='float32'))))

                        dataset_cache.set_format(type='torch', columns=['input_x', 'input_y', 'input_y_clean', "status", "split_threshold", "split_dimension"])
                        raw_datasets["train"] = concatenate_datasets([raw_datasets["train"], dataset_cache["train"]])
                        raw_datasets["validation"] = concatenate_datasets([raw_datasets["validation"], dataset_cache["validation"]])
                        raw_datasets.set_format(type='torch', columns=['input_x', 'input_y', 'input_y_clean', "status", "split_threshold", "split_dimension"])
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )

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
    
    
    if args.model_name_or_path:
        model = LlamaForMetaTree.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
    else:
        logger.info("Training new model from scratch")
        #model = GPT2MetaTreeModel(config=config)
        model = LlamaForMetaTree(config=config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]

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

    if args.clean_label:
        raw_datasets.set_format(type='torch', columns=['input_x', 'input_y', "input_y_clean", "status", "split_threshold", "split_dimension"])
    else:
        raw_datasets.set_format(type='torch', columns=['input_x', 'input_y', "status", "split_threshold", "split_dimension"])
    if not args.no_mapping:
        lm_datasets = raw_datasets.map(preprocess_dimension_patch, batched=False, num_proc=os.cpu_count()-1, fn_kwargs={"n_feature": args.n_feature, "n_class": args.n_class})
    else:
        lm_datasets = raw_datasets
    if args.data_filter_threshold > 0.0:
        lm_datasets["train"] = lm_datasets["train"].filter(lambda example: example["rtg"] >= args.data_filter_threshold, num_proc=args.preprocessing_num_workers)
        lm_datasets["validation"] = lm_datasets["validation"].filter(lambda example: example["rtg"] >= args.data_filter_threshold, num_proc=args.preprocessing_num_workers)
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
        eval_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size, num_workers=args.preprocessing_num_workers
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

     # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    if args.max_eval_steps is None:
        # set the max eval steps to max possible steps
        args.max_eval_steps = args.num_train_epochs * num_update_steps_per_epoch

    if args.deepspeed:
        optimizer = DummyOptim(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-5)
        lr_scheduler = DummyScheduler(optimizer=optimizer, total_num_steps=args.max_train_steps * args.gradient_accumulation_steps)
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-5)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

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
        accelerator.init_trackers("metatree_pretrain", experiment_config)

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
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    if args.n_class > 2:
        no_sample_metric = evaluate.combine(["accuracy"])
    else:
        no_sample_metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        if not args.inference_only:
            for step, batch in enumerate(active_dataloader):
                if args.deepspeed:
                    for key in batch.keys():
                        if key == "input_x": continue
                        if batch[key].dtype != torch.float16 and batch[key].dtype != torch.int64: batch[key] = batch[key].to(dtype=torch.float16)
                # Flips around input_y
                if args.clean_label:
                    input_y = batch.pop("input_y_clean")
                    batch['input_y'] = input_y
                
                with accelerator.accumulate(model):
                    outputs = model(**batch, accelerator=accelerator)
                    loss = outputs.loss

                    # Stop when training loss goes NaN
                    if torch.isnan(loss):
                        print(f"Stopping training at Epoch {epoch}, Iteration {step} due to NaN loss")
                        optimizer.zero_grad()
                        break
                    # We keep track of the loss at each epoch
                    total_loss += loss.detach().float()
                    accelerator.backward(loss)

                    # Prevent Gradient Explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
    
                if (step+1) % 100 == 0 and accelerator.is_main_process:
                    logger.info(f"epoch {epoch}, step {step}: train_loss: {(total_loss.item() / (step + 1e-6)):.4f}")

                if isinstance(checkpointing_steps, int):
                    if (completed_steps + 1) % checkpointing_steps == 0:
                        break
                if completed_steps >= args.max_train_steps:
                    break
            total_loss = total_loss.item() / (step + 1e-6)
        
            
        model.eval()
        losses = []
        if args.n_class > 2: weight_sum = "macro" 
        else: weight_sum = "binary"
        
        for step, batch in enumerate(eval_dataloader):
            if args.deepspeed:
                for key in batch.keys():
                    if key == "input_x": continue
                    if batch[key].dtype != torch.float16 and batch[key].dtype != torch.int64: batch[key] = batch[key].to(dtype=torch.float16)
            with torch.no_grad():
                if args.clean_label:
                    input_y = batch.pop("input_y_clean")
                    batch["input_y"] = input_y
                else:
                    input_y = batch["input_y"]
                outputs = model(**batch, generate=True)    
                no_sample_pred = outputs.logits.argmax(dim=-1)
                no_sample_metric.add_batch(predictions=no_sample_pred.view(-1), references=input_y.argmax(dim=-1).view(-1), labels=list(range(args.n_class)), average=weight_sum)

            loss = outputs.loss
            if loss == 0:
                loss = torch.tensor(0.0, device=batch['input_x'].device)
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
            if step >= args.max_eval_steps:
                break
        no_sample_eval_output = no_sample_metric.compute()
        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: eval_acc: {no_sample_eval_output['accuracy']} train_loss: {(total_loss):.4f} eval_loss: {eval_loss:.4f}")

        if args.with_tracking:
            accelerator.log(
                {
                    "no_sample_eval": no_sample_eval_output,
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
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

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)


if __name__ == "__main__":
    main()