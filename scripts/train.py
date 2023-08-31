import os
import argparse
from transformers import (
    AutoModelForCausalLM,
    default_data_collator,
    AutoTokenizer,
    set_seed,
)
from datasets import load_from_disk
import torch

from huggingface_hub import HfFolder
from transformers import Trainer, TrainingArguments



def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument("--model_id", type=str, default="google/flan-t5-xl", help="Model id to use for training.")
    parser.add_argument("--train_dataset_path", type=str, help="Path to processed dataset stored by sageamker.")
    parser.add_argument(
        "--repository_id", type=str, default=None, help="Hugging Face Repository id for uploading models"
    )
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size to use for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size to use for testing.")
    parser.add_argument("--optimizer", type=str, default="adamw_hf", help="Learning rate to use for training.")

    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Whether to use gradient checkpointing.")
    parser.add_argument("--access_token",type=str,default=None)
    parser.add_argument("--max_steps", type=int, default=None, help="Number of epochs to train for.")

    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=HfFolder.get_token(),
        help="Token to use for uploading models to Hugging Face Hub.",
    )
    args = parser.parse_known_args()
    print(args)
    return args


def training_function(args):
    # set seed
    set_seed(args.seed)

    from huggingface_hub.hf_api import HfFolder;
    HfFolder.save_token(args.access_token)

    dataset = load_from_disk(args.train_dataset_path)

    # load dataset from disk and tokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # load model from the hub
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        #token=args.access_token,
        cache_dir="/opt/ml/sagemaker/warmpoolcache",
        use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )

    # Define compute metrics function


    # Define training args
    output_dir = os.environ["SM_OUTPUT_DATA_DIR"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="no",
        save_total_limit=2,
        optim=args.optimizer,
        max_steps=args.max_steps,
        # push to hub parameters
 
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=default_data_collator
    )

    # Start training
    trainer.train()

    # Save our tokenizer and create model card
    tokenizer.save_pretrained(output_dir)


    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works
    trainer.save_model(os.environ["SM_MODEL_DIR"])
    tokenizer.save_pretrained(os.environ["SM_MODEL_DIR"])

def main():
    args, _ = parse_arge()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"local rank {local_rank} global rank {rank} world size {world_size}")
    training_function(args)


if __name__ == "__main__":
    main()
