import argparse
import json
import math
import os
from typing import Dict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from data_utils import load_lines, split_texts, build_tokenized_datasets


def eval_perplexity(
    model: AutoModelForCausalLM,
    dataset,
    data_collator,
    batch_size: int = 2,
    output_dir: str = "tmp_eval",
) -> Dict[str, float]:
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=batch_size,
        
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
    )
    metrics = trainer.evaluate(eval_dataset=dataset)
    loss = metrics["eval_loss"]
    metrics["eval_perplexity"] = float(math.exp(loss))
    return metrics


def train_on_domain(
    model: AutoModelForCausalLM,
    train_dataset,
    data_collator,
    output_dir: str,
    num_train_epochs: float = 1.0,
    batch_size: int = 2,
) -> AutoModelForCausalLM:
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
       
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    return model




def main():
    parser = argparse.ArgumentParser(
        description="R2 E1 - Preentrenamiento continuo A->B sin replay"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilgpt2",
        help="Nombre del modelo base en HuggingFace",
    )
    parser.add_argument(
        "--data_a",
        type=str,
        default="data/dominio_a.txt",
        help="Ruta al texto de dominio A (una muestra por línea)",
    )
    parser.add_argument(
        "--data_b",
        type=str,
        default="data/dominio_b.txt",
        help="Ruta al texto de dominio B (una muestra por línea)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_e1",
        help="Directorio donde se guardan checkpoints y métricas",
    )
    parser.add_argument(
        "--epochs_a",
        type=float,
        default=1.0,
        help="Número de épocas para Fase A",
    )
    parser.add_argument(
        "--epochs_b",
        type=float,
        default=1.0,
        help="Número de épocas para Fase B",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Longitud máxima de secuencia para el tokenizador",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size por dispositivo",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    
    model_base = AutoModelForCausalLM.from_pretrained(args.model_name)
    model_base.resize_token_embeddings(len(tokenizer))

   
    model_continual = AutoModelForCausalLM.from_pretrained(args.model_name)
    model_continual.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

   
    texts_a = load_lines(args.data_a)
    texts_b = load_lines(args.data_b)

    splits_a = split_texts(texts_a)
    splits_b = split_texts(texts_b)

    ds_a = build_tokenized_datasets(splits_a, tokenizer, max_length=args.max_length)
    ds_b = build_tokenized_datasets(splits_b, tokenizer, max_length=args.max_length)


    print("== Evaluando modelo base ==")
    metrics = {}

    metrics["base_A"] = eval_perplexity(
        model_base,
        ds_a["test"],
        data_collator=data_collator,
        batch_size=args.batch_size,
        output_dir=os.path.join(args.output_dir, "eval_base_A"),
    )

    metrics["base_B"] = eval_perplexity(
        model_base,
        ds_b["test"],
        data_collator=data_collator,
        batch_size=args.batch_size,
        output_dir=os.path.join(args.output_dir, "eval_base_B"),
    )

    
    print("== Fase A: entrenando en dominio A ==")
    model_continual = train_on_domain(
        model_continual,
        ds_a["train"],
        data_collator=data_collator,
        output_dir=os.path.join(args.output_dir, "fase_A"),
        num_train_epochs=args.epochs_a,
        batch_size=args.batch_size,
    )

    
    print("== Evaluando modelo después de Fase A ==")
    metrics["after_A_A"] = eval_perplexity(
        model_continual,
        ds_a["test"],
        data_collator=data_collator,
        batch_size=args.batch_size,
        output_dir=os.path.join(args.output_dir, "eval_after_A_A"),
    )

    metrics["after_A_B"] = eval_perplexity(
        model_continual,
        ds_b["test"],
        data_collator=data_collator,
        batch_size=args.batch_size,
        output_dir=os.path.join(args.output_dir, "eval_after_A_B"),
    )

   
    print("== Fase B: entrenando en dominio B (continual) ==")
    model_continual = train_on_domain(
        model_continual,
        ds_b["train"],
        data_collator=data_collator,
        output_dir=os.path.join(args.output_dir, "fase_B"),
        num_train_epochs=args.epochs_b,
        batch_size=args.batch_size,
    )

 
    print("== Evaluando modelo después de Fase B ==")
    metrics["after_B_A"] = eval_perplexity(
        model_continual,
        ds_a["test"],
        data_collator=data_collator,
        batch_size=args.batch_size,
        output_dir=os.path.join(args.output_dir, "eval_after_B_A"),
    )

    metrics["after_B_B"] = eval_perplexity(
        model_continual,
        ds_b["test"],
        data_collator=data_collator,
        batch_size=args.batch_size,
        output_dir=os.path.join(args.output_dir, "eval_after_B_B"),
    )

   
    metrics_path = os.path.join(args.output_dir, "metrics_e1.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Métricas guardadas en {metrics_path}")
    print("Resumen de métricas (perplejidad):")
    for key, m in metrics.items():
        print(
            f"{key}: ppl = {m['eval_perplexity']:.2f}, loss = {m['eval_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
