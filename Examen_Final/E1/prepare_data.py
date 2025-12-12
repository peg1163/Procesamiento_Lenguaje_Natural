
from datasets import load_dataset
from pathlib import Path


def export_texts(
    dataset_name: str,
    split: str,
    text_field: str,
    out_path: str,
    load_kwargs: dict | None = None,
    max_samples: int | None = 100_000,
    min_chars: int = 50,
) -> None:

    load_kwargs = load_kwargs or {}

    print(f"Cargando dataset {dataset_name} (split={split})...")
    ds = load_dataset(dataset_name, split=split, **load_kwargs)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            text = ex.get(text_field, "")
            if text is None:
                continue
            text = str(text).strip()
            if len(text) < min_chars:
                continue

           
            text = text.replace("\n", " ")
            f.write(text + "\n")

            n += 1
            if max_samples is not None and n >= max_samples:
                break

    print(f"Guardadas {n} líneas en {out_path}")


if __name__ == "__main__":

    # Dominio A: español general
    # Usamos josecannete/large_spanish_corpus

  
    export_texts(
        dataset_name="parquet",
        split="train",
        text_field="text",
        out_path="data/dominio_a.txt",
        load_kwargs={
            "data_files": "hf://datasets/josecannete/large_spanish_corpus@refs/convert/parquet/all_wikis/train/*.parquet",
        },
        max_samples=100_000,  
        min_chars=50,
    )


    # Dominio B: textos legales
    # Dataset grande de derecho en español

    export_texts(
        dataset_name="Ramitha/spanish-legal-data-2",
        split="train",
        text_field="Data",     
        out_path="data/dominio_b.txt",
        load_kwargs={},
        max_samples=100_000,
        min_chars=50,
    )
