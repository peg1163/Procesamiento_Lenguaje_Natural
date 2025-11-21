import argparse
import subprocess
import sys
import shutil
from pathlib import Path



def run(cmd, **kwargs):
    
    print(f"\n>>> Ejecutando: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True, **kwargs)


def cmd_setup(args):
    
    run([sys.executable, "-m", "pip", "install", "-U", "pip"])
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def cmd_data(args):
   
    run([sys.executable, "-m", "src.data", "--prepare"])


def cmd_train(args):
    
    config = args.config or "configs/train.yaml"
    run([sys.executable, "-m", "src.train", "--config", config])


def cmd_eval(args):
    
    split = args.split or "val"
    run([sys.executable, "-m", "src.eval", "--split", split])


def cmd_decode(args):
    
    strategy = args.strategy
    beam_size = str(args.beam_size)
    length_penalty = str(args.length_penalty)

    run([
        sys.executable, "-m", "src.decoding",
        "--strategy", strategy,
        "--beam_size", beam_size,
        "--length_penalty", length_penalty,
    ])


def cmd_profile(args):
    
    config = args.config or "configs/train.yaml"
    run([
        sys.executable, "-m", "src.train",
        "--config", config,
        "--profile",
    ])


def cmd_test(args):
    
    run(["pytest", "-q"])


def cmd_clean(args):
    
    for folder in ["outputs", "checkpoints", ".pytest_cache"]:
        path = Path(folder)
        if path.exists():
            print(f"Eliminando {folder}/ ...")
            shutil.rmtree(path)
    print("Limpieza completada.")




def main():
    parser = argparse.ArgumentParser(
        description="CLI de proyecto PC3 (reemplazo de Makefile)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # setup
    p_setup = subparsers.add_parser("setup", help="Instalar dependencias.")
    p_setup.set_defaults(func=cmd_setup)

    # data
    p_data = subparsers.add_parser("data", help="Preparar datos.")
    p_data.set_defaults(func=cmd_data)

    # train
    p_train = subparsers.add_parser("train", help="Entrenar modelo.")
    p_train.add_argument(
        "--config", type=str, default=None,
        help="Ruta a archivo de configuración YAML (por defecto configs/train.yaml).",
    )
    p_train.set_defaults(func=cmd_train)

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluar modelo.")
    p_eval.add_argument(
        "--split", type=str, default="val",
        help="Split a evaluar: val, test, etc.",
    )
    p_eval.set_defaults(func=cmd_eval)

    # decode
    p_decode = subparsers.add_parser("decode", help="Generar texto.")
    p_decode.add_argument(
        "--strategy", type=str, default="beam",
        choices=["greedy", "beam", "topk", "topp"],
        help="Estrategia de decodificación.",
    )
    p_decode.add_argument(
        "--beam_size", type=int, default=4,
        help="Tamaño de beam (cuando uses beam search).",
    )
    p_decode.add_argument(
        "--length_penalty", type=float, default=0.7,
        help="Penalización de longitud para beam search.",
    )
    p_decode.set_defaults(func=cmd_decode)

    # profile
    p_profile = subparsers.add_parser(
        "profile", help="Entrenar con profiling activado."
    )
    p_profile.add_argument(
        "--config", type=str, default=None,
        help="Ruta a archivo de configuración YAML.",
    )
    p_profile.set_defaults(func=cmd_profile)

    # test
    p_test = subparsers.add_parser("test", help="Correr tests.")
    p_test.set_defaults(func=cmd_test)

    # clean
    p_clean = subparsers.add_parser("clean", help="Limpiar outputs/checkpoints.")
    p_clean.set_defaults(func=cmd_clean)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
