"""Build interaction C++/CUDA extensions ahead of time.

Examples:

    python -m maskvar.interaction.build_extensions --backend cpu
    TORCH_CUDA_ARCH_LIST=8.0 python -m maskvar.interaction.build_extensions --backend cuda
    python -m maskvar.interaction.build_extensions --backend all --cuda_arch_list 8.0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--backend", default="auto", choices=["auto", "cpu", "cuda", "all"])
    parser.add_argument(
        "--cuda_arch_list",
        default=None,
        help="Optional TORCH_CUDA_ARCH_LIST value, e.g. '8.0' or '8.0;8.6'.",
    )
    args, setup_args = parser.parse_known_args()
    return args, setup_args


def build_extensions(backend: str):
    root = Path(__file__).resolve().parent
    csrc = root / "csrc"
    extensions = []

    if backend in {"cpu", "all"}:
        extensions.append(
            CppExtension(
                name="maskvar.interaction._cpu_ext",
                sources=[str(csrc / "cpu_distance.cpp")],
                extra_compile_args={"cxx": ["-O3"]},
            )
        )

    if backend in {"cuda", "all"}:
        extensions.append(
            CUDAExtension(
                name="maskvar.interaction._cuda_ext",
                sources=[
                    str(csrc / "cuda_distance.cpp"),
                    str(csrc / "cuda_distance_kernel.cu"),
                ],
                extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
            )
        )

    if not extensions:
        raise SystemExit("No extensions selected.")
    return extensions


def main():
    args, setup_args = parse_args()
    if args.cuda_arch_list:
        os.environ["TORCH_CUDA_ARCH_LIST"] = args.cuda_arch_list

    backend = args.backend
    if backend == "auto":
        backend = "cuda" if torch.cuda.is_available() else "cpu"

    if backend == "cuda" and not torch.cuda.is_available():
        print("Building CUDA extension without a visible GPU; set TORCH_CUDA_ARCH_LIST explicitly.", file=sys.stderr)

    sys.argv = [sys.argv[0], *(setup_args or ["build_ext", "--inplace"])]
    setup(
        name="maskvar_interaction_extensions",
        ext_modules=build_extensions(backend),
        cmdclass={"build_ext": BuildExtension},
        script_args=sys.argv[1:],
    )


if __name__ == "__main__":
    main()
