#!/usr/bin/env python3
"""Report likely idle GPUs based on `nvidia-smi` memory and utilization."""

import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-memory-used", type=int, default=1024, help="Maximum memory usage in MiB for an idle GPU.")
    parser.add_argument("--max-utilization", type=int, default=10, help="Maximum utilization percentage for an idle GPU.")
    return parser.parse_args()


def query_gpus():
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)

    gpus = []
    for line in completed.stdout.strip().splitlines():
        index, memory_used, utilization = [field.strip() for field in line.split(",")]
        gpus.append(
            {
                "index": int(index),
                "memory_used": int(memory_used),
                "utilization": int(utilization),
            }
        )
    return gpus


def main():
    args = parse_args()
    gpus = query_gpus()
    idle_gpus = [
        gpu
        for gpu in gpus
        if gpu["memory_used"] <= args.max_memory_used and gpu["utilization"] <= args.max_utilization
    ]

    for gpu in gpus:
        print(
            f"GPU {gpu['index']}: memory_used={gpu['memory_used']} MiB, utilization={gpu['utilization']}%"
        )

    if idle_gpus:
        selected = ",".join(str(gpu["index"]) for gpu in idle_gpus)
        print("")
        print(f"Idle GPUs: {selected}")
        print(f"CUDA_VISIBLE_DEVICES={selected}")
    else:
        print("")
        print("No GPUs matched the current idle thresholds.")


if __name__ == "__main__":
    main()
