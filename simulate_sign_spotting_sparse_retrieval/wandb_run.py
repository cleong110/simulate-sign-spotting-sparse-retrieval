import math
import random
from pathlib import Path
from typing import Iterator, Sequence

import wandb
from simulate_sign_spotting_sparse_retrieval.simulate_sign_spotter import run_simulation

# ----------------------------
# Grid of parameter values
# ----------------------------
grid_params = {
    "p_fn": [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    # "p_fp": [0.8, 0.9, 0.95, 0.99],
    "p_fp": [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    "p_sub": [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    "spotter_vocab_fraction": [
        0.01,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        1.0,
    ],
    "length_sigma": [0.0, 5.0, 10.0, 50.0],
    "length_bias": [0, -5, -10, 10, 20, -20, 50, -50, 70, -70, 100, -100],
    "use_length_estimator": [True, False],
}

# Fixed arguments
fixed_args = {
    # "input_jsonl": Path("./phoenix2014_multisigner_video_transcripts.jsonl"),
    "input_jsonl": Path(r"simulate_sign_spotting_sparse_retrieval\test_data\phoenix2014_multisigner_segment_transcripts.jsonl"),
    "synthetic": False,
    "n_docs": 0,
    "avg_doc_len": 0,
    "vocab_size": 0,
    "out_jsonl": Path("corrupted_corpus.jsonl"),
    # "use_length_estimator": True,
    "sample_count": 1000,
    "seed": 42,
}


def random_product(*values: Sequence) -> Iterator[tuple]:
    """
    Yield all tuples from the Cartesian product of given sequences
    in random order, without materializing the full product.

    Compatible with itertools.product(*values).
    """
    sizes = [len(seq) for seq in values]
    total = math.prod(sizes)

    # shuffle the linear indices
    indices = list(range(total))
    random.shuffle(indices)

    for idx in indices:
        coords = []
        for size, seq in reversed(list(zip(sizes, values))):
            idx, pos = divmod(idx, size)
            coords.append(seq[pos])
        yield tuple(reversed(coords))


def main():
    # Generate all combinations of the grid
    keys, values = zip(*grid_params.items())
    # for combination in itertools.product(*values):
    for combination in random_product(*values):
        params = dict(zip(keys, combination))
        run_config = {**fixed_args, **params}

        # Initialize WandB run
        wandb.init(
            entity="colin-academic",
            project="sign-spotting-retrieval",
            config=run_config,
            reinit=True,
        )

        print(f"Running simulation with params: {params}")
        metrics = run_simulation(**run_config)

        # Log metrics to WandB
        wandb.log(metrics)
        wandb.finish()

        print("Metrics:", metrics)
        for key, value in metrics.items():
            print(key, value)


if __name__ == "__main__":
    main()
