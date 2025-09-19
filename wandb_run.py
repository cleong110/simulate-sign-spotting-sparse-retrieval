import argparse
from pathlib import Path
import wandb
from simulate_sign_spotter import run_simulation

def main():
    parser = argparse.ArgumentParser(description="Run a single sign-spotting simulation with WandB logging")

    parser.add_argument("--input-jsonl", type=Path, default=None, help="Path to corpus JSONL")
    parser.add_argument("--synthetic", action="store_true", help="Generate a synthetic corpus instead of using input JSONL")
    parser.add_argument("--n-docs", type=int, default=100, help="Number of synthetic documents if synthetic=True")
    parser.add_argument("--avg-doc-len", type=int, default=50, help="Average length of synthetic documents")
    parser.add_argument("--vocab-size", type=int, default=500, help="Vocabulary size for synthetic documents")
    parser.add_argument("--out-jsonl", type=Path, default=Path("corrupted_corpus.jsonl"), help="Path to write corrupted corpus")
    parser.add_argument("--spotter-vocab-fraction", type=float, default=0.7)
    parser.add_argument("--p-fn", type=float, default=0.05)
    parser.add_argument("--p-fp", type=float, default=0.2)
    parser.add_argument("--p-sub", type=float, default=0.2)
    parser.add_argument("--use-length-estimator", action="store_true")
    parser.add_argument("--length-bias", type=int, default=-10)
    parser.add_argument("--length-sigma", type=float, default=10.0)
    parser.add_argument("--sample-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Initialize WandB
    wandb.init(
        entity="colin-academic",
        project="sign-spotting-retrieval",
        config=vars(args),
        name="single-run"
    )

    metrics = run_simulation(
        input_jsonl=args.input_jsonl,
        synthetic=args.synthetic,
        n_docs=args.n_docs,
        avg_doc_len=args.avg_doc_len,
        vocab_size=args.vocab_size,
        out_jsonl=args.out_jsonl,
        spotter_vocab_fraction=args.spotter_vocab_fraction,
        p_fn=args.p_fn,
        p_fp=args.p_fp,
        p_sub=args.p_sub,
        use_length_estimator=args.use_length_estimator,
        length_bias=args.length_bias,
        length_sigma=args.length_sigma,
        sample_count=args.sample_count,
        seed=args.seed,
    )

    # Log metrics to WandB
    wandb.log(metrics)

    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
