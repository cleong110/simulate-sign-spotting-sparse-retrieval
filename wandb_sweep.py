import wandb
from pathlib import Path
from simulate_sign_spotter import run_simulation

# ----------------------------
# Sweep configuration
# ----------------------------
sweep_config = {
    "method": "grid",  # or "random" / "bayes"
    "metric": {"name": "ndcg_drop", "goal": "minimize"},  # smaller drop is better
    "parameters": {
        "p_fn": {"values": [0.0, 0.05, 0.1]},
        "p_fp": {"values": [0.0, 0.1, 0.2]},
        "p_sub": {"values": [0.0, 0.1, 0.2]},
        "spotter_vocab_fraction": {"values": [0.5, 0.7, 1.0]},
        "length_sigma": {"values": [0.0, 5.0, 10.0]},
        "length_bias": {"values": [0, -5, -10]},
    },
}

# ----------------------------
# Sweep function
# ----------------------------
def sweep_train():
    # Initialize a run
    wandb.init(entity="colin-academic-org", project="sign-spotting-retrieval")
    cfg = wandb.config

    # Path to corpus JSONL
    input_jsonl = Path("./phoenix2014_multisigner_video_transcripts.jsonl")
    out_jsonl = Path(f"./corrupted_corpus_{wandb.run.id}.jsonl")

    # Run simulation
    metrics = run_simulation(
        input_jsonl=input_jsonl,
        synthetic=False,
        n_docs=0,
        avg_doc_len=0,
        vocab_size=0,
        out_jsonl=out_jsonl,
        spotter_vocab_fraction=cfg.spotter_vocab_fraction,
        p_fn=cfg.p_fn,
        p_fp=cfg.p_fp,
        p_sub=cfg.p_sub,
        use_length_estimator=True,
        length_bias=cfg.length_bias,
        length_sigma=cfg.length_sigma,
        sample_count=300,
        seed=42,
    )

    # Log metrics to WandB
    wandb.log(metrics)

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="sign_spotting_sim")
    wandb.agent(sweep_id, function=sweep_train)
