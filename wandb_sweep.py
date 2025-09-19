import wandb
from pathlib import Path
from simulate_sign_spotter import run_simulation

# ----------------------------
# Sweep configuration
# ----------------------------
sweep_config = {
    "method": "grid",
    "metric": {"name": "ndcg_drop", "goal": "minimize"},
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
    wandb.init(entity="colin-academic", project="sign-spotting-retrieval")
    cfg = wandb.config

    input_jsonl = Path("./phoenix2014_multisigner_video_transcripts.jsonl")
    out_jsonl = Path(f"./corrupted_corpus_{wandb.run.id}.jsonl")

    metrics = run_simulation(
        input_jsonl=input_jsonl,
        synthetic=False,
        n_docs=0,
        avg_doc_len=0,
        vocab_size=0,
        out_jsonl=out_jsonl,
        spotter_vocab_fraction=float(cfg.spotter_vocab_fraction),
        p_fn=float(cfg.p_fn),
        p_fp=float(cfg.p_fp),
        p_sub=float(cfg.p_sub),
        use_length_estimator=True,
        length_bias=int(cfg.length_bias),
        length_sigma=float(cfg.length_sigma),
        sample_count=300,
        seed=42,
    )

    wandb.log(metrics)

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep_config,
        entity="colin-academic",
        project="sign-spotting-retrieval",
    )
    wandb.agent(sweep_id, function=sweep_train)
