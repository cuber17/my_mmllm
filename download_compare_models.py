import argparse
import os


MODEL_ZOO = {
    "phi35": {
        "repo_id": "microsoft/Phi-3.5-mini-instruct",
        "local_dir": "Phi-3.5-mini-instruct",
    },
    "qwen25_3b": {
        "repo_id": "Qwen/Qwen2.5-3B-Instruct",
        "local_dir": "Qwen2.5-3B-Instruct",
    },
    "llama31_8b": {
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "local_dir": "Llama-3.1-8B-Instruct",
    },
}


def download_one(model_key: str, output_root: str, snapshot_download_fn, hf_endpoint: str = "", hf_token: str = "") -> None:
    meta = MODEL_ZOO[model_key]
    save_dir = os.path.join(output_root, meta["local_dir"])
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n=== Downloading {model_key} ===")
    print(f"repo_id   : {meta['repo_id']}")
    print(f"save_dir  : {save_dir}")

    snapshot_download_fn(
        repo_id=meta["repo_id"],
        local_dir=save_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        endpoint=hf_endpoint or None,
        token=hf_token or None,
    )

    print(f"Done: {model_key}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download baseline LLMs for comparison experiments.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["phi35", "qwen25_3b", "llama31_8b"],
        choices=list(MODEL_ZOO.keys()),
        help="Model keys to download.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./huggingface",
        help="Local root dir for downloaded models.",
    )
    parser.add_argument(
        "--hf_endpoint",
        type=str,
        default="https://hf-mirror.com",
        help="Optional mirror endpoint, e.g. https://hf-mirror.com",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default="",
        help="Optional Hugging Face token for gated/private models.",
    )

    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        print(f"Using HF_ENDPOINT={args.hf_endpoint}")

    # Import after env setup to ensure huggingface_hub picks endpoint config correctly.
    from huggingface_hub import snapshot_download

    print("Model download plan:")
    for m in args.models:
        print(f"- {m}: {MODEL_ZOO[m]['repo_id']}")

    for model_key in args.models:
        try:
            download_one(
                model_key=model_key,
                output_root=args.output_root,
                snapshot_download_fn=snapshot_download,
                hf_endpoint=args.hf_endpoint,
                hf_token=args.hf_token,
            )
        except Exception as e:
            print(f"\nFailed to download {model_key}: {e}")
            print("Troubleshooting:")
            print("1) Test mirror reachability: curl -I https://hf-mirror.com")
            print("2) If mirror unavailable, retry with --hf_endpoint https://huggingface.co")
            print("3) For gated models, pass --hf_token <your_token>")
            print("4) If server has no outbound network, download on another machine then copy to ./huggingface/")
            raise

    print("\nAll requested downloads finished.")
    print("Note: Llama-3.1-8B-Instruct requires accepted Meta license token on Hugging Face.")


if __name__ == "__main__":
    main()
