import argparse

import fasttext


def main():
    parser = argparse.ArgumentParser(description="Train a FastText word-vector model")
    parser.add_argument("--input", required=True, help="Input sentence file")
    parser.add_argument("--output", required=True, help="Output FastText model file")
    parser.add_argument("--dim", type=int, default=300, help="Vector dimensions")
    parser.add_argument("--window", type=int, default=5, help="Context window size")
    parser.add_argument("--min-count", type=int, default=5, help="Minimum word count")
    parser.add_argument("--minn", type=int, default=3, help="Minimum character n-gram length")
    parser.add_argument("--maxn", type=int, default=6, help="Maximum character n-gram length")
    parser.add_argument("--bucket", type=int, default=2_000_000, help="Character n-gram hash buckets")
    parser.add_argument("--neg", type=int, default=5, help="Negative samples")
    parser.add_argument("--lr", type=float, default=0.05, help="Initial learning rate")
    parser.add_argument("--sampling", type=float, default=1e-4, help="Frequent-word sampling threshold")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--threads", type=int, default=12, help="Training threads")
    args = parser.parse_args()

    model = fasttext.train_unsupervised(
        input=args.input,
        model="cbow",
        loss="ns",
        dim=args.dim,
        lr=args.lr,
        ws=args.window,
        minCount=args.min_count,
        minn=args.minn,
        maxn=args.maxn,
        bucket=args.bucket,
        neg=args.neg,
        t=args.sampling,
        epoch=args.epochs,
        thread=args.threads,
        verbose=2,
    )
    model.save_model(args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
