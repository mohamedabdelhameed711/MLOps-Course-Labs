import argparse
from churn.train import train


def parse_args():
    parser = argparse.ArgumentParser("Bank-churn training CLI")
    parser.add_argument(
        "--model",
        choices=["rf", "gb", "et", "lr"],
        default="rf",
        help="Model type (rf, gb, et, lr)",
    )

    # common tree params
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=None)

    # gb-specific
    parser.add_argument("--learning_rate", type=float, default=0.05)

    # lr-specific
    parser.add_argument("--c", type=float, default=1.0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    kwargs = {}
    if args.model in {"rf", "et"}:
        kwargs.update(n_estimators=args.n_estimators, max_depth=args.max_depth)
    elif args.model == "gb":
        kwargs.update(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
        )
    elif args.model == "lr":
        kwargs["C"] = args.c

    train(model_name=args.model, model_params=kwargs)
