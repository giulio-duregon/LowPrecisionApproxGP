import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="base", type=str)
    parser.add_argument("--training_iter", default=50, type=int)
    parser.add_argument("--dtype", default="float64", type=str)
    return parser.parse_args()


def main():
    ...


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
