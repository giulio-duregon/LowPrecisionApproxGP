import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str)
    parser.add_argument("-o", type=str)
    return parser.parse_args()


def convert(i, o):
    df = pd.read_excel(i)
    df.to_csv(o)


if __name__ == "__main__":
    args = vars(parse_args())
    print(f"Converting the file {args['o']} to {args['o']}...")
    convert(**args)
    print("Done")
