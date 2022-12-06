import ast
import pandas as pd

model_ids = set()
model_list = []


def main():
    with open("/Users/giulio/LowPrecisionApproxGP/Experiments/Model_Index.log") as file:
        for line in file.readlines():
            # Seperate Message From TimeStamp
            line = line.split("-", maxsplit=3)[3]

            # First Segment is "model_ID": Unique Model ID
            segments = line.split(",", maxsplit=1)
            model_key, model_id = segments[0].split(":")

            # Check if its in the set, this is necessary as outputs come in 2 lines
            # First line is initialization
            # Second line is time duration for training
            if model_id not in model_ids:
                # In this block if we're the first line
                model_ids.add(model_id)
                # Turn into dictionary
                temp = ast.literal_eval(segments[1])
                temp[model_key] = model_id
                model_list.append(temp)
            else:
                # In this block if we're the second line
                for value in model_list:
                    if value.get(model_key) == model_id:
                        rel_index = model_list.index(value)
                dict_to_update = model_list[rel_index]

                # Merge the two lines into a single dictionary
                dict_to_update.update(ast.literal_eval(segments[1].strip()))

    df = pd.json_normalize(model_list)


if __name__ == "__main__":
    main()
