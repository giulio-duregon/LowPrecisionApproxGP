import ast
import pandas as pd
import os
from pathlib import Path

model_ids = set()
model_list = []


def main():
    experiment_folder = os.getenv(
        "EXPERIMENT_OUTPUTS", Path(os.getcwd() + "/Experiments")
    )

    with open(f"{experiment_folder}/Model_Index.log") as file:
        for line in file.readlines():
            # Seperate Message From TimeStamp
            line = line.split("-", maxsplit=3)[3]

            # First Segment is "model_ID": Unique Model ID
            segments = line.split(",", maxsplit=1)
            model_key, model_id = segments[0].split(":")
            model_key = model_key.strip()
            
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

    index_df = pd.json_normalize(model_list)

    training_list = []
    files = [
        file
        for file in os.listdir(experiment_folder)
        if file != "Model_Index.log" and ".log" in file
    ]
    for logs in files:
        with open(str(experiment_folder) + "/" + logs) as file:
            for line in file.readlines():
                arg = line.split("-", maxsplit=3)[3].strip()
                training_list.append(ast.literal_eval(arg))

    training_df = pd.json_normalize(training_list)
    
    # Save each
    training_df.to_csv("Training_data.csv")    
    index_df.to_csv("Model_Index.csv")
    
    # Set index, join, and save
    training_df.set_index("Model", inplace=True)
    index_df.set_index("Model_ID",inplace=True)
    index_df.join(training_df).to_csv("experiment_outputs.csv")


if __name__ == "__main__":
    main()
