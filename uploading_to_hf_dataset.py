import pandas as pd
import os
from datasets import Dataset, DatasetDict, Features, Image, Value

# Base directory for your data
base_path = "datasets/ilocr.iiit.ac.in/IIIT-INDIC-HW-WORDS-Hindi"


def create_split(split_name, gt_filename):
    # Construct paths
    gt_path = os.path.join(base_path, split_name, gt_filename)
    split_dir = os.path.join(base_path, split_name)

    # Load the text file (Tab separated)
    # The file contains: images/8/251/21.jpg <TAB> केंद्र
    df = pd.read_csv(gt_path, sep="\t", names=["image_path", "text"], header=None)

    # Prepend the split directory to the image_path so the script finds the files
    # Example: datasets/.../train/ + images/8/251/21.jpg
    df["image_path"] = df["image_path"].apply(lambda x: os.path.join(split_dir, x))

    features = Features(
        {
            "image": Image(),  # This triggers the image-text preview in HF UI
            "text": Value("string"),
        }
    )

    # Rename 'image_path' to 'image' for the final dataset schema
    dataset = Dataset.from_pandas(
        df.rename(columns={"image_path": "image"}), features=features
    )
    return dataset


# 2. Build the DatasetDict
print("Preparing splits...")
dataset_dict = DatasetDict(
    {
        "train": create_split("train", "train_gt.txt"),
        "test": create_split("test", "test_gt.txt"),
        "validation": create_split("val", "val_gt.txt"),
    }
)

# 3. Push to Hugging Face
# This will automatically convert your data to Parquet format
print("Uploading to Hugging Face...")
dataset_dict.push_to_hub("ianuragbhatt/IIIT-INDIC-HW-WORDS-Hindi")
