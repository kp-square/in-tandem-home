# --- Cell 3: Load and Prepare Dataset ---
import json
from datasets import load_dataset, concatenate_datasets
from sklearn.utils import shuffle

tokenizer = None
# Define a function to apply the chat template to each example
def format_chat_template(row):
    business_desc = row["business_description"]
    domain_list = row["domains"]

    assistant_response = json.dumps({"domains": domain_list})
    messages = [
        {"role": "user", "content": f"Generate 10 creative .com domain names for this business: {business_desc}"},
        {"role": "assistant", "content": assistant_response}
    ]
    # The tokenizer formats this list into the model-specific string
    row["text"] = messages# tokenizer.apply_chat_template(messages, tokenize=False)
    return row

def get_dataset():
    # Load the datasets from the JSONL files.
    # Ensure 'domain_gen_dataset.jsonl' and 'negative_domain_gen_dataset.jsonl' are uploaded to your Colab session.
    dataset_positive = load_dataset("json", data_files="domain_gen_dataset.jsonl", split="train")
    dataset_negative = load_dataset("json", data_files="negative_domain_gen_dataset.jsonl", split="train")

    # Combine the datasets
    dataset = concatenate_datasets([dataset_positive, dataset_negative])

    # Shuffle the combined dataset
    dataset = dataset.shuffle(seed=42)

    # Apply the formatting function to the entire dataset
    dataset = dataset.map(format_chat_template)

    split_dataset = dataset.train_test_split(test_size=100, seed=42)

    train_dataset, test_dataset = split_dataset["train"], split_dataset["test"]
    return train_dataset, test_dataset