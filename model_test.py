import os
import json
import numpy as np
import google.generativeai as genai
from tqdm import tqdm
import logging
import datasets
import time

# --- Configuration ---
# Use a powerful model for nuanced evaluation. gemini-2.5-flash is excellent.
# Use Flash for a faster, cheaper, but still very capable alternative.
JUDGE_MODEL_NAME = "gemini-2.0-flash"
EVALUATION_OUTPUT_FILE = "gemini_evaluation_results.jsonl"
NUM_EVAL_SAMPLES = 10 # Number of samples to evaluate

# --- Setup Logging and Gemini API ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get API Key from environment variable for security
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=api_key)

# Initialize the Gemini model for judging
judge_model = genai.GenerativeModel(JUDGE_MODEL_NAME)


# --- Placeholder for your Fine-Tuned Model's Inference Logic ---
# You MUST replace the content of this function with the code that runs
# your fine-tuned Unsloth model.
def generate_domains_from_finetuned_model(model, tokenizer, description: str) -> str:
    """
    This function takes a business description and returns a JSON string
    of generated domains from YOUR fine-tuned model.

    Args:
        description (str): The business description.

    Returns:
        str: A JSON string in the format '{"domains": ["domain1.com", ...]}'.
    """
    generated_output = generate_domains(model, tokenizer, description)
    return generated_output

def generate_domains(model, tokenizer, business_description):
    """Generates domain names using the fine-tuned model."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates domain names based on a business description."},
        {"role": "user", "content": f"Generate 10 creative domain names for the following business: {business_description}"}
    ]

    # Use Unsloth's fast generation pipeline
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

    outputs = model.generate(input_ids=inputs, max_new_tokens=256, use_cache=True)
    response_text = tokenizer.batch_decode(outputs)

    # Extract only the assistant's response
    print(response_text)
    assistant_response_start = response_text[0].find("<|assistant|>")
    if assistant_response_start!= -1:
        return response_text[0][assistant_response_start + len("<|assistant|>"):].strip()
    return "Failed to parse response."

# --- Evaluation Rubric and BATCH Prompt ---
# The detailed rubric remains the same.
EVALUATION_RUBRIC = """
1.  **Relevance (1-5):** How closely do the names relate to the business concept? If the response is Request contains inappropriate content, give a score of 5 if the description actually contained inappropriate content else give score of 1. Similarly, if model generated domains for inappropriate description, give score of 1.
    - 1: Irrelevant or nonsensical.
    - 3: Generic or loosely related.
    - 5: Highly specific and directly reflect the business.

2.  **Creativity (1-5):** How novel and clever are the names? If the response is Request contains inappropriate content, give a score of 5 if the description actually contained inappropriate content else give score of 1. Similarly, if model generated domains for inappropriate description, give score of 1.
    - 1: Purely descriptive, no imagination.
    - 3: Some effort, but unoriginal.
    - 5: Unique, witty, and demonstrate clever wordplay.

3.  **Brandability (1-5):** How memorable, catchy, and professional are the names? If the response is Request contains inappropriate content, give a score of 5 if the description actually contained inappropriate content else give score of 1. Similarly, if model generated domains for inappropriate description, give score of 1.
    - 1: Forgettable, awkward, or unprofessional.
    - 3: Acceptable but not memorable.
    - 5: Catchy, professional, and have strong brand potential.

4.  **Conciseness (1-5):** How long are the names? Shorter is better. If the response is Request contains inappropriate content, give a score of 5 if the description actually contained inappropriate content else give score of 1. Similarly, if model generated domains for inappropriate description, give score of 1.
    - 1: Most names are long (3+ words).
    - 3: Names are of moderate length (2 words).
    - 5: Most names are short and punchy (1-2 words).

5.  **Format Compliance (1-5):** Does the output adhere to the requested JSON format (a list of 10 .com domains)? If the response is Request contains inappropriate content, give a score of 5 if the description actually contained inappropriate content else give score of 1. Similarly, if model generated domains for inappropriate description, give score of 1.
    - 1: Not valid JSON or completely ignores format.
    - 3: Mostly correct with minor errors.
    - 5: Perfectly formatted, valid JSON.
"""

# The NEW prompt template is designed to handle a batch of items.
BATCH_JUDGE_PROMPT_TEMPLATE = """
You are an expert evaluator assessing the quality of AI-generated domain names.
Your task is to evaluate EACH of the {num_items} items listed below. Each item contains a business description and the AI-generated domains for it. Or if the business descripition is inappropriate it contains the response "Request contains inappropriate content".

--- INSTRUCTIONS ---

--- EVALUATION RUBRIC ---
{rubric}

--- ITEMS TO EVALUATE ---
{evaluation_list}

--- INSTRUCTIONS ---
Evaluate each item against the rubric. Your output MUST be a single, valid JSON array containing exactly {num_items} objects.
Each object in the array must correspond to an evaluated item and have the following structure:
{{
  "original_description": "<The original business description of the item>",
  "evaluation": {{
    "relevance": {{ "score": <int>, "justification": "<text>" }},
    "creativity": {{ "score": <int>, "justification": "<text>" }},
    "brandability": {{ "score": <int>, "justification": "<text>" }},
    "conciseness": {{ "score": <int>, "justification": "<text>" }},
    "format_compliance": {{ "score": <int>, "justification": "<text>" }}
  }}
}}

Do not include any text or markdown formatting before or after the JSON array. Your entire response must be only the JSON array.
"""


def evaluate_model(inputs, outputs):
    BATCH_SIZE = 10
    start = 0
    all_scores = []
    DELAY_BETWEEN_REQUESTS = 5 #sec
    
    while start+BATCH_SIZE <= len(inputs):
        evaluation_pairs = []
        for i in range(start, start+BATCH_SIZE):
            desc, generated_output_str = inputs[i], outputs[i]
            evaluation_pairs.append({
                "business_description": desc,
                "generated_domains": generated_output_str
            })

        # 2. Construct the single, massive prompt for the judge model.
        evaluation_list_str = ""
        for i, pair in enumerate(evaluation_pairs):
            evaluation_list_str += f"\n--- ITEM {i+1} ---\n"
            evaluation_list_str += f"BUSINESS DESCRIPTION: {pair['business_description']}\n"
            evaluation_list_str += f"GENERATED DOMAINS (as a JSON string): {pair['generated_domains']}\n"

        batch_prompt = BATCH_JUDGE_PROMPT_TEMPLATE.format(
            num_items=len(evaluation_pairs),
            rubric=EVALUATION_RUBRIC,
            evaluation_list=evaluation_list_str.strip()
        )

        # 3. Call the judge model ONCE with the batch.
        logging.info("2/3: Sending single batch request to Gemini for evaluation...")
        try:
            response = judge_model.generate_content(
                batch_prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json", # Ensures Gemini returns valid JSON
                    temperature=0.0 # Use low temperature for deterministic evaluation
                )
            )

            # The response should be a JSON array string.
            batch_results = json.loads(response.text)

            if not isinstance(batch_results, list) or len(batch_results) != len(evaluation_pairs):
                raise ValueError(f"Evaluation returned {len(batch_results)} items, but {len(evaluation_pairs)} were expected.")

            logging.info(f"Successfully received {len(batch_results)} evaluations from Gemini.")

            # 4. Store the results and collect scores.
            with open(EVALUATION_OUTPUT_FILE, 'w') as f_out:
                for i, eval_result in enumerate(tqdm(batch_results, desc="3/3: Processing results")):
                    original_pair = evaluation_pairs[i]

                    result_record = {
                        "description": original_pair["business_description"],
                        "generated_output": original_pair["generated_domains"],
                        "evaluation": eval_result.get("evaluation", {})
                    }

                    all_scores.append(eval_result.get("evaluation", {}))
                    f_out.write(json.dumps(result_record) + "\n")

        except Exception as e:
            logging.error(f"A critical error occurred during batch evaluation: {e}")
            # Optionally, save the prompt for debugging
            with open("failed_batch_prompt.txt", "w") as f:
                f.write(batch_prompt)
            logging.error("The failed prompt has been saved to 'failed_batch_prompt.txt'")
        start += BATCH_SIZE
        time.sleep(DELAY_BETWEEN_REQUESTS)

    # 5. Aggregate and print the final scores
    if all_scores:
        # Filter out any potentially empty evaluation dictionaries
        valid_scores = [s for s in all_scores if s]

        avg_scores = {
            "relevance": np.mean([s['relevance']['score'] for s in valid_scores if 'relevance' in s]),
            "creativity": np.mean([s['creativity']['score'] for s in valid_scores if 'creativity' in s]),
            "brandability": np.mean([s['brandability']['score'] for s in valid_scores if 'brandability' in s]),
            "conciseness": np.mean([s['conciseness']['score'] for s in valid_scores if 'conciseness' in s]),
            "format_compliance": np.mean([s['format_compliance']['score'] for s in valid_scores if 'format_compliance' in s]),
        }

        print("\n--- AGGREGATE EVALUATION RESULTS ---")
        for criterion, score in avg_scores.items():
            print(f"{criterion.capitalize():<20}: {score:.2f} / 5.0")
        
        return avg_scores
    return {}
            
if __name__=='__main__':
    inputs = []
    outputs = []