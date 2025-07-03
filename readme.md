# Documentation for URL Generating Model

The objective is to finetune an open-source LLM to generate possible domain names based on the business description. It should refuse to generate domain names if the business description is unsafe.

## Technologies Used

- **Open Source Model:** [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)  
  3.8 billion parameters, 4k context length.  
  Selected for its small size and good performance.

- **Unsloth** is used for fine-tuning the model.  
  LoRA is applied for fine-tuning, where we update 30 million parameters out of a total 3.8 billion.

## Dataset Generation

- **Faker** is used to generate business descriptions.
- **Gemini Flash 2.5** is used to generate potential domain names for 1,000 different business descriptions.  
- For unsafe businesses, the same model (Gemini Flash 2.5) is used to generate business descriptions.  
  1,000 unsafe business descriptions were generated.

*Note:* A Gemini 2.5 API key can be obtained free of cost via Google AI Studio.

The data was combined and shuffled. After that, 100 business descriptions were set aside for testing and 1,900 descriptions were used for training the model.

## Training & Evaluation

- **Batch size:** 8 prompts  
- After every 50 steps, the model was evaluated.  
- For evaluation, Gemini Flash 2.0 was used. Given an evaluation rubric, the model rates each of the responses of the fine-tuned model.

### Checkpoint Results

#### Checkpoint - 150

--- AGGREGATE EVALUATION RESULTS ---
Relevance : 4.37 / 5.0
Creativity : 4.30 / 5.0
Brandability : 4.33 / 5.0
Conciseness : 4.68 / 5.0
Format_compliance : 4.98 / 5.0


#### Checkpoint - 200

Relevance : 4.41 / 5.0
Creativity : 4.32 / 5.0
Brandability : 4.32 / 5.0
Conciseness : 4.66 / 5.0
Format_compliance : 5.00 / 5.0


#### Checkpoint - 250

--- AGGREGATE EVALUATION RESULTS ---
Relevance : 4.39 / 5.0
Creativity : 4.31 / 5.0
Brandability : 4.32 / 5.0
Conciseness : 4.50 / 5.0
Format_compliance : 5.00 / 5.0


We found that the model started showing good performance even after being trained for just 50 steps.

## Loss Progression

![alt text](<Training Loss vs Steps.png>)