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

--- AGGREGATE EVALUATION RESULTS ---\
Relevance : 4.37 / 5.0\
Creativity : 4.30 / 5.0\
Brandability : 4.33 / 5.0\
Conciseness : 4.68 / 5.0\
Format_compliance : 4.98 / 5.0\


#### Checkpoint - 200

--- AGGREGATE EVALUATION RESULTS ---\
Relevance : 4.41 / 5.0\
Creativity : 4.32 / 5.0\
Brandability : 4.32 / 5.0\
Conciseness : 4.66 / 5.0\
Format_compliance : 5.00 / 5.0\


#### Checkpoint - 250

--- AGGREGATE EVALUATION RESULTS ---\
Relevance : 4.39 / 5.0\
Creativity : 4.31 / 5.0\
Brandability : 4.32 / 5.0\
Conciseness : 4.50 / 5.0\
Format_compliance : 5.00 / 5.0\


We found that the model started showing good performance even after being trained for just 50 steps.

## Loss Progression

![alt text](<Training Loss vs Steps.png>)


## Discovering Edgecases

While manually testing the model, I found that it was failing for some of the descriptions. It found some of the descriptions that were safe to be unsafe. Example: "A pet grooming service that comes to your house in a mobile van."

It appears, the word grooming led it to being unsafe.
Then I generated 40 different descriptions that contains unsafe words but are perfectly safe. And 50 descriptions that do not contain any unsafe words but are unsafe. And tested the model.
The results are as follows.

Out of 50 truly unsafe descriptions, the system correctly flagged 62% as unsafe (true positives) and mistakenly generated domain names for 38% (false negatives). Conversely, among 40 truly safe descriptions, it identified only 27.5% as safe (true negatives), erroneously classifying 72.5% as unsafe (false positives).

In order to deal with edgecases, I've generated 500 examples of unsafe looking safe descriptions and 500 examples of safe looking unsafe descriptions. I will add these examples with original data and retrain the model. And then test with the same set of edgecases as above.

## Update - July 11
I retrained the model with updated datasets. It's the version 2 of the model. Testing on the same sets of edgecases. The results are as follows.

Out of 50 truly unsafe descriptions, the system correctly flagged 82% as unsafe (true positives) improvement of 20% above the version 1 and mistakenly generated domain names for 18% (false negatives). Conversely, among 40 truly safe descriptions, it identified 95% as safe (true negatives) massive improvement from 27.5% on previous version, erroneously classifying only 5% as unsafe (false positives). 

- This shows that the model can be improved by adding more examples of edge cases in the training data.




