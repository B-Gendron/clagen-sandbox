from transformers import AutoTokenizer
import transformers
import torch
from utils import *

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

test_set = dailydialog['test']

# for each dialogue in test set
all_preds = []
for _ in range(10):
    print("Start inference")
    preds = []
    for i in range(len(test_set)):
        # for each utterance in the dialogue

        sequences = pipeline(
            format_prompt_last_utterance('test', i),
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=2000,
        )

        for seq in sequences:
            pred = map_emotion_to_index(predicted_emotion(seq['generated_text']))
        preds.append(pred)
    all_preds.append(preds)

all_trues = [[test_set[i]['emotion'][len(test_set[i]['emotion'])-1] for i in range(len(test_set))] for _ in range(10)]
results = compute_metrics_and_variance(all_trues, all_preds)
store_classification_metrics(results, model)
