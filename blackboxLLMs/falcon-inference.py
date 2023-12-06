from transformers import AutoTokenizer
import transformers
import torch
from utils import *
from transformers import AutoModelForCausalLM


model = "tiiuae/falcon-7b"
# model = "amazon/FalconLite"

# # run smaller version of Falcon
tokenizer = AutoTokenizer.from_pretrained(model)
# # pipeline = transformers.pipeline(
# #     "text-generation",
# #     model=model,
# #     tokenizer=tokenizer,
# #     torch_dtype=torch.bfloat16,
# #     trust_remote_code=True,
# #     load_in_8bit=True,
# #     device="cuda",
# # )

# run falcon 40b lite
pipe = transformers.pipeline(
    "text-generation", 
    model=model,
    tokenizer=tokenizer, 
    trust_remote_code=True)

test_set = dailydialog['test']

# for each dialogue in test set
all_preds = []
for _ in range(10):
    print("Start inference")
    preds = []
    for i in range(len(test_set)):
        # for each utterance in the dialogue

        sequences = pipe(
            format_prompt_last_utterance_falcon('test', i),
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=2000,
        )

        for seq in sequences:
            pred = map_emotion_to_index(predicted_emotion(seq['generated_text']))
        preds.append(pred)
        print(pred)
    all_preds.append(preds)

all_trues = [[test_set[i]['emotion'][len(test_set[i]['emotion'])-1] for i in range(len(test_set))] for _ in range(10)]
results = compute_metrics_and_variance(all_trues, all_preds)
store_classification_metrics(results, model)
