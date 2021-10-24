import pandas as pd
import transformers
import datasets
import shap
import pickle


dataset = datasets.load_dataset("emotion", split="train")
data = pd.DataFrame({"text": dataset["text"], "emotion": dataset["label"]})
# emotions: "sadness", "joy", "love", "anger", "fear", "surprise"
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "nateraw/bert-base-uncased-emotion", use_fast=True
)
model = transformers.AutoModelForSequenceClassification.from_pretrained(
    "nateraw/bert-base-uncased-emotion"
).cuda()
predictor = transformers.pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0,
    return_all_scores=True,
)
explainer = shap.Explainer(predictor)
obj = [predictor, explainer]
with open("score_objects.pkl", "wb") as handle:
    pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
