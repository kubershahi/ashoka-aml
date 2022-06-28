# importing necessary packages

# for manipulating dataset
import numpy as np
import pandas as pd

# for building the model
import torch
print("Torch Version:" , torch.__version__)
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Available Torc Device:", torch_device)

# importing and loading the model
from transformers import PegasusForConditionalGeneration, PegasusTokenizerFast, Trainer, TrainingArguments

# splittin dataset
from sklearn.model_selection import train_test_split

# evaluation metric
from ignite.metrics import Rouge, RougeN, RougeL

print("\nReading the Dataset...")
df_headline = pd.read_csv('./dataset/news_headline.csv', header=0)
print(df_headline.shape)

print("\nSplitting the Dataset...")
x_train, x_test, y_train, y_test = train_test_split(df_headline['text'], df_headline['summary'], test_size=0.2,random_state=25, shuffle=True)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train_list, y_train_list = x_train.tolist(), y_train.tolist()
x_test_list, y_test_list = x_test.tolist(), y_test.tolist()

print("Length of the Training and Test Set...")
print(len(x_train_list), len(y_train_list))
print(len(x_test_list), len(y_test_list))


tokenizer_large = PegasusTokenizerFast.from_pretrained("google/pegasus-large")
model_large = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large").to(torch_device)


# function to get summary of a text of list of texts
def get_summary(tokenizer, model, x):
    x_tokenized = tokenizer(x, truncation=True, padding = True, return_tensors="pt").to(torch_device)
    print("Input X tokenized. Generating Summary ...")
    y_pred_tokenized= model.generate(**x_tokenized).to(torch_device)
    print("Summary Generated. Decoding Summary ...")
    y_pred = tokenizer.batch_decode(y_pred_tokenized, skip_special_tokens=True)
    print("Summary Decoded.")
    return y_pred

def calculate_rouge(m, y_pred, y):
    
    candidate = [i.split() for i in y_pred ]
    reference = [i.split() for i in y]
    # print(candidate, reference)
    m.update((candidate, reference))
    
    return m.compute()

print("\nTesting the pretrained Model:")

m = Rouge(variants=["L", 1], multiref="best")
r = 0
for i in range(0, 250, 10):
    y_test_pred = get_summary(tokenizer_large,model_large, x_test_list[i:i+10])
    r = calculate_rouge(m, y_test_pred, y_test_list[i:i+10])

print("Rouge Score: ", r)


print("\nPrinting the predicted sumamry:\n")
print(y_test_pred[:10])

print("\nEnd of job")