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
    y_pred_tokenized= model.generate(**x_tokenized).to(torch_device)
    y_pred = tokenizer.batch_decode(y_pred_tokenized, skip_special_tokens=True)
    return y_pred

def calculate_rouge(m, y_pred, y):
    
    candidate = [i.split() for i in y_pred ]
    reference = [i.split() for i in y]
    # print(candidate, reference)
    m.update((candidate, reference))
    
    return m.compute()

# dataset class to efficiently manage our dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, text, summary):
        self.text= text
        self.summary = summary
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.text.items()}
        item['labels'] = torch.tensor(self.summary['input_ids'][idx])  # torch.tensor(self.summary[idx])
        return item
    def __len__(self):
        return len(self.text['input_ids'])

# function to prepare training data for model fine-tuning
def prepare_data(tokenizer, x_train, y_train, x_val=None, y_val=None):
    
    val = False if x_val is None or y_val is None else True
    def tokenize_data(text, summary):
        text_tokenized = tokenizer(text, truncation=True, padding=True)
        summary_tokenized = tokenizer(summary, truncation=True, padding=True)
        dataset_tokenized = Dataset(text_tokenized, summary_tokenized)
        return dataset_tokenized
    
    train_dataset = tokenize_data(x_train, y_train)
    val_dataset = tokenize_data (x_val, y_val) if val else None
    
    return train_dataset, val_dataset

# function to prepare and configure base model for fine-tuning
def prepare_finetuning(model, train_dataset, val_dataset=None, freeze_encoder=False, output_dir='./results'):

  if freeze_encoder:                                # if freeze_encoder is true
    for param in model.model.encoder.parameters():  # freeze the encode parameters
      param.requires_grad = False

  if val_dataset is not None:
    training_args = TrainingArguments(
      output_dir=output_dir,            # output directory
      adafactor=True,                   # use adafactor instead of AdamW
      num_train_epochs=10,              # total number of training epochs
      per_device_train_batch_size=20,    # batch size per device during training
      per_device_eval_batch_size=20,     # batch size for evaluation
      save_steps=500,                   # number of updates steps before checkpoint saves
      save_total_limit=5,               # limit the total amount of checkpoints and deletes the older checkpoints
      evaluation_strategy='steps',      # evaluation strategy to adopt during training
      eval_steps=500,                   # number of update steps before evaluation
      warmup_steps=500,                 # number of warmup steps for learning rate scheduler
      weight_decay=0.01,                # strength of weight decay
      logging_dir='./logs',             # directory for storing logs
      logging_steps=10,
    )

    trainer = Trainer(
      model=model,                         # the instantiated transformer model
      args=training_args,                  # training arguments as defined above
      train_dataset=train_dataset,         # training dataset
      eval_dataset=val_dataset             # evaluation dataset
    )

  else:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      adafactor=True,                  # use adafactor instead of AdamW
      num_train_epochs=10,             # total number of training epochs
      per_device_train_batch_size=20,  # batch size per device during training, can increase if memory allows
      save_steps=1000,                    # number of updates steps before checkpoint saves
      save_total_limit=5,              # limit the total amount of checkpoints and deletes the older checkpoints
      warmup_steps=200,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      logging_steps=10,
    )

    trainer = Trainer(
      model=model,                     # the instantiated transformer model
      args=training_args,              # training arguments as defined above
      train_dataset=train_dataset,     # training dataset
    )

  return trainer

print("\nTokening the dataset")
train_dataset,_ = prepare_data(tokenizer_large, x_train_list[:10000], y_train_list[:10000])

print("Length of the dataset:", len(train_dataset))

print("\nPreparing model_large for finetuning...")
trainer = prepare_finetuning(model_large, train_dataset) # compile the trainer model

print("\nTraining...")
trainer.train()

print("\nEnd of Training...")


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