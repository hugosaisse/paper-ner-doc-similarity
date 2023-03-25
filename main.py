from text_preprocessor.text_preprocessor import TextPreprocessor
from classifier_ner.classifier_ner import Classifier
import numpy as np
import os
import pandas as pd
import torch
from torch import nn

# Preprocess and Tokenize
model_path = 'dominguesm/legal-bert-ner-base-cased-ptbr'
preprocessor = TextPreprocessor(model_path)
df = pd.read_csv('./data/data.csv')
preprocessed_df = preprocessor.preprocess_text(df, raw_text_column='docContSplitted')

# Get predictions (Named Entities)
classifier = Classifier(model_path)
classified_df = classifier.predict(preprocessed_df)


# List infractions evaluated by experts (for holdout sample)
df_expert1 = pd.read_excel('./data/similarity_experts_evaluation_i.xlsm', sheet_name='Similaridade')
holdout_violations = np.unique(df_expert1[['Inf A', 'Inf B']].values)

# Select infractions for fine-tuning the Word2Vec FastText model
df_train = preprocessed_df[~preprocessed_df['infracaoId'].isin(holdout_violations)]
df_train.reset_index(drop=True, inplace=True)

# Check that MPS (MacOS M1 Pro) is available, use GPU if available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
    device = torch.device("cpu")
else:
    device = torch.device("mps")

# Convert FastText model to PyTorch
pretrained_model_path = './models/cc.pt.300.bin'
ft_model = fasttext.load_model(pretrained_model_path)
pt_model = FastText(ft_model)
pt_model.to(device)

# Convert dataset to PyTorch format
train_dataset = LegalTextDataset(df_train)
train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn)

# Fine-tune the PyTorch model on the legal text dataset
optimizer = torch.optim.Adam(pt_model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
num_epochs = 10

for epoch in range(num_epochs):
    pt_model.train()
    for batch_idx, (texts, labels) in enumerate(train_loader):
        texts = texts.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = pt_model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item()}')

# Save the fine-tuned model
pt_model.save_state_dict('./models/legal_text_ft.pt')