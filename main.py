# import libraries
import yaml
from glob import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from tqdm import tqdm
from data_prepare import CustomDataset
from data_labels import label_decoder
from model import CNN2RNN, accuracy_function
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# load config
with open('config.yml', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
device = config['device']

# train step
def train_step(batch_item, training):
    img = batch_item['img'].to(device)
    csv_feature = batch_item['csv_feature'].to(device)
    label = batch_item['label'].to(device)
    if training is True:
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(img, csv_feature)
            loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        score = accuracy_function(label, output)
        return loss, score
    else:
        model.eval()
        with torch.no_grad():
            output = model(img, csv_feature)
            loss = criterion(output, label)
        score = accuracy_function(label, output)
        return loss, score

# whole train process
def train():
    loss_plot, val_loss_plot = [], []
    metric_plot, val_metric_plot = [], []

    for epoch in range(config['epochs']):
        total_loss, total_val_loss = 0, 0
        total_acc, total_val_acc = 0, 0
        
        tqdm_dataset = tqdm(enumerate(train_dataloader))
        training = True
        for batch, batch_item in tqdm_dataset:
            batch_loss, batch_acc = train_step(batch_item, training)
            total_loss += batch_loss
            total_acc += batch_acc
            
            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Loss': '{:06f}'.format(batch_loss.item()),
                'Mean Loss' : '{:06f}'.format(total_loss/(batch+1)),
                'Mean F-1' : '{:06f}'.format(total_acc/(batch+1))
            })
        loss_plot.append(total_loss/(batch+1))
        metric_plot.append(total_acc/(batch+1))
        
        tqdm_dataset = tqdm(enumerate(val_dataloader))
        training = False
        for batch, batch_item in tqdm_dataset:
            batch_loss, batch_acc = train_step(batch_item, training)
            total_val_loss += batch_loss
            total_val_acc += batch_acc
            
            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Val Loss': '{:06f}'.format(batch_loss.item()),
                'Mean Val Loss' : '{:06f}'.format(total_val_loss/(batch+1)),
                'Mean Val F-1' : '{:06f}'.format(total_val_acc/(batch+1))
            })
        val_loss_plot.append(total_val_loss/(batch+1))
        val_metric_plot.append(total_val_acc/(batch+1))
        
        if np.max(val_metric_plot) == val_metric_plot[-1]:
            torch.save(model.state_dict(), config['save_path'])

# prediction process
def predict(dataset):
    model.eval()
    tqdm_dataset = tqdm(enumerate(dataset))
    results = []
    for batch, batch_item in tqdm_dataset:
        img = batch_item['img'].to(device)
        seq = batch_item['csv_feature'].to(device)
        with torch.no_grad():
            output = model(img, seq)
        output = torch.tensor(torch.argmax(output, dim=1), dtype=torch.int32).cpu().numpy()
        results.extend(output)
    return results


def main():
    # load dataset
    train = sorted(glob('data/train/*'))
    test = sorted(glob('data/test/*'))
    labelsss = pd.read_csv('data/train.csv')['label']

    # split data for cross validation
    train, val = train_test_split(train, test_size=0.2, stratify=labelsss)

    # make CustomDataset & DataLoader
    train_dataset = CustomDataset(train)
    val_dataset = CustomDataset(val)
    test_dataset = CustomDataset(test, mode = 'test')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=16, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=16, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=16, shuffle=False)

    # define multimodal model
    model = CNN2RNN(max_len=config['max_len'], embedding_dim=config['embedding_dim'], num_features=config['num_features'], 
                    class_n=config['class_n'], rate=config['dropout_rate'])
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # train session
    train()

    # load best model
    model = CNN2RNN(max_len=config['max_len'], embedding_dim=config['embedding_dim'], num_features=config['num_features'], 
                    class_n=config['class_n'], rate=config['dropout_rate'])
    model.load_state_dict(torch.load(config['save_path'], map_location=device))
    model.to(device)

    # predict session
    preds = predict(test_dataloader)
    preds = np.array([label_decoder[int(val)] for val in preds])

    # make submission file for competition
    submission = pd.read_csv('data/sample_submission.csv')
    submission['label'] = preds
    submission.to_csv('baseline_submission.csv', index=False)


if __name__ == "__main__":
	main()