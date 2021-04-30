import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

import numpy as np
import pandas as pd

import hedonometer.settings as st
from hedonometer.model import SentimentClassifier

# from metrics import measure_predicted_vs_targets, add_single_metric
MEMORY = pd.DataFrame(data={
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []},
)

CURRENT_EPOCH = 0


def _train_epoch(
        model: SentimentClassifier,
        data_loader: DataLoader,
        loss_fn,
        optimizer,
        device: str,
        scheduler
):
    """
    Trains the model over all of the data in the data_loader
    :param model: The model to be trained, must be a ClassClassifier
    :param data_loader: The data loader containing the data for the training
    :param loss_fn: The loss function to be used in gradient decent(CROSS_ENTROPY)
    :param optimizer: The optimizer to be used in the training process
    :param device: A string specifying the device to be used usually cuda:0 or cpu
    :param scheduler: The training scheduler to be used in the training process
    :returns:
    training_acc: Accuracy of the model over all entries in the data_loader
    training_loss: Mean training loss over all batches in the epoch
    """
    model = model.train()
    losses = []
    total_predictions = 0
    correct_predictions = 0
    number_of_batches = len(data_loader)

    for i, d in enumerate(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        one_hot= d["one_hot"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, predictions = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(predictions == targets)
        total_predictions += len(targets)
        loss = loss_fn(outputs.float(), one_hot.float())
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        print("{:.1f}%: loss:{:.7f}".format(100 * i / number_of_batches, np.mean(losses)), end='\r')
    return np.mean(losses), correct_predictions / total_predictions


def eval_model(
        model: SentimentClassifier,
        data_loader: DataLoader,
        loss_fn,
        device: str):
    """
    Evaluates
    :param loss_fn: The loss function to be used in the model evaluation(CROSS_ENTROPY)
    :param model: he model to be evaluated, must be a ClassClassifier
    :param data_loader: The data loader containing the data for the evaluation
    :param device: A string specifying the device to be used usually cuda:0 or cpu
    :return:
        validation_acc: Accuracy of the model over all entries in the data_loader
        validation_loss: The mean loss over all of the batches in the data_loader
    """
    model = model.eval()
    correct_predictions = 0
    total_predictions = 0
    losses = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            one_hot=d["one_hot"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, predictions = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(predictions == targets)
            total_predictions += len(targets)
            loss = loss_fn(outputs.float(), one_hot.float())
            losses.append(loss.item())

    return np.mean(losses), correct_predictions / total_predictions


# trains the model
def train_model(model, loss_fn, optimizer, train_data_loader, val_data_loader, epochs=10):
    """
    Trains the model and saves a checkpoint
    :param model: Model which is going to be trained
    :param loss_fn: Selected los function for evaluating training
    :param optimizer: Selected optimizer
    :param train_data_loader: Data for training
    :param val_data_loader: Data for validation
    :param epochs: Number of epochs
    """
    global MEMORY
    total_steps = len(train_data_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    best_accuracy = 0
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        train_loss, train_acc = _train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            st.DEVICE,
            scheduler,
        )

        val_loss, val_acc = eval_model(model, val_data_loader, loss_fn, st.DEVICE)
        MEMORY = MEMORY.append({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }, ignore_index=True)
        print(f'Val accuracy {val_acc}')
        print()
        if val_acc > best_accuracy:
            print("save...")
            torch.save(model.state_dict(), st.SAVE_PATH.format(model.name))
            best_accuracy = val_acc
    MEMORY.to_csv('hedonometer/data/memory.csv')
