import torch
from tqdm import tqdm
import numpy as np
from cat.evaluator import f1_score_
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt

def img_clf_train(model, train_loader, valid_loader, optimizer, criterion, 
                  epochs=20, device=torch.device('cuda'), clip_value=1.0, early_stop_threshold=10):
  loss_list = []
  f1_list = []
  valid_loss_list = [np.inf] # Start with a high validation loss
  valid_f1_list = []
  early_stop_counter = 0
  model.train()

  for ep in tqdm(range(epochs)):
    ep_loss = 0.0
    ep_f1 = 0.0
    for step, (x, y) in enumerate(train_loader):
      x = x.float().to(device)
      y = y.float().to(device) 
      optimizer.zero_grad()
      p = model(x)
      loss = criterion(p, y)
      f1 = f1_score_(p.detach().cpu(), y.detach().cpu())
      ep_loss += loss.item()
      ep_f1 += f1
      loss.backward()
      clip_grad_norm_(model.parameters(), clip_value)
      optimizer.step()
    loss_list.append(ep_loss/(step+1))
    f1_list.append(ep_f1/(step+1))

    model.eval()
    valid_loss = 0.0
    for x, y in valid_loader:
      x = x.float().to(device)
      y = y.float().to(device)
      p = model(x)
      loss = criterion(p, y)
      f1 = f1_score_(p.detach().cpu(), y.detach().cpu())
      valid_loss += loss.item()
      valid_f1_list.append(f1)

    if valid_loss/len(valid_loader) < valid_loss_list[-1]:
      early_stop_counter = 0
    else:
      early_stop_counter += 1

    valid_loss_list.append(valid_loss/len(valid_loader))
    model.train()

    if early_stop_counter >= early_stop_threshold:
      print("\nModel training finished due to early stopping.")
      break

  fig, axs = plt.subplots(1, 2, figsize=(12, 4))
  axs[0].plot(loss_list, color='red', label='Training Loss')
  axs[0].set_xlabel('Epochs')
  axs[0].set_ylabel('Loss')
  axs[0].set_title('Loss during Training')
  axs[0].legend(loc='best')
  
  axs[1].plot(f1_list, color=(1.0, 0.5, 0.0), label='Training F1 Score')
  axs[1].set_xlabel('Epochs')
  axs[1].set_ylabel('F1 Score')
  axs[1].set_title('F1 Score during Training')
  axs[1].legend(loc='best')
  plt.show()

def txt_clf_train(model, train_loader, valid_loader, optimizer, criterion, scheduler, 
                  epochs=20, device=torch.device('cuda'), clip_value=1.0, early_stop_threshold=10):
  loss_list = []
  f1_list = []
  valid_loss_list = [np.inf] # Start with a high validation loss
  valid_f1_list = []
  early_stop_counter = 0
  model.train()

  for ep in tqdm(range(epochs)):
    ep_loss = 0.0
    ep_f1 = 0.0
    for step, batch in enumerate(train_loader):
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['label'].float().to(device)

      optimizer.zero_grad()
      output = model(input_ids, attention_mask=attention_mask)
      logits = output.logits
      loss = criterion(logits, labels)
      f1 = f1_score_(logits, labels.long())
      ep_loss += loss.item()
      ep_f1 += f1
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
      optimizer.step()
      scheduler.step()
    loss_list.append(ep_loss/(step+1))
    f1_list.append(ep_f1/(step+1))

    model.eval()
    valid_loss = 0.0
    valid_f1 = 0.0
    for batch in valid_loader:
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['label'].float().to(device)
      output = model(input_ids, attention_mask=attention_mask)
      logits = output.logits
      loss = criterion(logits, labels)
      f1 = f1_score_(logits, labels.long())
      valid_loss += loss.item()
      valid_f1 += f1

    valid_loss_list.append(valid_loss/len(valid_loader))
    valid_f1_list.append(valid_f1/len(valid_loader))
    model.train()

    if valid_loss_list[-1] >= valid_loss_list[-2]:
      early_stop_counter += 1
    else:
      early_stop_counter = 0

    if early_stop_counter >= early_stop_threshold:
      print("\nModel training finished due to early stopping.")
      break

  fig, axs = plt.subplots(1, 2, figsize=(12, 4))
  axs[0].plot(loss_list, color='red', label='Training Loss')
  axs[0].set_xlabel('Epochs')
  axs[0].set_ylabel('Loss')
  axs[0].set_title('Loss during Training')
  axs[0].legend(loc='best')
  
  axs[1].plot(f1_list, color=(1.0, 0.5, 0.0), label='Training F1 Score')
  axs[1].set_xlabel('Epochs')
  axs[1].set_ylabel('F1 Score')
  axs[1].set_title('F1 Score during Training')
  axs[1].legend(loc='best')
  plt.show()

def mul_clf_train(model, train_loader, valid_loader, optimizer, criterion, scheduler, 
                  epochs=20, device=torch.device('cuda'), clip_value=1.0, early_stop_threshold=10):
  loss_list = []
  f1_list = []
  valid_loss_list = [np.inf] # Start with a high validation loss
  valid_f1_list = []
  early_stop_counter = 0
  model.train()

  for ep in tqdm(range(epochs)):
    ep_loss = 0.0
    ep_f1 = 0.0
    for step, (img, txt) in enumerate(train_loader):
      img = img.float().to(device)
      input_ids = txt['input_ids'].to(device)
      attention_mask = txt['attention_mask'].to(device)
      labels = txt['label'].float().to(device)

      optimizer.zero_grad()
      logits = model(img, input_ids, attention_mask)
      loss = criterion(logits, labels)
      f1 = f1_score_(logits.detach().cpu(), labels.long())
      ep_loss += loss.item()
      ep_f1 += f1
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
      optimizer.step()
      scheduler.step()
    loss_list.append(ep_loss/(step+1))
    f1_list.append(ep_f1/(step+1))

    model.eval()
    valid_loss = 0.0
    valid_f1 = 0.0
    for img, txt in valid_loader:
      img = img.float().to(device)
      input_ids = txt['input_ids'].to(device)
      attention_mask = txt['attention_mask'].to(device)
      labels = txt['label'].float().to(device)
      logits = model(img, input_ids, attention_mask)
      loss = criterion(logits, labels)
      f1 = f1_score_(logits.detach().cpu(), labels.long())
      valid_loss += loss.item()
      valid_f1 += f1

    valid_loss_list.append(valid_loss/len(valid_loader))
    valid_f1_list.append(valid_f1/len(valid_loader))
    model.train()

    if valid_loss_list[-1] >= valid_loss_list[-2]:
      early_stop_counter += 1
    else:
      early_stop_counter = 0

    if early_stop_counter >= early_stop_threshold:
      print("\nModel training finished due to early stopping.")
      break

  fig, axs = plt.subplots(1, 2, figsize=(12, 4))
  axs[0].plot(loss_list, color='red', label='Training Loss')
  axs[0].set_xlabel('Epochs')
  axs[0].set_ylabel('Loss')
  axs[0].set_title('Loss during Training')
  axs[0].legend(loc='best')
  
  axs[1].plot(f1_list, color=(1.0, 0.5, 0.0), label='Training F1 Score')
  axs[1].set_xlabel('Epochs')
  axs[1].set_ylabel('F1 Score')
  axs[1].set_title('F1 Score during Training')
  axs[1].legend(loc='best')
  plt.show()