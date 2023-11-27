import torch
from io import BytesIO
from sklearn.metrics import f1_score

def f1_score_(y_pred, y_true, threshold=0.5):
  y_pred = torch.sigmoid(y_pred)
  y_pred = (y_pred.detach().cpu().numpy() >= threshold).astype(int)
  y_true = y_true.detach().cpu().numpy()
  return f1_score(y_true, y_pred, average='micro')

def model_size(model):
  buffer = BytesIO()
  torch.save(model.state_dict(), buffer, _use_new_zipfile_serialization=True)
  size = buffer.tell() / (1024*1024)
  buffer.close()
  return size

def img_model_f1_score_(model, data_loader, threshold=0.5):
  model.eval()
  all_y_true = []
  all_y_pred = []

  with torch.no_grad():
    for x, y_true in data_loader:
      if torch.cuda.is_available():
        x = x.to('cuda')
      y_pred = model(x)
      y_pred = y_pred.to('cpu')
      y_true = y_true.to('cpu')
      all_y_pred.append(y_pred)
      all_y_true.append(y_true)
  all_y_pred = torch.cat(all_y_pred, dim=0)
  all_y_true = torch.cat(all_y_true, dim=0)
  return f1_score_(all_y_pred, all_y_true, threshold)

def txt_model_f1_score_(model, data_loader, threshold=0.5, device=torch.device('cuda')):
  model.eval()
  all_y_true = []
  all_y_pred = []

  with torch.no_grad():
    for batch in data_loader:
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['label'].to(device)

      output = model(input_ids, attention_mask=attention_mask)
      logits = output.logits
      all_y_pred.append(logits)
      all_y_true.append(labels)

  all_y_pred = torch.cat(all_y_pred, dim=0)
  all_y_true = torch.cat(all_y_true, dim=0)

  return f1_score_(all_y_pred.cpu(), all_y_true.cpu(), threshold)

def mul_model_f1_score_(model, data_loader, threshold=0.5, device=torch.device('cuda')):
  model.eval()
  all_y_true = []
  all_y_pred = []

  with torch.no_grad():
    for batch in data_loader:
      images, txt = batch
      images = images.to(device)
      input_ids = txt['input_ids'].to(device)
      attention_mask = txt['attention_mask'].to(device)
      labels = txt['label'].to(device)

      logits = model(images, input_ids, attention_mask)
      all_y_pred.append(logits)
      all_y_true.append(labels)

  all_y_pred = torch.cat(all_y_pred, dim=0)
  all_y_true = torch.cat(all_y_true, dim=0)

  return f1_score_(all_y_pred.cpu(), all_y_true.cpu(), threshold)