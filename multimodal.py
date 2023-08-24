from attentions import SelfAttention, CrossAttention
import torch
from torch import nn
from torchvision import models
from transformers import BertModel

class DensityBert(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    # Define the image feature extractor
    self.img_featrs = models.densenet121(pretrained=True)
    self.img_featrs.classifier = nn.Identity()
    # Freeze the image feature extractor
    for param in self.img_featrs.parameters():
      param.requires_grad = False
    # Define the text feature extractor
    self.txt_featrs = BertModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    # Freeze the text feature extractor
    for param in self.txt_featrs.parameters():
      param.requires_grad = False
    # Define the attention layer
    self.image_attention = SelfAttention(self.img_featrs.features.norm5.num_features)
    self.text_attention = SelfAttention(self.txt_featrs.config.hidden_size)
    # Define the fully-connected layers
    self.img_featrs_fc = nn.Linear(self.img_featrs.features.norm5.num_features, 512)
    self.text_features_fc = nn.Linear(self.txt_featrs.config.hidden_size, 512)
    self.classifier = nn.Linear(512 * 2, num_classes)
    
  def forward(self, images, input_ids, attention_mask):
    image_features = self.img_featrs(images)
    text_features = self.txt_featrs(input_ids=input_ids, attention_mask=attention_mask)[0]
    
    image_attention_outputs = self.image_attention(image_features.view(image_features.size(0), -1, self.img_featrs.features.norm5.num_features))
    text_attention_outputs = self.text_attention(text_features)
    # Average pooling
    image_attention_weights = image_attention_outputs.mean(dim=1)
    text_attention_weights = text_attention_outputs.mean(dim=1)
    
    image_features = self.img_featrs_fc(image_attention_weights)
    text_features = self.text_features_fc(text_attention_weights)
    
    combined_features = torch.cat([image_features, text_features], dim=1)
    logits = self.classifier(combined_features)
    return logits

class MoDensityBert(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    # Define the image feature extractor
    self.img_featrs = models.densenet121(pretrained=True)
    self.img_featrs.classifier = nn.Identity()
    # Freeze the image feature extractor
    for param in self.img_featrs.parameters():
      param.requires_grad = False
    # Define the text feature extractor
    self.txt_featrs = BertModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    # Freeze the text feature extractor
    for param in self.txt_featrs.parameters():
      param.requires_grad = False
    # Define the attention layer
    self.image_attention = SelfAttention(self.img_featrs.features.norm5.num_features)
    self.text_attention = SelfAttention(self.txt_featrs.config.hidden_size)
    # Define the fully-connected layers
    self.img_featrs_fc = nn.Linear(self.img_featrs.features.norm5.num_features, 512)
    self.text_features_fc = nn.Linear(self.txt_featrs.config.hidden_size, 512)
    self.classifier = nn.Linear(512 * 2, num_classes)
    
  def forward(self, images, input_ids, attention_mask):
    image_features = self.img_featrs(images)
    text_features = self.txt_featrs(input_ids=input_ids, attention_mask=attention_mask)[0]
    
    image_attention_outputs = self.image_attention(image_features.view(image_features.size(0), -1, self.img_featrs.features.norm5.num_features))
    text_attention_outputs = self.text_attention(text_features)
    # Average pooling
    image_attention_weights = image_attention_outputs.mean(dim=1)
    text_attention_weights = text_attention_outputs.mean(dim=1)
    
    image_features = self.img_featrs_fc(image_attention_weights)
    text_features = self.text_features_fc(text_attention_weights)
    
    combined_features = torch.cat([image_features, text_features], dim=1)
    logits = self.classifier(combined_features)
    return logits
  
class WarmDBert(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    # Define the image feature extractor
    self.img_featrs = models.densenet121(pretrained=True)
    self.img_featrs.classifier = nn.Identity()
    # Freeze part layers of the image feature extractor
    for name, param in self.img_featrs.named_parameters():
      if 'denseblock4' not in name:
        param.requires_grad = False
    # Define the text feature extractor
    self.txt_featrs = BertModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    # Freeze part layers of the text feature extractor
    for name, param in self.txt_featrs.named_parameters():
      if 'encoder.layer.3' not in name:
        param.requires_grad = False
    # Define the attention layer
    self.image_attention = SelfAttention(self.img_featrs.features.norm5.num_features)
    self.text_attention = SelfAttention(self.txt_featrs.config.hidden_size)
    # Define the fully-connected layers
    self.img_featrs_fc = nn.Linear(self.img_featrs.features.norm5.num_features, 512)
    self.text_features_fc = nn.Linear(self.txt_featrs.config.hidden_size, 512)
    self.classifier = nn.Linear(512 * 2, num_classes)
    
  def forward(self, images, input_ids, attention_mask):
    image_features = self.img_featrs(images)
    text_features = self.txt_featrs(input_ids=input_ids, attention_mask=attention_mask)[0]
    
    image_attention_outputs = self.image_attention(image_features.view(image_features.size(0), -1, self.img_featrs.features.norm5.num_features))
    text_attention_outputs = self.text_attention(text_features)
    # Average pooling
    image_attention_weights = image_attention_outputs.mean(dim=1)
    text_attention_weights = text_attention_outputs.mean(dim=1)
    
    image_features = self.img_featrs_fc(image_attention_weights)
    text_features = self.text_features_fc(text_attention_weights)
    
    combined_features = torch.cat([image_features, text_features], dim=1)
    logits = self.classifier(combined_features)
    return logits
  
class WarmerDBert(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    # Define the image feature extractor
    self.img_featrs = models.densenet121(pretrained=True)
    self.img_featrs.classifier = nn.Identity()
    # Freeze part layers of the image feature extractor
    for name, param in self.img_featrs.named_parameters():
      if 'denseblock3' not in name:
        param.requires_grad = False
    # Define the text feature extractor
    self.txt_featrs = BertModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    # Freeze part layers of the text feature extractor
    for name, param in self.txt_featrs.named_parameters():
      if 'encoder.layer.2' not in name:
        param.requires_grad = False
    # Define the attention layer
    self.image_attention = SelfAttention(self.img_featrs.features.norm5.num_features)
    self.text_attention = SelfAttention(self.txt_featrs.config.hidden_size)
    # Define the fully-connected layers
    self.img_featrs_fc = nn.Linear(self.img_featrs.features.norm5.num_features, 512)
    self.text_features_fc = nn.Linear(self.txt_featrs.config.hidden_size, 512)
    self.classifier = nn.Linear(512 * 2, num_classes)
    
  def forward(self, images, input_ids, attention_mask):
    image_features = self.img_featrs(images)
    text_features = self.txt_featrs(input_ids=input_ids, attention_mask=attention_mask)[0]
    
    image_attention_outputs = self.image_attention(image_features.view(image_features.size(0), -1, self.img_featrs.features.norm5.num_features))
    text_attention_outputs = self.text_attention(text_features)
    # Average pooling
    image_attention_weights = image_attention_outputs.mean(dim=1)
    text_attention_weights = text_attention_outputs.mean(dim=1)
    
    image_features = self.img_featrs_fc(image_attention_weights)
    text_features = self.text_features_fc(text_attention_weights)
    
    combined_features = torch.cat([image_features, text_features], dim=1)
    logits = self.classifier(combined_features)
    return logits

class WarmerDBert_(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    # Define the image feature extractor
    self.img_featrs = models.densenet121(pretrained=True)
    self.img_featrs.classifier = nn.Identity()
    # Freeze part layers of the image feature extractor
    img_flag = True
    for name, param in self.img_featrs.named_parameters():
      if 'denseblock3' in name:
        img_flag = False
      param.requires_grad = not img_flag
    # Define the text feature extractor
    self.txt_featrs = BertModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    # Freeze part layers of the text feature extractor
    txt_flag = True
    for name, param in self.txt_featrs.named_parameters():
      if 'encoder.layer.2' in name:
        txt_flag = False
      param.requires_grad = not txt_flag
    # Define the attention layer
    self.image_attention = SelfAttention(self.img_featrs.features.norm5.num_features)
    self.text_attention = SelfAttention(self.txt_featrs.config.hidden_size)
    # Define the fully-connected layers
    self.img_featrs_fc = nn.Linear(self.img_featrs.features.norm5.num_features, 512)
    self.text_features_fc = nn.Linear(self.txt_featrs.config.hidden_size, 512)
    self.classifier = nn.Linear(512 * 2, num_classes)
    
  def forward(self, images, input_ids, attention_mask):
    image_features = self.img_featrs(images)
    text_features = self.txt_featrs(input_ids=input_ids, attention_mask=attention_mask)[0]
    
    image_attention_outputs = self.image_attention(image_features.view(image_features.size(0), -1, self.img_featrs.features.norm5.num_features))
    text_attention_outputs = self.text_attention(text_features)
    # Average pooling
    image_attention_weights = image_attention_outputs.mean(dim=1)
    text_attention_weights = text_attention_outputs.mean(dim=1)
    
    image_features = self.img_featrs_fc(image_attention_weights)
    text_features = self.text_features_fc(text_attention_weights)
    
    combined_features = torch.cat([image_features, text_features], dim=1)
    logits = self.classifier(combined_features)
    return logits
  
class WWDBert(nn.Module):
  def __init__(self, num_classes, dropout_rate=0.5):
    super().__init__()
    # Define the image feature extractor
    self.img_featrs = models.densenet121(pretrained=True)
    self.img_featrs.classifier = nn.Identity()
    # Freeze part layers of the image feature extractor
    for name, param in self.img_featrs.named_parameters():
      if 'denseblock3' not in name:
        param.requires_grad = False
    # Define the text feature extractor
    self.txt_featrs = BertModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    # Freeze part layers of the text feature extractor
    for name, param in self.txt_featrs.named_parameters():
      if 'encoder.layer.2' not in name:
        param.requires_grad = False
    # Define the attention layer
    self.image_attention = SelfAttention(self.img_featrs.features.norm5.num_features)
    self.text_attention = SelfAttention(self.txt_featrs.config.hidden_size)
    # Define the fully-connected layers
    self.img_featrs_fc = nn.Linear(self.img_featrs.features.norm5.num_features, 896)
    self.text_features_fc = nn.Linear(self.txt_featrs.config.hidden_size, 896)
    # Add Batch Normalization layers
    self.bn1 = nn.BatchNorm1d(896)
    self.bn2 = nn.BatchNorm1d(896)
    # Add Dropout layers
    self.dropout1 = nn.Dropout(dropout_rate)
    self.dropout2 = nn.Dropout(dropout_rate)
    self.dropout3 = nn.Dropout(dropout_rate)
    self.classifier = nn.Linear(896 * 2, num_classes)
    
  def forward(self, images, input_ids, attention_mask):
    image_features = self.img_featrs(images)
    text_features = self.txt_featrs(input_ids=input_ids, attention_mask=attention_mask)[0]
    
    image_attention_outputs = self.image_attention(image_features.view(image_features.size(0), -1, self.img_featrs.features.norm5.num_features))
    text_attention_outputs = self.text_attention(text_features)
    # Average pooling
    image_attention_weights = image_attention_outputs.mean(dim=1)
    text_attention_weights = text_attention_outputs.mean(dim=1)
    
    image_features = self.img_featrs_fc(image_attention_weights)
    text_features = self.text_features_fc(text_attention_weights)
    # Batch normalisation
    image_features = self.bn1(image_features)
    text_features = self.bn2(text_features)
    # Add activation function
    image_features = torch.relu(image_features)
    text_features = torch.relu(text_features)
    # Add Dropout layers
    image_features = self.dropout1(image_features)
    text_features = self.dropout2(text_features)
    combined_features = torch.cat([image_features, text_features], dim=1)
    logits = self.classifier(combined_features)
    return logits

class CDBert(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    # Define the image feature extractor
    self.img_featrs = models.densenet121(pretrained=True)
    self.img_featrs.classifier = nn.Identity()
    # Freeze the image feature extractor
    for param in self.img_featrs.parameters():
        param.requires_grad = False
    # Define the text feature extractor
    self.txt_featrs = BertModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    # Freeze the text feature extractor
    for param in self.txt_featrs.parameters():
        param.requires_grad = False
    # Define the attention layer
    self.attention = CrossAttention(self.txt_featrs.config.hidden_size, self.img_featrs.features.norm5.num_features)
    # Define the fully-connected layers
    self.img_featrs_fc = nn.Linear(self.img_featrs.features.norm5.num_features, 512)
    self.text_features_fc = nn.Linear(self.txt_featrs.config.hidden_size, 512)
    self.classifier = nn.Linear(512 * 2, num_classes)
    
  def forward(self, images, input_ids, attention_mask):
    image_features = self.img_featrs(images)
    text_features = self.txt_featrs(input_ids=input_ids, attention_mask=attention_mask)[0]
    image_features = image_features.view(image_features.size(0), -1, self.img_featrs.features.norm5.num_features)
    
    attention_outputs = self.attention(text_features, image_features)
    # Average pooling
    attention_weights = attention_outputs.mean(dim=1)
    
    image_features = self.img_featrs_fc(attention_weights)
    text_features = self.text_features_fc(text_features.mean(dim=1))
    
    combined_features = torch.cat([image_features, text_features], dim=1)
    logits = self.classifier(combined_features)
    return logits

class ImCDBert(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    # Define the image feature extractor
    self.img_featrs = models.densenet121(pretrained=True)
    self.img_featrs.classifier = nn.Identity()
    # Freeze the image feature extractor
    for param in self.img_featrs.parameters():
        param.requires_grad = False
    # Define the text feature extractor
    self.txt_featrs = BertModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    # Freeze the text feature extractor
    for param in self.txt_featrs.parameters():
        param.requires_grad = False
    # Define the attention layer
    self.attention = CrossAttention(self.img_featrs.features.norm5.num_features, self.txt_featrs.config.hidden_size)
    # Define the fully-connected layers
    self.img_featrs_fc = nn.Linear(self.img_featrs.features.norm5.num_features, 512)
    self.text_features_fc = nn.Linear(self.txt_featrs.config.hidden_size, 512)
    self.classifier = nn.Linear(512 * 2, num_classes)
    
  def forward(self, images, input_ids, attention_mask):
    image_features = self.img_featrs(images)
    text_features = self.txt_featrs(input_ids=input_ids, attention_mask=attention_mask)[0]
    image_features = image_features.view(image_features.size(0), -1, self.img_featrs.features.norm5.num_features)
    
    attention_outputs = self.attention(image_features, text_features)
    # Average pooling
    attention_weights = attention_outputs.mean(dim=1)
    
    image_features = self.img_featrs_fc(attention_weights)
    text_features = self.text_features_fc(text_features.mean(dim=1))
    
    combined_features = torch.cat([image_features, text_features], dim=1)
    logits = self.classifier(combined_features)
    return logits

class Bensity(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    # Define the image feature extractor
    self.img_featrs = models.densenet169(pretrained=True)
    self.img_featrs.classifier = nn.Identity()
    # Freeze the image feature extractor
    for param in self.img_featrs.parameters():
      param.requires_grad = False
    # Define the text feature extractor
    self.txt_featrs = BertModel.from_pretrained("prajjwal1/bert-tiny")
    # Freeze the text feature extractor
    for param in self.txt_featrs.parameters():
      param.requires_grad = False
    # Define the attention layer
    self.image_attention = SelfAttention(self.img_featrs.features.norm5.num_features)
    self.text_attention = SelfAttention(self.txt_featrs.config.hidden_size)
    # Define the fully-connected layers
    self.img_featrs_fc = nn.Linear(self.img_featrs.features.norm5.num_features, 512)
    self.text_features_fc = nn.Linear(self.txt_featrs.config.hidden_size, 512)
    self.classifier = nn.Linear(512 * 2, num_classes)
    
  def forward(self, images, input_ids, attention_mask):
    image_features = self.img_featrs(images)
    text_features = self.txt_featrs(input_ids=input_ids, attention_mask=attention_mask)[0]
    
    image_attention_outputs = self.image_attention(image_features.view(image_features.size(0), -1, self.img_featrs.features.norm5.num_features))
    text_attention_outputs = self.text_attention(text_features)
    # Average pooling
    image_attention_weights = image_attention_outputs.mean(dim=1)
    text_attention_weights = text_attention_outputs.mean(dim=1)
    
    image_features = self.img_featrs_fc(image_attention_weights)
    text_features = self.text_features_fc(text_attention_weights)
    
    combined_features = torch.cat([image_features, text_features], dim=1)
    logits = self.classifier(combined_features)
    return logits

class Censity(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    # Define the image feature extractor
    self.img_featrs = models.densenet169(pretrained=True)
    self.img_featrs.classifier = nn.Identity()
    # Freeze the image feature extractor
    for param in self.img_featrs.parameters():
      param.requires_grad = False
    # Define the text feature extractor
    self.txt_featrs = BertModel.from_pretrained("prajjwal1/bert-tiny")
    # Freeze the text feature extractor
    for param in self.txt_featrs.parameters():
      param.requires_grad = False
    # Define the attention layer
    self.attention = CrossAttention(self.txt_featrs.config.hidden_size, self.img_featrs.features.norm5.num_features)
    # Define the fully-connected layers
    self.img_featrs_fc = nn.Linear(self.img_featrs.features.norm5.num_features, 512)
    self.text_features_fc = nn.Linear(self.txt_featrs.config.hidden_size, 512)
    self.classifier = nn.Linear(512 * 2, num_classes)
    
  def forward(self, images, input_ids, attention_mask):
    image_features = self.img_featrs(images)
    text_features = self.txt_featrs(input_ids=input_ids, attention_mask=attention_mask)[0]
    image_features = image_features.view(image_features.size(0), -1, self.img_featrs.features.norm5.num_features)
    # Cross attention
    attention_outputs = self.attention(text_features, image_features)
    # Average pooling
    attention_weights = attention_outputs.mean(dim=1)
    
    image_features = self.img_featrs_fc(attention_weights)
    text_features = self.text_features_fc(text_features.mean(dim=1))
    
    combined_features = torch.cat([image_features, text_features], dim=1)
    logits = self.classifier(combined_features)
    return logits

class ImCensity(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    # Define the image feature extractor
    self.img_featrs = models.densenet169(pretrained=True)
    self.img_featrs.classifier = nn.Identity()
    # Freeze the image feature extractor
    for param in self.img_featrs.parameters():
      param.requires_grad = False
    # Define the text feature extractor
    self.txt_featrs = BertModel.from_pretrained("prajjwal1/bert-tiny")
    # Freeze the text feature extractor
    for param in self.txt_featrs.parameters():
      param.requires_grad = False
    # Define the attention layer
    self.attention = CrossAttention(self.img_featrs.features.norm5.num_features, self.txt_featrs.config.hidden_size)
    # Define the fully-connected layers
    self.img_featrs_fc = nn.Linear(self.img_featrs.features.norm5.num_features, 512)
    self.text_features_fc = nn.Linear(self.txt_featrs.config.hidden_size, 512)
    self.classifier = nn.Linear(512 * 2, num_classes)
    
  def forward(self, images, input_ids, attention_mask):
    image_features = self.img_featrs(images)
    text_features = self.txt_featrs(input_ids=input_ids, attention_mask=attention_mask)[0]
    image_features = image_features.view(image_features.size(0), -1, self.img_featrs.features.norm5.num_features)
    
    attention_outputs = self.attention(image_features, text_features)
    # Average pooling
    attention_weights = attention_outputs.mean(dim=1)
    
    image_features = self.img_featrs_fc(attention_weights)
    text_features = self.text_features_fc(text_features.mean(dim=1))
    
    combined_features = torch.cat([image_features, text_features], dim=1)
    logits = self.classifier(combined_features)
    return logits

class ResT(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    # Define the image feature extractor
    self.img_featrs = models.resnet34(pretrained=True)
    self.img_featrs.fc = nn.Identity()
    # Freeze the image feature extractor
    for param in self.img_featrs.parameters():
        param.requires_grad = False
    # Define the text feature extractor
    self.txt_featrs = BertModel.from_pretrained("prajjwal1/bert-tiny")
    # Freeze the text feature extractor
    for param in self.txt_featrs.parameters():
        param.requires_grad = False
    # Define the attention layer
    self.attention = CrossAttention(512, self.txt_featrs.config.hidden_size, 512)
    # Define the fully-connected layers
    self.img_featrs_fc = nn.Linear(512, 512)
    self.text_features_fc = nn.Linear(self.txt_featrs.config.hidden_size, 512)
    self.classifier = nn.Linear(512 * 2, num_classes)
    
  def forward(self, images, input_ids, attention_mask):
    image_features = self.img_featrs(images)
    text_features = self.txt_featrs(input_ids=input_ids, attention_mask=attention_mask)[0]
    image_features = image_features.view(image_features.size(0), -1)
    
    attention_outputs = self.attention(image_features, text_features)
    # Average pooling
    attention_weights = attention_outputs.mean(dim=1)
    
    image_features = self.img_featrs_fc(attention_weights)
    text_features = self.text_features_fc(text_features.mean(dim=1))
    
    combined_features = torch.cat([image_features, text_features], dim=1)
    logits = self.classifier(combined_features)
    return logits