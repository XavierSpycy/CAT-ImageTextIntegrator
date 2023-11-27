import os
import torch
import numpy as np

def mul_clf_predict(model, model_name, test_loader, threshold=0.4, device=torch.device('cuda')):
    model.to(device)
    model.load_state_dict(torch.load(os.path.join("model_hub", f"{model_name}.pth"), map_location=device))
    label_test = []
    model.eval()
    with torch.no_grad():
        for img, txt in test_loader:
            img = img.to(device)
            input_ids = txt['input_ids'].to(device)
            attention_mask = txt['attention_mask'].to(device)
            labels = txt['label'].to(device)
            prediction = model(img, input_ids, attention_mask)
            predicted_prob = torch.sigmoid(prediction)
            predicted_labels = (predicted_prob > threshold).cpu().numpy()
            label_test.append(predicted_labels)
    label_test = np.concatenate(label_test).astype(int)
    return label_test