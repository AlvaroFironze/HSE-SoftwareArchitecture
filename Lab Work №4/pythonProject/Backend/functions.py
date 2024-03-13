import albumentations as A
from tqdm import tqdm
import torch
def get_train_augs(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),      # Horizontal Flip with 0.5 probability
        A.VerticalFlip(p=0.5)         # Vertical Flip with 0.5 probability
    ], is_check_shapes=False)

def get_val_augs(img_size):
    return A.Compose([
        A.Resize(img_size, img_size)
    ], is_check_shapes=False)

# Function to train the model
def train_model(data_loader, model, optimizer,device):
    total_loss = 0.0
    model.train()
    total_dices = 0.0

    for images, masks in tqdm(data_loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits, loss,dices = model(images, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dices += dices.item()

    return total_loss / len(data_loader), total_dices/len(data_loader)
# Function to evaluate the model
def eval_model(data_loader, model,device):
    total_loss = 0.0
    model.eval()
    total_dices=0.0

    with torch.no_grad():
        for images, masks in tqdm(data_loader):
            images = images.to(device)
            masks = masks.to(device)

            logits, loss, dices = model(images, masks)
            total_loss += loss.item()
            total_dices +=dices.item()

        return total_loss / len(data_loader), total_dices/len(data_loader)

