import threading
import shutil
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from functions import get_train_augs,get_val_augs,train_model,eval_model
from SegmentationModel import SegmentationModel
from SegmentationDataset import SegmentationDataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ENCODER = 'timm-efficientnet-b0'
WEIGHTS = 'imagenet'
locker = threading.Lock()
training_threads = {}
model_counter = 1
modeles = [
    {
        "id": 0,
        "value": "Unet"
    },
        {
        "id": 1,
        "value": "UnetPlusPlus"
    },
        {
        "id": 2,
        "value": "MAnet"
    },
        {
        "id": 3,
        "value": "Linknet"
    },
        {
        "id": 4,
        "value": "FPN"
    },
        {
        "id": 5,
        "value": "PSPNet"
    },
            {
        "id": 6,
        "value": "DeepLabV3"
    },
            {
        "id": 7,
        "value": "DeepLabV3Plus"
    },
            {
        "id": 8,
        "value": "PAN"
    },
    ]

def delete_everything_in_folder(folder_path):
    shutil.rmtree(folder_path)
    os.mkdir(folder_path)
def start_training(TRAIN_DATA_PATH,EPOCHS,LR,IMG_SIZE,BATCH_SIZE,MODEL,TEST_SIZE):
    global model_counter
    model_name = model_counter
    model_counter += 1
    t = threading.Thread(target=train_thread, args=(TRAIN_DATA_PATH,EPOCHS,LR,IMG_SIZE,BATCH_SIZE,MODEL,TEST_SIZE))
    t.name=str(model_name)
    t.start()
    training_threads[model_name] = t
    return f"Training Model {model_name} started."

def train_thread(TRAIN_DATA_PATH,EPOCHS,LR,IMG_SIZE,BATCH_SIZE,MODEL,TEST_SIZE):
    print("НАЧАЛО ОБУЧЕНИЯ")
    print(EPOCHS, BATCH_SIZE, MODEL)
    
    DATA_DIR = 'Datasets/' + threading.current_thread().name + '/'
    SAVE_DIR = 'Saves/' + threading.current_thread().name + '/'
    global DEVICE, ENCODER, WEIGHTS

    #Это читка файла разметки
    df = pd.read_csv(DATA_DIR + TRAIN_DATA_PATH)
    locker.acquire()
    print(df.shape)
    print(df.head(10))
    locker.release()

    # Разбитие на тестовую и валидационную выборку
    train_df, val_df = train_test_split(df, test_size=TEST_SIZE, random_state=57)

    train_data = SegmentationDataset(train_df, get_train_augs(IMG_SIZE), DATA_DIR)
    val_data = SegmentationDataset(val_df, get_val_augs(IMG_SIZE), DATA_DIR)
    locker.acquire()
    print(f"Size of Trainset : {len(train_data)}")
    print(f"Size of Validset : {len(val_data)}")
    locker.release()

    #Это подгрузка изображений по группам
    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    locker.acquire()
    print(f"Total number of batches in Train Loader: {len(trainloader)}")
    print(f"Total number of batches in Val Loader: {len(valloader)}")
    locker.release()

    #Это объявление модели
    model = SegmentationModel(MODEL, ENCODER, WEIGHTS)
    model.to(DEVICE)
    print(model.parameters(),ENCODER,WEIGHTS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = 1e9

    # Цикл обучения по эпохам
    for i in range(1, EPOCHS + 1):
        train_loss, train_dice = train_model(trainloader, model, optimizer, DEVICE)
        val_loss, val_dice = eval_model(valloader, model, DEVICE)
        #Если в этот цикл потерь меньше - сохраняем
        if val_loss < best_val_loss:
            # Save the best model
            torch.save(model.state_dict(), SAVE_DIR + "best_model.pt")
            print("MODEL SAVED")

            best_val_loss = val_loss
        #Иначе - обрубаем обучение
        #

        #Вот это итоги эпохи
        locker.acquire()
        print(f"\033[1m\033[92m Epoch {i} Train Loss {train_loss} Train dice {train_dice} Val Loss {val_loss} Val Dice {val_dice}")
        locker.release()

    locker.acquire()
    print(f"Model {threading.current_thread().name} training completed.")
    locker.release()
    # ДАЛЬШЕ ВСЯ ХУЙНЯ ЧТОБ СДЕЛАТЬ ПРЕДСКАЗАНИЕ
    #model.load_state_dict(torch.load(SAVE_DIR + "best_model.pt"))

    # Function to output the prediction mask
    #def make_inference(idx):
        #image, mask = val_data[idx]
        #logits_mask = model(image.to(DEVICE).unsqueeze(0))  # (C, H, W) -> (1, C, H, W)

        # Predicted mask
        #pred_mask = torch.sigmoid(logits_mask)
        #pred_mask = (pred_mask > 0.5) * 1.0

        #return image, mask, pred_mask

    # Compare predictions with original
    #for i in np.random.randint(0, len(val_data), 5):
        #image, mask, pred_mask = make_inference(i)

        # Show image
        #plt.figure(figsize=(10, 3))
        #plt.subplot(1, 3, 1)
        #plt.imshow(np.transpose(image, (1, 2, 0)))
        #plt.axis('off')
        #plt.title('IMAGE')

        # Show original mask
        #plt.subplot(1, 3, 2)
        #plt.imshow(np.transpose(mask, (1, 2, 0)), cmap='gray')
        #plt.axis('off')
        #plt.title('GROUND TRUTH')

        # Show predicted mask
        #plt.subplot(1, 3, 3)
        #plt.imshow(np.transpose(pred_mask.detach().cpu().squeeze(0), (1, 2, 0)), cmap='gray')
        #plt.axis('off')
        #plt.title('PREDICTION')
        #plt.show(block=False)

#start_training(TRAIN_DATA_PATH='dataset/train.csv',EPOCHS=2,LR = 0.001,IMG_SIZE = 320,BATCH_SIZE = 32,MODEL="PAN",TEST_SIZE=0.2   )
