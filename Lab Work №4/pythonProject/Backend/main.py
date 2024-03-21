import threading
import shutil
import glob
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import datetime
from fastapi import FastAPI,HTTPException

from functions import get_train_augs,get_val_augs,train_model,eval_model
from SegmentationModel import SegmentationModel
from SegmentationDataset import SegmentationDataset
app = FastAPI()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ENCODER = 'timm-efficientnet-b0'
WEIGHTS = 'imagenet'
locker = threading.Lock()
training_threads = {}
model_counter = 1
modeles = ["Unet", "UnetPlusPlus", "MAnet", "Linknet", "FPN", "PSPNet", "DeepLabV3", "DeepLabV3Plus", "PAN"]
training_threads['rrrr'] = {'model': 't', 'is_stop': False}
@app.get("/study/")
def get(ModelName:str):
    global training_threads
    if ModelName in training_threads:
        return {"AnswerDate": datetime.datetime.now().isoformat(), 'train_loss': training_threads[ModelName]['train_loss'],
                'train_dice': training_threads[ModelName]['train_dice'],'val_loss': training_threads[ModelName]['val_loss'],
                'val_dice': training_threads[ModelName]['val_dice']}
    else:
        raise HTTPException(status_code=404, detail="Item not found")
@app.get("/modeles/")
def get():
    global modeles
    return {"AnswerDate":datetime.datetime.now().isoformat(),"Description":modeles}

def delete_everything_in_folder(folder_path):
    shutil.rmtree(folder_path)
    os.mkdir(folder_path)
@app.put("/study/")
def put(ModelName:str, Epochs:int):
    global training_threads
    if ModelName in training_threads:
        if len(training_threads[ModelName]['train_loss'])>=Epochs:
            training_threads[ModelName]['epochs'] = Epochs
            return {"AnswerDate": datetime.datetime.now().isoformat(), "Description": "Study will be stopped after current epoch"}
        else:
            training_threads[ModelName]['epochs'] = Epochs
            return {"AnswerDate": datetime.datetime.now().isoformat(), "Description": "Successful"}
    else:
        raise HTTPException(status_code=404, detail="Study not found")
@app.post("/study/")
def post(User: str, FileName: str, Epochs:int,LR:float,ImgSize:int,BatchSize:int,Model:str,TestSize:float):
    if (Epochs<1 or LR<+0 or LR>1 or ImgSize<122 or ImgSize>1048 or BatchSize<1 or BatchSize <1 or BatchSize>100 or
            Model not in modeles or TestSize<=0 or TestSize>0.9 or os.path.exists(f'Datasets/{User}/{FileName}')):
        raise HTTPException(status_code=400, detail="Wrong parameters")
    global model_counter
    model_name = User+'-'+str(model_counter)
    model_counter += 1
    t = threading.Thread(target=train_thread, args=(FileName,LR,ImgSize,BatchSize,Model,TestSize))
    t.name=str(model_name)
    t.start()
    global training_threads
    training_threads[str(model_name)] = {'model':t, 'is_stop':False, 'epochs':Epochs}
    print(f"Training Model {model_name} started.")
    return {"AnswerDate":datetime.datetime.now().isoformat(),"ModelName":model_name}
@app.delete("/study/")
def delete(ModelName:str):
    global training_threads
    if ModelName in training_threads:
        training_threads[ModelName]['is_stop']=True
        list_of_files = glob.glob(f'Saves/{ModelName}/*.pt')
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)
        return {"AnswerDate": datetime.datetime.now().isoformat(), "FilePath":f"{latest_file.replace('\\','/')}"}

    else:
        raise HTTPException(status_code=404, detail="Item not found")
@app.get("/beststudy/")
def get(ModelName:str, Epoch:int):
    global training_threads
    if ModelName in training_threads:
        return {"AnswerDate": datetime.datetime.now().isoformat(), "FilePath":f"Saves/{ModelName}/{Epoch}.pt"}
    else:
        raise HTTPException(status_code=404, detail="Item not found")
def train_thread(TRAIN_DATA_PATH,LR,IMG_SIZE,BATCH_SIZE,MODEL,TEST_SIZE):
    global training_threads
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
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = 1e9

    # Цикл обучения по эпохам
    for i in range(1, training_threads[threading.current_thread().name]['epochs'] + 1):
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
        if 'train_loss' in training_threads[threading.current_thread().name]:
            training_threads[threading.current_thread().name]['train_loss'].append(train_loss)
            training_threads[threading.current_thread().name]['train_dice'].append(train_dice)
            training_threads[threading.current_thread().name]['val_loss'].append(val_loss)
            training_threads[threading.current_thread().name]['val_dice'].append(val_dice)
        else:
            training_threads[threading.current_thread().name]['train_loss']=[train_loss]
            training_threads[threading.current_thread().name]['train_dice']=[train_dice]
            training_threads[threading.current_thread().name]['val_loss']=[val_loss]
            training_threads[threading.current_thread().name]['val_dice']=[val_dice]
        print(f"\033[1m\033[92m Epoch {i} Train Loss {train_loss} Train dice {train_dice} Val Loss {val_loss} Val Dice {val_dice}")
        locker.release()
        if training_threads[threading.current_thread().name]['is_stop']:
            break

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
