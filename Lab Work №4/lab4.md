# Лабораторная работа №4
Здравствуйте, возращаюсь в уличную гонку. Сделал лабу по коммандному проекту (потому что он первее по сдаче, чем диплом). Там тема ~Веб-сервис для обучения сегментационных моделей НС. Вводятся параметры обучения, 2 файла: датасет и файл разметки, выбирается модель и начинается обучение. Во время обучения можно посмотреть метрики обучения или остановить обучение. По окончании обучения скачивается файл с обученной моделью. <br/>
Проектирование REST API. <br/>
# Запросы
## 1. Данный запрос выводит список доступных для обучения моделей
### Метод: 
GET
### Путь: 
/modeles
### Параметры запроса:
-
### Ответ: 
В случае успешного запроса, сервер возвращает JSON-объект:
- AnswerDate: время ответа (полезная штука, один раз спасла меня, теперь везде пихаю)
- Description: массив строк с наименованиями моделей обучения НС
![image_2024-02-04_22-50-38](https://github.com/AlvaroFironze/HSE-SoftwareArchitecture/assets/85906595/d74de63f-3e59-4743-b244-5b878f53c862)
![image_2024-02-04_22-50-38](https://github.com/AlvaroFironze/HSE-SoftwareArchitecture/assets/85906595/d41f670a-8404-4268-8646-4627ad11a42a)
### Реализация API:
```
modeles = ["Unet", "UnetPlusPlus", "MAnet", "Linknet", "FPN", "PSPNet", "DeepLabV3", "DeepLabV3Plus", "PAN"]
@app.get("/modeles/")
def get():
    global modeles
    return {"AnswerDate":datetime.datetime.now().isoformat(),"Description":modeles}
```
## 2. Отправка введённых пользователей параметров обучения на сервер  
### Метод:  
POST 
### Путь:
/study 
### Параметры запроса:
- User: str, - идентификатор подключения пользователя <br/>
- FileName: str, - файл структуры разметки, прикрепляемый пользователем (Наименование во внутреннем хранилище) <br/>
- Epochs: int, - Кол-во эпох обучения >0 <br/>
- LR: float, - Интенсивность обучения (0;1] <br/>
- ImgSize:int, - Размерность изображений при обучении [112;1048]  <br/>
- BatchSize:int, - Размер партии изображений в выборке обучения [1,100] <br/>
- Model:str, - Наименование модели из списка <br/>
- TestSize:float. - Процент валидационной выборки относительно всех файлов (0;0.9] <br/>
### Ответ: 
В случае успешного запроса, сервер возвращает JSON-объект:
- AnswerDate: Время ответа
- ModelName: Идентификатор обучения на сервере
![image_2024-02-04_22-50-38](https://github.com/AlvaroFironze/HSE-SoftwareArchitecture/assets/85906595/d44fe139-5137-4a99-983d-04de0db6a244)
![image_2024-02-04_22-50-38](https://github.com/AlvaroFironze/HSE-SoftwareArchitecture/assets/85906595/5d9dc84c-0722-4858-b7d9-e38fd5ad5835)
### Ошибки:
В случае возникновения ошибки, сервер возвращает соответствующий HTTP-статус и информацию об ошибке в формате JSON.
status_code=400, detail="Wrong parameters"
![image_2024-02-04_22-50-38](https://github.com/AlvaroFironze/HSE-SoftwareArchitecture/assets/85906595/6eedeb66-b1a2-4bd0-938b-cb8cd45a91cc)
![image_2024-02-04_22-50-38](https://github.com/AlvaroFironze/HSE-SoftwareArchitecture/assets/85906595/11044bf2-8c2d-41f8-b657-bd424a69eaa6)

### Реализация API:
```
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
```
## 3. Изменение количества эпох обучения
### Метод:  
PUT
### Путь:
/study 
### Параметры запроса:
- ModelName:str, - Идентификатор обучения на сервере
- Epochs:int. - - Кол-во эпох обучения >0 <br/>
### Ответ: 
- AnswerDate:
- Description: Если введённое количество эпох <= уже пройденному количеству: "Study will be stopped after current epoch", иначе "Successful"
![image_2024-02-04_22-50-38](https://github.com/AlvaroFironze/HSE-SoftwareArchitecture/assets/85906595/71b9e058-1241-4e72-8883-34a74decf401)
![image_2024-02-04_22-50-38](https://github.com/AlvaroFironze/HSE-SoftwareArchitecture/assets/85906595/2ae8b4bf-8635-44f0-a216-2b7089cc2cf4)
### Ошибки:
В случае возникновения ошибки, сервер возвращает соответствующий HTTP-статус и информацию об ошибке в формате JSON.
status_code=404, detail="Study not found"
![image_2024-02-04_22-50-38](https://github.com/AlvaroFironze/HSE-SoftwareArchitecture/assets/85906595/7707418e-b528-4643-b5c7-29cb322f1d06)
![image_2024-02-04_22-50-38](https://github.com/AlvaroFironze/HSE-SoftwareArchitecture/assets/85906595/5da0a042-58d8-4a65-bfbb-953cd6fdb275)
### Реализация API:
```
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
```
## 4. Получение метрик обучения
### Метод:  
GET
### Путь:
/study 
### Параметры запроса:
- ModelName:str, - Идентификатор обучения на сервере
### Ответ: 
- AnswerDate:,
- train_loss: массив Процентов потерь тренировочной выборки,
- train_dice: массив Коэффицентов Дайса тренировочной выборки,
- val_loss: массив Процентов потерь валидационной выборки,
- val_dice: массив Коэффицентов Дайса валидационной выборки,
![image_2024-02-04_22-50-38](https://github.com/AlvaroFironze/HSE-SoftwareArchitecture/assets/85906595/e78b2c1f-9145-43a2-be52-f6e57b9bfecc)
![image_2024-02-04_22-50-38](https://github.com/AlvaroFironze/HSE-SoftwareArchitecture/assets/85906595/421761de-bce6-4867-b280-ce045a6aacf2)
### Ошибки:
В случае возникновения ошибки, сервер возвращает соответствующий HTTP-статус и информацию об ошибке в формате JSON.
status_code=404, detail="Study not found"
### Реализация API:
```
@app.get("/study/")
def get(ModelName:str):
    global training_threads
    if ModelName in training_threads:
        return {"AnswerDate": datetime.datetime.now().isoformat(), 'train_loss': training_threads[ModelName]['train_loss'],
                'train_dice': training_threads[ModelName]['train_dice'],'val_loss': training_threads[ModelName]['val_loss'],
                'val_dice': training_threads[ModelName]['val_dice']}
    else:
        raise HTTPException(status_code=404, detail="Item not found")
```
## 5. Принудительная остановка обучения
### Метод:  
DELETE
### Путь:
/study 
### Параметры запроса:
- ModelName:str, - Идентификатор обучения на сервере
### Ответ: 
- AnswerDate:
- FilePath: путь во внутреннем хранилище до чекпоинта для выгрузки модели на фронте
![image_2024-02-04_22-50-38](https://github.com/AlvaroFironze/HSE-SoftwareArchitecture/assets/85906595/90cb828c-69cd-4898-be38-be27f3fc4c0a)
![image_2024-02-04_22-50-38](https://github.com/AlvaroFironze/HSE-SoftwareArchitecture/assets/85906595/a1337094-c184-4696-a7d5-a2cd2b04f143)

### Ошибки:
В случае возникновения ошибки, сервер возвращает соответствующий HTTP-статус и информацию об ошибке в формате JSON.
status_code=404, detail="Study not found"
### Реализация API:
```
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
```
## 6. Получение модели с определённой эпохи
### Метод:  
GET
### Путь:
/checkstudy 
### Параметры запроса:
- ModelName:str, - Идентификатор обучения на сервере
- Epoch:int, - Номер эпохи обучения
### Ответ: 
- AnswerDate: ,
- FilePath: путь во внутреннем хранилище до чекпоинта для выгрузки модели на фронте
![image_2024-02-04_22-50-38](https://github.com/AlvaroFironze/HSE-SoftwareArchitecture/assets/85906595/a088b3c3-912f-4de5-8899-9dfd90702029)
![image_2024-02-04_22-50-38](https://github.com/AlvaroFironze/HSE-SoftwareArchitecture/assets/85906595/e29b0a02-cfc5-48ee-a33f-1ac27713da37)

### Ошибки:
В случае возникновения ошибки, сервер возвращает соответствующий HTTP-статус и информацию об ошибке в формате JSON.
status_code=404, detail="Study not found"
### Реализация API:
```
@app.get("/checkstudy/")
def get(ModelName:str, Epoch:int):
    global training_threads
    if ModelName in training_threads:
        return {"AnswerDate": datetime.datetime.now().isoformat(), "FilePath":f"Saves/{ModelName}/{Epoch}.pt"}
    else:
        raise HTTPException(status_code=404, detail="Item not found")
```
## 6. Добавление моделей в список
### Метод:  
POST
### Путь:
/modeles
### Параметры запроса:
- ModelName:str, - Название модели
### Ответ: 
- AnswerDate: ,
- Description
## 7. Удаление модели в список
### Метод:  
DELETE
### Путь:
/modeles
### Параметры запроса:
- ModelName:str, - Название модели
### Ответ: 
- AnswerDate: ,
- Description
