from fastapi import FastAPI,UploadFile,File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app=FastAPI()

@app.get("/")
async def home_page():
    return "Hello this is the home page"

model=tf.keras.models.load_model("CNN_model.h5")
classes=['EarlyBlight', 'Healthy', 'LateBlight']

def get_image_as_numpy(data)->np.ndarray:
    image=np.array(Image.open((BytesIO(data))))
    return image


@app.post("/predict")
async def predict(
    file:UploadFile=File(...)
):
    image=get_image_as_numpy(await file.read())
    resized_image=np.expand_dims(image,0)
    preds=[np.argmax(i) for i in model.predict(resized_image)][0]
    return f"The predicted label is {classes[preds]}"

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=7000)
    
