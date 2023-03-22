import fastapi
import functions as f
import cv2
from PIL import Image
from collections import Counter
import numpy as np
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile
import base64
import skin_model as m
import requests
            

app = FastAPI()

origins = [
    "http://localhost:3000"  # 스프링 부트 애플리케이션이 실행 중인 도메인
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/image")
async def image(data: dict):

    try:
        image_data = data["image"]
        decoded_image = base64.b64decode(image_data.split(",")[1])

        with open("saved.jpg","wb") as fi:
            fi.write(decoded_image)
      
        f.save_skin_mask("saved.jpg")
   
        ans = m.get_season("temp.jpg")
        os.remove("temp.jpg")
        os.remove("saved.jpg")
   
        if ans == 3:
            ans += 1
        elif ans == 0:
            ans = 3

        test = {'result': ans}
        encoded_data = base64.b64encode(str(test).encode('utf-8')).decode('utf-8')
 
        response = requests.post('http://localhost:3000/output',json={'encodedData':encoded_data})
        return JSONResponse(content={"message":"complete"})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="fail")


@app.post("/lip")
async def lip(data: dict):
    try:
        image_data = data["image"]
        decoded_image = base64.b64decode(image_data.split(",")[1])
       
        with open("saved.jpg","wb") as fi:
            fi.write(decoded_image)
        
        path = r"saved.jpg"
       
        rgb_codes = f.get_rgb_codes(path)  #check point
     
        random_rgb_codes = f.filter_lip_random(rgb_codes,40) #set number of randomly picked sample as 40

        os.remove("saved.jpg")
     
        types = Counter(f.calc_dis(random_rgb_codes))
    
        max_value_key = max(types, key=types.get)
        print(max_value_key)
        if max_value_key == 'sp':
            result = 1
        elif max_value_key == 'su':
            result = 2
        elif max_value_key == 'au':
            result = 3
        elif max_value_key == 'win':
            result = 4
        
        data = {'image':image_data,'result':result}
        encoded_data = base64.b64encode(str(data).encode('utf-8')).decode('utf-8')        
        response = requests.post("http://localhost:3000/output2", json={'encodedData':encoded_data})
        
        print(response)
        
        return JSONResponse(content={"message":"complete"})
    except Exception as e:
        raise HTTPException(status_code=500, detail="fail")

