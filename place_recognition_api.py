import pickle
import json
import io
import cv2
import numpy as np
import pandas as pd
from rank_from_superpoints_utils import rank_superpoints
from typing import Optional
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

# load database
db_dir = 'superpoints_database.csv'
database_df = pd.read_csv(db_dir)
database_df = database_df.set_index('file_name')

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

@app.post('/do_ranking/')
async def do_ranking(item: UploadFile = File(...), rank: int = 2):
    image_bytes = item.file.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    rank_result = rank_superpoints(image, database_df, 'api_output',
                                   image_size=(320, 240), rank=rank)
    print('Returned rank_result value:', rank_result)
    return json.dumps(rank_result)