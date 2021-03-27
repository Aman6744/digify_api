import os
import cv2
import uuid
import numpy as np
from api.predict import predict
from flask import Flask, request

app = Flask(__name__)

@app.route('/digify', methods=['POST'])
def index():
    fileStorageImage = request.files['image'].read()
    npImage = np.fromstring(fileStorageImage, np.uint8)
    image = cv2.imdecode(npImage, cv2.IMREAD_GRAYSCALE)
    current_uuid = str(uuid.uuid4())
    image_name = current_uuid + ".png"
    image_path = os.path.join("api", "temp", image_name)
    cv2.imwrite(image_path, image)
    try: 
        text = predict(image_name)
        print(f"Text: {text}", end="\n\n")
        os.system(f"rm -r api/temp/{current_uuid}*")
        return {"status": "Success", 
                "text": text}, 202
    except:
        os.system(f"rm -r api/temp/{current_uuid}*")
        return {"status": "Maar li gyi h"}, 405