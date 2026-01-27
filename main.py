from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
import uuid
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# создаём папки
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

# загружаем модель
model = YOLO("yolov8n.pt")


def is_inside(cx, cy, x1, y1, x2, y2):
    return x1 <= cx <= x2 and y1 <= cy <= y2


@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...)
):
    # сохраняем изображение
    image_id = str(uuid.uuid4())
    image_path = f"uploads/{image_id}.jpg"

    with open(image_path, "wb") as f:
        f.write(await file.read())

    # читаем изображение
    image = cv2.imread(image_path)

    # детекция людей
    results = model(image)

    count = 0

    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:  # person
                x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
                cx = (x1b + x2b) // 2
                cy = (y1b + y2b) // 2

                # рисуем bbox
                cv2.rectangle(image, (x1b, y1b), (x2b, y2b), (0, 255, 0), 2)

                if is_inside(cx, cy, x1, y1, x2, y2):
                    count += 1
                    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

    # рисуем зону стола
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(
        image,
        f"Guests: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # сохраняем результат
    result_path = f"results/{image_id}.jpg"
    cv2.imwrite(result_path, image)

    return JSONResponse({
        "guests_count": count,
        "result_image": result_path
    })
