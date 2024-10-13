import streamlit as st
from io import BytesIO
from PIL import Image
import os
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import tempfile
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import io
# Создаем директории для сохранения файлов, если они не существуют
data_list = []
IMAGE_SAVE_DIR = 'saved_images'
VIDEO_SAVE_DIR = 'saved_videos'

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# media_file = st.file_uploader("Загрузите фото или видео для обнаружения объектов, видео только в формате mp4",
#                               type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],key=f"uploader_{st.session_state.uploader_key}",accept_multiple_files =True)


def clear_file_uploader():
    st.session_state.uploader_key += 1


os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)

st.write("# Обнаружение воздушных объектов с помощью анализа видеоинформации")

# Функция для обработки изображения (пока пустая)
def process_image(image_path: str):
    cap = cv2.VideoCapture(image_path)
    model = YOLO("best.pt")
    _, img = cap.read()
    results = model.predict(source=image_path, show=False)
    annotator = Annotator(img)
    for r in results:
                
        annotator = Annotator(img)
                
        boxes = r.boxes
        for box in boxes:
                    
            b = box.xyxy[0]
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
                
    img = annotator.result()
    return img

# Функция для обработки видео
def process_video(video_path: str):
    frame_no = 0
    model = YOLO('best.pt')
    names = model.names
    cap = cv2.VideoCapture(video_path)
    cap.set(3, 640)
    cap.set(4, 480)
    detected_timestamps = []
    st.title(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_window = st.image([])
    st.spinner("wait")
    while cap.isOpened():
        ret, img = cap.read()
        frame_no+=1
        if not ret:
            break
        if frame_no%20==0:
            classes = {}
            print(frame_no)
            results = model.predict(img)
            for r in results:
                annotator = Annotator(img)
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]
                    c = box.cls
                    vals = b.tolist()
                    classes[model.names[int(c)]] = vals
                    annotator.box_label(b, model.names[int(c)])
            pr = float(min(frame_no//total_frames,total_frames))
            print(pr)
            annotated_img = annotator.result()
            
            for r in results:
                    for c in r.boxes.cls:
                        print(classes)
            timestamp_sec = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 2)
            message = f"{classes} timestamp is: {timestamp_sec} seconds"
            if len(classes)>0:
                detected_timestamps.append(message)

            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

    cap.release()
    cv2.destroyAllWindows()
    output_file = "detected_timestamps.txt"
    with open(output_file, "w") as f:
        for timestamp in detected_timestamps:
            f.write(timestamp + "\n")
    with open(output_file, "r") as s:
        st.download_button(
            label=video_path,
            data=s,
            file_name="objects.txt",
            mime="text/plain",
            on_click=clear_file_uploader()
        )
        data_list.append(s)
    # with open('DetectObj/app/img/thn.jpg', mode="rb") as a:
    #     uploaded_file = io.BytesIO(a.read())
    #     st.file_uploader = uploaded_file
    return st.success("Обработка завершена!")


# Загрузка медиафайла
media_file = st.file_uploader(
    "Загрузите видео для обнаружения объектов, формат видео: mp4, avi, mov",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
    key=f"uploader_{st.session_state.uploader_key}", accept_multiple_files=True
)
if media_file is not None:
    for x in media_file:
            if x.type.startswith("video"):
                video_filename = os.path.join(VIDEO_SAVE_DIR, x.name)
                with open(video_filename, 'wb') as out_file:
                    out_file.write(x.read())

                # Вызываем функцию обработки видео
                process_video(video_filename)

                # Кнопка для сброса загрузчика
        
            if x.type.startswith("image"):
                    image = Image.open(x)
                    # Конвертируем изображение в формат JPG
                    img_io = BytesIO()
                    image = image.convert("RGB")  # Преобразуем изображение в RGB
                    image.save(img_io, format='JPEG')
                    img_io.seek(0)

                    # Сохранение изображения в папку
                    image_filename = os.path.join(IMAGE_SAVE_DIR, f"{x.name.split('.')[0]}.jpg")
                    print(image_filename)
                    with open(image_filename, 'wb') as out_file:
                        out_file.write(img_io.getbuffer())

                    # Вызываем функцию обработки изображения
                    pr_img = process_image(image_filename)
                    st.image(pr_img, caption="Загруженное изображение и сохраненное как JPG", use_column_width=True)
    st.button("Сбросить", on_click=clear_file_uploader)

                

