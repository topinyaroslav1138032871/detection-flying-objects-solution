import streamlit as st
from PIL import Image

# Заголовок страницы
st.title('Комплексное решение по кейсу "Обнаружение воздушных объектов с помощью анализа видеоинформации" от команды Deers Squad.')
st.write("")
# Описание кейса
st.markdown("""
    Данный WEB-сервис представляет собой систему, способную обнаруживать беспилотные летательные аппараты 
    на видеоинформации с использованием дообученной модели YOLOv11. Объекты классифицируются на самолеты и вертолеты, 
    результаты обнаружения отображаются с таймкодами для анализа.
""")
st.write("")

# Размещение логотипов в колонках
col1, col2 = st.columns([1, 1])  # Пропорции колонок можно менять для регулировки ширины

with col1:
    image_mpt = Image.open('DetectObj/app/img/mpt_logo_red.png')
    st.image(image_mpt, use_column_width=True)

with col2:
    image_rus = Image.open('DetectObj/app/img/russia.png')
    st.image(image_rus, use_column_width=True)



# Добавление дополнительного контента при необходимости
