import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import streamlit as st

st.title("Image Segmentation Mask App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    orig_size = image.size

stroke_width = st.sidebar.slider("Stroke width: ", 1, 50, 25)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="#eee",
    background_image=Image.open(uploaded_file) if uploaded_file else None,
    update_streamlit=True,
    height=500,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    mask = canvas_result.image_data.astype("uint8")[..., 3]
    mask[mask > 0] = 255
    if st.button("Save Mask Image") and orig_size:
        mask = cv2.resize(mask, orig_size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite("mask.png", mask)
        st.success("Mask Image saved successfully.")
