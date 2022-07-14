import utils
import streamlit as st
from PIL import Image

st.title("Implementation of *DALL-E*")
st.subheader("Creating Images from Text")

prompt = st.text_input("Enter your text here", "Text")


def container_imgs(prompt):

    set_imgs = utils.get_images_from_dalle(prompt, num_images=4)

    with st.container():
        for col in st.columns(1):
            col.image(set_imgs, width=150)


container_imgs(prompt)
