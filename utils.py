# import required libraries

from docarray import Document
import datetime
from PIL import Image
import base64
from io import BytesIO
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt


load_dotenv()

DALLE_URL = 'grpcs://dalle-flow.dev.jina.ai'
#if above url is not working, use
#'grpc://dalle-flow.jina.ai:51005'

# Image to base64 string
def img_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str


def base64_to_img(img_str):
    img_data = base64.b64decode(img_str)
    img = Image.open(BytesIO(img_data))
    return img


def get_images_from_dalle(prompt, num_images=5):
    now = datetime.datetime.now()
    da = (
        Document(text=prompt)
        .post(DALLE_URL, parameters={"num_images": num_images})
        .matches
    )
#     print(
#         "{} response in {}".format(da, (datetime.datetime.now() - now).total_seconds())
#     )
    img_list = []

    for _d in da:
        _d.load_uri_to_image_tensor()
        _img = Image.fromarray(_d.tensor)
        # Return the first one
        img_list.append(_img)
        # return _img

    return img_list

def plot_images(prompt,result,num_images):
    plt.figure(figsize=(15,15)) # specifying the overall grid size

    for i in range(num_images):
        plt.subplot(1,num_images,i+1)    # the number of images in the grid is 1*5 (5)
        plt.imshow(result[i])
        plt.axis('off')
        plt.savefig(prompt[:10]+'.png')
        plt.title(prompt,fontsize=20,loc="center",
                  fontstyle='italic')
        plt.show()
    return "success"

    
