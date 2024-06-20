
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import tf_bodypix
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
from tf_bodypix.draw import draw_poses
from tensorflow.keras import preprocessing
import cv2
import json
from matplotlib import pyplot as plt
import numpy as np
from calculations import measure_body_sizes
import gradio as gr
import pandas as pd




# Load BodyPix model
bodypix_model = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))

rainbow = [
  [110, 64, 170], [143, 61, 178], [178, 60, 178], [210, 62, 167],
  [238, 67, 149], [255, 78, 125], [255, 94, 99],  [255, 115, 75],
  [255, 140, 56], [239, 167, 47], [217, 194, 49], [194, 219, 64],
  [175, 240, 91], [135, 245, 87], [96, 247, 96],  [64, 243, 115],
  [40, 234, 141], [28, 219, 169], [26, 199, 194], [33, 176, 213],
  [47, 150, 224], [65, 125, 224], [84, 101, 214], [99, 81, 195]
]

def process_images(front_img, side_img, real_height_cm):
    fimage_array = preprocessing.image.img_to_array(front_img)
    simage_array = preprocessing.image.img_to_array(side_img)

    # bodypix prediction
    frontresult = bodypix_model.predict_single(fimage_array)
    sideresult = bodypix_model.predict_single(simage_array)

    front_mask = frontresult.get_mask(threshold=0.75)
    side_mask = sideresult.get_mask(threshold=0.75)

    # preprocessing.image.save_img(f'{output_path}/frontbodypix-mask.jpg',front_mask)
    # preprocessing.image.save_img(f'{output_path}/sidebodypix-mask.jpg',side_mask)

    front_colored_mask = frontresult.get_colored_part_mask(front_mask, rainbow)
    side_colored_mask = sideresult.get_colored_part_mask(side_mask, rainbow)

    # preprocessing.image.save_img(f'{output_path}/frontbodypix-colored-mask.jpg',front_colored_mask)
    # preprocessing.image.save_img(f'{output_path}/sidebodypix-colored-mask.jpg',side_colored_mask)

    frontposes = frontresult.get_poses()
    front_image_with_poses = draw_poses(
        fimage_array.copy(), # create a copy to ensure we are not modifing the source image
        frontposes,
        keypoints_color=(255, 100, 100),
        skeleton_color=(100, 100, 255)
    )

    sideposes = sideresult.get_poses()
    side_image_with_poses = draw_poses(
        simage_array.copy(), # create a copy to ensure we are not modifing the source image
        sideposes,
        keypoints_color=(255, 100, 100),
        skeleton_color=(100, 100, 255)
    )
    # print(np.array(simage).shape)
    # print(np.array(side_colored_mask).shape)

    # preprocessing.image.save_img(f'{output_path}/frontbodypix-poses.jpg', front_image_with_poses)
    # preprocessing.image.save_img(f'{output_path}/sidebodypix-poses.jpg', side_image_with_poses)

    body_sizes = measure_body_sizes(side_colored_mask, front_colored_mask, sideposes, frontposes, real_height_cm, rainbow)
    measurements_df = pd.DataFrame([body_sizes[0]])
    return measurements_df

# Create the Gradio interface
interface = gr.Interface(
    fn=process_images,
    inputs=[
        gr.Image(label="Upload Front Pose"),
        gr.Image(label="Upload Side Pose"),
        gr.Number(label="Enter Height (cm)")
    ],
    outputs=[
        gr.DataFrame(label="Body Measurements")
    ],
    title="Body Sizing System Demo",
    description="Upload two images: Front View and Side View, and input the height in cm."
)

# Launch the app
interface.launch(share=True)