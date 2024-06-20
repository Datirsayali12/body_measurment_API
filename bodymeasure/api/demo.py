import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import preprocessing
import os
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
import math



def euclidean_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def convert_to_real_measurements(pixel_measurement, pixel_height, real_height_cm):
    height_ratio = real_height_cm / pixel_height
    return pixel_measurement * height_ratio


def measure_body_sizes(side_colored_mask, front_colored_mask, sideposes, frontposes, real_height_cm, rainbow):
    """Measure various body sizes based on detected poses."""
    measurements = []

    for pose in frontposes:
        # Assuming each `pose` is a dictionary with 'keypoints' that are already in the required format
        keypoints = pose[0]  # This should directly give us the dictionary

        # Extract positions directly from keypoints
        left_eye = keypoints[1].position
        right_eye = keypoints[2].position
        nose = keypoints[3].position
        right_ear = keypoints[4].position
        left_shoulder = keypoints[5].position
        right_shoulder = keypoints[6].position
        left_elbow = keypoints[7].position
        right_elbow = keypoints[8].position
        left_wrist = keypoints[9].position
        right_wrist = keypoints[10].position
        left_hip = keypoints[11].position
        right_hip = keypoints[12].position
        left_knee = keypoints[13].position
        right_knee = keypoints[14].position
        left_ankle = keypoints[15].position
        right_ankle = keypoints[16].position

        # Calculate pixel height (from the top of the head to the bottom of the ankle)
        pixel_height = euclidean_distance((left_eye.x, left_eye.y), (left_ankle.x, left_ankle.y))

        shoulder_width_cm = convert_to_real_measurements(
            euclidean_distance((left_shoulder.x, left_shoulder.y), (right_shoulder.x, right_shoulder.y)),
            pixel_height, real_height_cm
        )

        # arm_length_cm = convert_to_real_measurements(
        #     euclidean_distance((right_shoulder.x, right_shoulder.y), (right_elbow.x, right_elbow.y)),
        #     pixel_height, real_height_cm
        # ) + convert_to_real_measurements(
        #     euclidean_distance((right_elbow.x, right_elbow.y), (right_wrist.x, right_wrist.y)),
        #     pixel_height, real_height_cm
        # )

        # leg_length_cm = convert_to_real_measurements(
        #     euclidean_distance((left_hip.x, left_hip.y), (left_knee.x, left_knee.y)),
        #     pixel_height, real_height_cm
        # ) + convert_to_real_measurements(
        #     euclidean_distance((left_knee.x, left_knee.y), (left_ankle.x, left_ankle.y)),
        #     pixel_height, real_height_cm
        # )

        arm_length_cm = convert_to_real_measurements(
            euclidean_distance((left_shoulder.x, left_shoulder.y), (left_wrist.x, left_wrist.y)),
            pixel_height, real_height_cm
        )

        leg_length_cm = convert_to_real_measurements(
            euclidean_distance((left_hip.x, left_hip.y), (left_ankle.x, right_ankle.y)),
            pixel_height, real_height_cm
        )

        shoulder_to_waist_cm = convert_to_real_measurements(
            euclidean_distance((left_shoulder.x, left_shoulder.y), (left_hip.x, left_hip.y)),
            pixel_height, real_height_cm
        )

        # Calculate waist circumference using the ellipse circumference formula
        a = euclidean_distance((left_hip.x, left_hip.y), (right_hip.x, right_hip.y)) / 2
        # b = euclidean_distance((), ()) / 2

        # Use Ramanujan's approximation for the circumference of an ellipse
        # waist_circumference_px = math.pi * (3*(a + b) - math.sqrt((3*a + b)*(a + 3*b)))
        waist_circumference_cm = 90  # convert_to_real_measurements(waist_circumference_px, pixel_height, real_height_cm)

        # Convert pixel measurements to real measurements using the height ratio
        measurements.append({
            "shoulder_width_cm": shoulder_width_cm,
            "leg_length_cm": leg_length_cm,
            "arm_length_cm": arm_length_cm,
            "shoulder_to_waist_cm": shoulder_to_waist_cm,
            "height_cm": real_height_cm,
            "waist_circumference_cm": waist_circumference_cm
        })

    return measurements





os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#app.py file
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
