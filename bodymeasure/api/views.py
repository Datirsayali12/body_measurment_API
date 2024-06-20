from django.http import JsonResponse
from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

import os
from PIL import Image
import numpy as np
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
from tf_bodypix.draw import draw_poses
import pandas as pd

from api.forms import ImageUploadForm
from .calculations import measure_body_sizes

# Set environment variables for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Load BodyPix model
bodypix_model = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))

rainbow = [
    [110, 64, 170], [143, 61, 178], [178, 60, 178], [210, 62, 167],
    [238, 67, 149], [255, 78, 125], [255, 94, 99], [255, 115, 75],
    [255, 140, 56], [239, 167, 47], [217, 194, 49], [194, 219, 64],
    [175, 240, 91], [135, 245, 87], [96, 247, 96], [64, 243, 115],
    [40, 234, 141], [28, 219, 169], [26, 199, 194], [33, 176, 213],
    [47, 150, 224], [65, 125, 224], [84, 101, 214], [99, 81, 195]
]

class ProcessImagesView(APIView):
    #serializer=ImageUploadSerializer()
    def post(self, request, *args, **kwargs):

        front_img = request.FILES.get('front_image')
        side_img = request.FILES.get('side_image')
        real_height_cm = float(request.data.get('height_cm'))
        if not front_img or not side_img or not real_height_cm :
               return Response(data={"message":"please upload  required images and height in cm "})




        # Open the image files using PIL
        front_img = Image.open(front_img)
        side_img = Image.open(side_img)

        # Convert the PIL images to numpy arrays
        fimage_array = np.array(front_img)
        simage_array = np.array(side_img)

        # BodyPix prediction
        frontresult = bodypix_model.predict_single(fimage_array)
        sideresult = bodypix_model.predict_single(simage_array)

        front_mask = frontresult.get_mask(threshold=0.75)
        side_mask = sideresult.get_mask(threshold=0.75)

        front_colored_mask = frontresult.get_colored_part_mask(front_mask, rainbow)
        side_colored_mask = sideresult.get_colored_part_mask(side_mask, rainbow)

        frontposes = frontresult.get_poses()
        front_image_with_poses = draw_poses(
            fimage_array.copy(),
            frontposes,
            keypoints_color=(255, 100, 100),
            skeleton_color=(100, 100, 255)
        )

        sideposes = sideresult.get_poses()
        side_image_with_poses = draw_poses(
            simage_array.copy(),
            sideposes,
            keypoints_color=(255, 100, 100),
            skeleton_color=(100, 100, 255)
        )

        body_sizes = measure_body_sizes(side_colored_mask, front_colored_mask, sideposes, frontposes,
                                        real_height_cm, rainbow)
        measurements_df = pd.DataFrame([body_sizes[0]])
        print(measurements_df)

        row_dict = measurements_df.iloc[0].to_dict()

        return Response(row_dict, status=status.HTTP_200_OK)



def process_images(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            front_img = form.cleaned_data['front_image']
            side_img = form.cleaned_data['side_image']
            real_height_cm = form.cleaned_data['height_cm']

            # Open the image files using PIL
            front_img = Image.open(front_img)
            side_img = Image.open(side_img)

            # Convert the PIL images to numpy arrays
            fimage_array = np.array(front_img)
            simage_array = np.array(side_img)

            # BodyPix prediction
            frontresult = bodypix_model.predict_single(fimage_array)
            sideresult = bodypix_model.predict_single(simage_array)

            front_mask = frontresult.get_mask(threshold=0.75)
            side_mask = sideresult.get_mask(threshold=0.75)

            front_colored_mask = frontresult.get_colored_part_mask(front_mask, rainbow)
            side_colored_mask = sideresult.get_colored_part_mask(side_mask, rainbow)

            frontposes = frontresult.get_poses()
            front_image_with_poses = draw_poses(
                fimage_array.copy(),
                frontposes,
                keypoints_color=(255, 100, 100),
                skeleton_color=(100, 100, 255)
            )

            sideposes = sideresult.get_poses()
            side_image_with_poses = draw_poses(
                simage_array.copy(),
                sideposes,
                keypoints_color=(255, 100, 100),
                skeleton_color=(100, 100, 255)
            )

            body_sizes = measure_body_sizes(side_colored_mask, front_colored_mask, sideposes, frontposes, real_height_cm, rainbow)
            measurements_df = pd.DataFrame([body_sizes[0]])

            # Convert DataFrame to dictionary for rendering in template
            measurements_dict = measurements_df.to_dict(orient='records')[0]

            return render(request, 'process_images.html', {'form': form, 'measurements': measurements_dict})
        else:
            return JsonResponse({'message': 'Please upload the required images and height in cm.'}, status=400)

    else:
        form = ImageUploadForm()
    return render(request, 'process_images.html', {'form': form})


# # Add measure_body_sizes function if it's not imported from another module
# def measure_body_sizes(side_colored_mask, front_colored_mask, sideposes, frontposes, real_height_cm, rainbow):
#     measurements = []
#     for pose in frontposes:
#         keypoints = pose["keypoints"]
#         left_eye = keypoints[1]["position"]
#         right_eye = keypoints[2]["position"]
#         nose = keypoints[0]["position"]
#         left_shoulder = keypoints[5]["position"]
#         right_shoulder = keypoints[6]["position"]
#         left_wrist = keypoints[9]["position"]
#         left_hip = keypoints[11]["position"]
#         right_hip = keypoints[12]["position"]
#         left_ankle = keypoints[15]["position"]
#
#         pixel_height = euclidean_distance((left_eye["x"], left_eye["y"]), (left_ankle["x"], left_ankle["y"]))
#
#         shoulder_width_cm = convert_to_real_measurements(
#             euclidean_distance((left_shoulder["x"], left_shoulder["y"]), (right_shoulder["x"], right_shoulder["y"])),
#             pixel_height, real_height_cm
#         )
#
#         arm_length_cm = convert_to_real_measurements(
#             euclidean_distance((left_shoulder["x"], left_shoulder["y"]), (left_wrist["x"], left_wrist["y"])),
#             pixel_height, real_height_cm
#         )
#
#         leg_length_cm = convert_to_real_measurements(
#             euclidean_distance((left_hip["x"], left_hip["y"]), (left_ankle["x"], left_ankle["y"])),
#             pixel_height, real_height_cm
#         )
#
#         shoulder_to_waist_cm = convert_to_real_measurements(
#             euclidean_distance((left_shoulder["x"], left_shoulder["y"]), (left_hip["x"], left_hip["y"])),
#             pixel_height, real_height_cm
#         )
#
#         a = euclidean_distance((left_hip["x"], left_hip["y"]), (right_hip["x"], right_hip["y"])) / 2
#         waist_circumference_cm = 90  # Placeholder value
#
#         measurements.append({
#             "shoulder_width_cm": shoulder_width_cm,
#             "leg_length_cm": leg_length_cm,
#             "arm_length_cm": arm_length_cm,
#             "shoulder_to_waist_cm": shoulder_to_waist_cm,
#             "height_cm": real_height_cm,
#             "waist_circumference_cm": waist_circumference_cm
#         })
#     return measurements
#
#
# def euclidean_distance(point1, point2):
#     return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
#
#
# def convert_to_real_measurements(pixel_measurement, pixel_height, real_height_cm):
#     height_ratio = real_height_cm / pixel_height
#     return pixel_measurement * height_ratio
