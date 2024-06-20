from rest_framework import serializers

class ImageUploadSerializer(serializers.Serializer):
    front_image = serializers.ImageField()
    side_image = serializers.ImageField()
    height_cm = serializers.FloatField()