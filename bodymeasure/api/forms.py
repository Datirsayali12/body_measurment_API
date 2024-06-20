from django import forms

class ImageUploadForm(forms.Form):
    front_image = forms.ImageField(label='Upload Front Body Image')
    side_image = forms.ImageField(label='Upload Side Body Image')
    height_cm = forms.FloatField(label="upload height in cm")
