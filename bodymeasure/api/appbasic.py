import gradio as gr
import pandas as pd

# Sample data for the table
data = {
    "Column 1": ["A", "B", "C"],
    "Column 2": [1, 2, 3],
    "Column 3": [4.5, 5.5, 6.5]
}
df = pd.DataFrame(data)

def display_images(img1, img2):
    # Function to display images and return them as outputs
    return img1, img2, df

# Create the Gradio interface
interface = gr.Interface(
    fn=display_images,    
    inputs=[
        gr.Image(label="Upload Front Pose"),
        gr.Image(label="Upload Side Pose"),
        gr.Number(label="Enter Height (cm)")
    ],
    outputs=[
        gr.DataFrame(label="Tabular Data"),
    ],
    title="Body Sizing System Demo",
    description="Upload two images. Front View and Side View"
)

# Launch the app
interface.launch(share=False)