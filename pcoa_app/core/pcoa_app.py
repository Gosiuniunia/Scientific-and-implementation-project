import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
import io
from core.enums import *


class PCOAApp:
    def __init__(self):
        self.photo_uploaded = PhotoUploadStatus.NOT_UPLOADED
        self.photo_validated = PhotoValidationStatus.NOT_VALIDATED
        self.prediction_done = PredictionStatus.NOT_DONE
        self.predicted_types = {}

    def set_prediction_done(self, status: PredictionStatus):
        self.prediction_done = status

    def predict_color_type(self, image: PCOAImage) -> ColorType:
        # Here put the prediction model & logic
        return ColorType.SPRING
    
    def show_uploaded_image(self, image: PCOAImage) -> np.ndarray:
        return image.get_image()
    
    def build_ui(self):
        with gr.Blocks() as demo:
            # App title
            gr.Markdown("# Personal Color Analysis System 🎨")

            with gr.Row():
                with gr.Column():
                    # Input: Upload image
                    img_input = gr.Image(label="Upload Your Photo", type="numpy")
                    img_input.change(fn=self.show_uploaded_image, inputs=img_input)

                    # Button to trigger analysis
                    analyze_button = gr.Button("Analyze Color")
                    analyze_button.click(fn=lambda x: True, inputs=img_input, outputs=[])

                if self.prediction_done == PredictionStatus.DONE:
                    with gr.Column():
                        # Output: Display color type and example matching colors
                        gr.Markdown(f"## Your Personal Color Type is: {self.predicted_types}")
                        color_output = gr.ColorPicker(label="Dominant Color 1")
                        color_output2 = gr.ColorPicker(label="Dominant Color 2")
                        color_output3 = gr.ColorPicker(label="Dominant Color 3")
        return demo
    



