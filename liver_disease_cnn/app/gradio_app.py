"""
Simple Gradio Interface for Liver Disease Classification

This provides a quick and easy web interface for liver disease diagnosis
using the trained deep learning model with Grad-CAM explanations.
"""

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import sys
from pathlib import Path

# Add project paths
sys.path.append('../gradcam')
sys.path.append('../')

# Configuration
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Normal', 'Cirrhosis', 'Liver Cancer', 'Fatty Liver', 'Hepatitis']
MODEL_PATH = "../models/liver_disease_final.h5"

# Load model
@gr.utils.create_cache
def load_model():
    """Load the trained liver disease model"""
    try:
        if Path(MODEL_PATH).exists():
            model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Model loaded successfully!")
            return model
        else:
            print("❌ Model file not found. Please train the model first.")
            return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# Simple Grad-CAM implementation
class SimpleGradCAM:
    def __init__(self, model):
        self.model = model
        # Find last convolutional layer
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                self.last_conv_layer_name = layer.name
                break
        
        # Create grad model
        self.grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(self.last_conv_layer_name).output, model.output]
        )
    
    def generate_heatmap(self, image, class_index=None):
        """Generate Grad-CAM heatmap"""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            class_output = predictions[:, class_index]
        
        gradients = tape.gradient(class_output, conv_outputs)
        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy(), predictions[0].numpy()

# Initialize model and Grad-CAM
model = load_model()
gradcam = SimpleGradCAM(model) if model else None

def predict_liver_disease(image):
    """
    Predict liver disease from uploaded image
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        tuple: (prediction_text, confidence_plot, gradcam_image)
    """
    if model is None:
        return "❌ Model not loaded. Please check model file.", None, None
    
    try:
        # Preprocess image
        if isinstance(image, np.ndarray):
            if image.max() > 1:
                image = image / 255.0
            processed_image = cv2.resize(image, IMG_SIZE)
        else:
            processed_image = image.resize(IMG_SIZE)
            processed_image = np.array(processed_image) / 255.0
        
        # Ensure 3 channels
        if len(processed_image.shape) == 2:
            processed_image = np.stack([processed_image] * 3, axis=-1)
        elif processed_image.shape[2] == 4:
            processed_image = processed_image[:, :, :3]
        
        # Predict
        input_image = np.expand_dims(processed_image, axis=0)
        predictions = model.predict(input_image, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = predictions[predicted_class_idx]
        
        # Generate prediction text
        prediction_text = f"""
# 🏥 AI Diagnosis Results

## 🎯 **Primary Diagnosis**
**{predicted_class}**
**Confidence: {confidence:.1%}**

## 📊 **All Probabilities:**
"""
        
        for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, predictions)):
            emoji = "🔴" if i == predicted_class_idx else "⚪"
            prediction_text += f"{emoji} **{class_name}**: {prob:.1%}\\n"
        
        # Medical interpretation
        interpretations = {
            'Normal': "✅ **Healthy liver tissue detected.** No significant pathology observed.",
            'Cirrhosis': "⚠️ **Cirrhosis detected.** Advanced liver scarring present. Medical attention required.",
            'Liver Cancer': "🚨 **Liver cancer detected.** Hepatocellular carcinoma identified. Urgent consultation needed.",
            'Fatty Liver': "💛 **Fatty liver detected.** Hepatic steatosis present. Lifestyle changes recommended.",
            'Hepatitis': "🔥 **Hepatitis detected.** Liver inflammation present. Further testing required."
        }
        
        prediction_text += f"\\n## 🩺 **Interpretation:**\\n{interpretations.get(predicted_class, 'Unknown condition.')}"
        
        prediction_text += """

⚠️ **IMPORTANT:** This is for research/educational purposes only. 
Always consult medical professionals for actual diagnosis.
"""
        
        # Create confidence plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(CLASS_NAMES, predictions)
        
        # Color bars
        for i, bar in enumerate(bars):
            if i == predicted_class_idx:
                bar.set_color('red')
                bar.set_alpha(0.8)
            else:
                bar.set_color('lightblue')
                bar.set_alpha(0.6)
        
        ax.set_xlabel('Confidence')
        ax.set_title('Liver Disease Classification Probabilities')
        ax.set_xlim(0, 1)
        
        # Add percentage labels
        for bar, prob in zip(bars, predictions):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{prob:.1%}', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Generate Grad-CAM
        gradcam_image = None
        if gradcam:
            try:
                heatmap, _ = gradcam.generate_heatmap(processed_image)
                
                # Resize heatmap and create overlay
                heatmap_resized = cv2.resize(heatmap, IMG_SIZE[::-1])
                heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
                
                # Create overlay
                alpha = 0.4
                gradcam_image = heatmap_colored * alpha + processed_image * (1 - alpha)
                gradcam_image = (gradcam_image * 255).astype(np.uint8)
                
            except Exception as e:
                print(f"Grad-CAM error: {e}")
                gradcam_image = (processed_image * 255).astype(np.uint8)
        else:
            gradcam_image = (processed_image * 255).astype(np.uint8)
        
        return prediction_text, fig, gradcam_image
        
    except Exception as e:
        error_msg = f"""
# ❌ Analysis Error

**Error:** {str(e)}

**Please check:**
- Image format (JPG, PNG supported)
- Image quality and resolution
- Ensure it's a medical liver image
"""
        return error_msg, None, None

# Create Gradio interface
def create_interface():
    """Create the Gradio interface"""
    
    interface = gr.Interface(
        fn=predict_liver_disease,
        inputs=[
            gr.Image(
                type="numpy",
                label="📋 Upload Liver Medical Image",
                height=400
            )
        ],
        outputs=[
            gr.Markdown(label="🏥 AI Diagnosis Results"),
            gr.Plot(label="📊 Confidence Scores"),
            gr.Image(label="🔍 AI Focus Areas (Grad-CAM)", height=300)
        ],
        title="🏥 AI Liver Disease Classification System",
        description="""
### 🎯 **Automated Liver Disease Detection**

Upload a liver medical image (CT scan, MRI, or Ultrasound) to receive:
- **AI-powered diagnosis** with confidence scores
- **Visual explanation** showing which areas the AI focused on
- **Clinical interpretation** with medical context

**⚠️ IMPORTANT DISCLAIMER:** This system is for **research and educational purposes only**. 
It is NOT intended for clinical diagnosis. Always consult qualified medical professionals.

**📋 Supported Images:** CT scans, MRI, Ultrasound | **Formats:** JPG, PNG, JPEG
        """,
        examples=[
            # Add example images here if available
            # ["examples/normal_liver.jpg"],
            # ["examples/cirrhosis_liver.jpg"],
        ],
        theme=gr.themes.Soft(),
        allow_flagging="never",
        analytics_enabled=False,
        css="""
        .gradio-container {
            max-width: 1200px;
            margin: auto;
        }
        .gr-button {
            background: linear-gradient(45deg, #2196F3, #21CBF3);
            border: none;
            color: white;
        }
        .gr-button:hover {
            background: linear-gradient(45deg, #1976D2, #00BCD4);
        }
        """
    )
    
    return interface

# Main execution
if __name__ == "__main__":
    print("🚀 Starting Liver Disease AI Classification System...")
    
    if model is None:
        print("❌ Cannot start interface - model not loaded")
        print("Please ensure the model file exists at:", MODEL_PATH)
        exit(1)
    
    # Create and launch interface
    app = create_interface()
    
    print("✅ Interface created successfully!")
    print("🌐 Launching web application...")
    
    # Launch configuration
    app.launch(
        server_name="127.0.0.1",  # Local access
        server_port=7860,         # Default Gradio port
        share=False,              # Set to True for public sharing
        show_error=True,          # Show detailed errors
        quiet=False,              # Show startup logs
        inbrowser=True,           # Auto-open browser
        debug=False               # Debug mode
    )
