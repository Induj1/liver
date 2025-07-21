"""
Grad-CAM (Gradient-weighted Class Activation Mapping) utilities
for liver disease classification model explainability.

This module provides functions to generate and visualize Grad-CAM heatmaps
to understand which regions of liver images the model focuses on when making predictions.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """
    Grad-CAM implementation for CNN model explainability
    """
    
    def __init__(self, model, class_names, last_conv_layer_name=None):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained Keras model
            class_names: List of class names
            last_conv_layer_name: Name of last convolutional layer (auto-detected if None)
        """
        self.model = model
        self.class_names = class_names
        
        # Auto-detect last convolutional layer if not provided
        if last_conv_layer_name is None:
            last_conv_layer_name = self._find_last_conv_layer()
        
        self.last_conv_layer_name = last_conv_layer_name
        
        # Create grad model
        self.grad_model = self._create_grad_model()
        
        print(f"✅ Grad-CAM initialized with layer: {last_conv_layer_name}")
    
    def _find_last_conv_layer(self):
        """Automatically find the last convolutional layer"""
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:  # Conv layers have 4D output
                return layer.name
        
        # Fallback to common names
        common_names = ['conv5_block3_out', 'conv_pw_13', 'top_activation']
        for name in common_names:
            try:
                self.model.get_layer(name)
                return name
            except ValueError:
                continue
        
        raise ValueError("Could not find a convolutional layer")
    
    def _create_grad_model(self):
        """Create gradient model for Grad-CAM computation"""
        grad_model = keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
        )
        return grad_model
    
    def generate_gradcam(self, image, class_index=None, alpha=0.4):
        """
        Generate Grad-CAM heatmap for an image
        
        Args:
            image: Input image (preprocessed)
            class_index: Target class index (None for predicted class)
            alpha: Transparency for heatmap overlay
        
        Returns:
            tuple: (heatmap, superimposed_img, prediction_probs)
        """
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Get conv layer output and predictions
            conv_outputs, predictions = self.grad_model(image)
            
            # Use predicted class if not specified
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            
            # Get class output
            class_output = predictions[:, class_index]
        
        # Compute gradients
        gradients = tape.gradient(class_output, conv_outputs)
        
        # Global average pooling on gradients
        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
        
        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Create superimposed image
        superimposed_img = self._create_superimposed_img(
            image[0], heatmap, alpha
        )
        
        return heatmap, superimposed_img, predictions[0].numpy()
    
    def _create_superimposed_img(self, original_img, heatmap, alpha=0.4):
        """Create superimposed image with heatmap overlay"""
        # Resize heatmap to match image size
        img_size = original_img.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (img_size[1], img_size[0]))
        
        # Convert heatmap to RGB
        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
        
        # Ensure original image is in [0, 1] range
        if original_img.max() > 1:
            original_img = original_img / 255.0
        
        # Superimpose heatmap on original image
        superimposed_img = heatmap_colored * alpha + original_img * (1 - alpha)
        
        return superimposed_img
    
    def visualize_gradcam(self, image, true_label=None, save_path=None, figsize=(15, 5)):
        """
        Visualize Grad-CAM results
        
        Args:
            image: Input image
            true_label: True class label (optional)
            save_path: Path to save visualization
            figsize: Figure size
        """
        # Generate Grad-CAM
        heatmap, superimposed_img, predictions = self.generate_gradcam(image)
        
        # Get prediction details
        predicted_class_idx = np.argmax(predictions)
        predicted_class_name = self.class_names[predicted_class_idx]
        confidence = predictions[predicted_class_idx]
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        display_img = image[0] if len(image.shape) == 4 else image
        if display_img.max() > 1:
            display_img = display_img / 255.0
        
        axes[0].imshow(display_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Superimposed image
        axes[2].imshow(superimposed_img)
        title = f'Pred: {predicted_class_name} ({confidence:.3f})'
        if true_label is not None:
            title += f'\\nTrue: {true_label}'
        axes[2].set_title(title)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Grad-CAM visualization saved to: {save_path}")
        
        plt.show()
        
        # Print prediction probabilities
        print("\\n📊 Prediction Probabilities:")
        for i, (class_name, prob) in enumerate(zip(self.class_names, predictions)):
            status = "👉" if i == predicted_class_idx else "  "
            print(f"{status} {class_name}: {prob:.4f} ({prob*100:.2f}%)")
        
        return heatmap, superimposed_img, predictions
    
    def generate_gradcam_for_all_classes(self, image, save_dir=None):
        """
        Generate Grad-CAM for all classes
        
        Args:
            image: Input image
            save_dir: Directory to save individual class visualizations
        
        Returns:
            Dictionary of heatmaps for each class
        """
        results = {}
        
        fig, axes = plt.subplots(2, len(self.class_names), figsize=(4*len(self.class_names), 8))
        if len(self.class_names) == 1:
            axes = axes.reshape(2, 1)
        
        for i, class_name in enumerate(self.class_names):
            # Generate Grad-CAM for specific class
            heatmap, superimposed_img, predictions = self.generate_gradcam(image, class_index=i)
            
            results[class_name] = {
                'heatmap': heatmap,
                'superimposed_img': superimposed_img,
                'activation_strength': predictions[i]
            }
            
            # Plot heatmap
            im1 = axes[0, i].imshow(heatmap, cmap='jet')
            axes[0, i].set_title(f'{class_name}\\nHeatmap')
            axes[0, i].axis('off')
            
            # Plot superimposed image
            axes[1, i].imshow(superimposed_img)
            axes[1, i].set_title(f'Activation: {predictions[i]:.3f}')
            axes[1, i].axis('off')
            
            # Save individual visualization if directory provided
            if save_dir:
                individual_save_path = f"{save_dir}/gradcam_{class_name}.png"
                self.visualize_gradcam(image, save_path=individual_save_path)
        
        plt.tight_layout()
        plt.suptitle('Grad-CAM for All Classes', fontsize=16, y=1.02)
        
        if save_dir:
            all_classes_path = f"{save_dir}/gradcam_all_classes.png"
            plt.savefig(all_classes_path, dpi=300, bbox_inches='tight')
            print(f"💾 All classes visualization saved to: {all_classes_path}")
        
        plt.show()
        
        return results


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess image for Grad-CAM analysis
    
    Args:
        image_path: Path to image file
        target_size: Target image size
    
    Returns:
        Preprocessed image array
    """
    # Load image
    image = keras.preprocessing.image.load_img(image_path, target_size=target_size)
    image_array = keras.preprocessing.image.img_to_array(image)
    
    # Normalize to [0, 1]
    image_array = image_array / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array


def batch_gradcam_analysis(gradcam, image_paths, output_dir, true_labels=None):
    """
    Perform Grad-CAM analysis on a batch of images
    
    Args:
        gradcam: GradCAM instance
        image_paths: List of image paths
        output_dir: Output directory for visualizations
        true_labels: List of true labels (optional)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔍 Performing Grad-CAM analysis on {len(image_paths)} images...")
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        
        # Load and preprocess image
        image = load_and_preprocess_image(image_path)
        
        # Get true label if available
        true_label = true_labels[i] if true_labels else None
        
        # Generate visualization
        save_path = f"{output_dir}/gradcam_image_{i+1}.png"
        gradcam.visualize_gradcam(image, true_label=true_label, save_path=save_path)
    
    print(f"✅ Batch Grad-CAM analysis completed. Results saved to: {output_dir}")


# Example usage functions
def demo_gradcam_usage():
    """Demonstrate how to use Grad-CAM with the liver disease model"""
    print("📚 GRAD-CAM USAGE EXAMPLE:")
    print("="*50)
    print("""
# 1. Load trained model
model = keras.models.load_model('path/to/liver_disease_model.h5')

# 2. Initialize Grad-CAM
class_names = ['normal', 'cirrhosis', 'liver_cancer', 'fatty_liver', 'hepatitis']
gradcam = GradCAM(model, class_names)

# 3. Load and analyze an image
image_path = 'path/to/liver_image.jpg'
image = load_and_preprocess_image(image_path)

# 4. Generate Grad-CAM visualization
gradcam.visualize_gradcam(image, save_path='gradcam_result.png')

# 5. Analyze all classes
results = gradcam.generate_gradcam_for_all_classes(image, save_dir='gradcam_results/')
""")
    print("="*50)


if __name__ == "__main__":
    demo_gradcam_usage()
