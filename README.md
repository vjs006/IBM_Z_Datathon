---

# Image Classification using Pre-trained CNN Model

This project demonstrates how to load a trained convolutional neural network (CNN) model, preprocess images, and make predictions for image classification tasks. It uses Keras and TensorFlow to predict whether an input image is one of the following classes: **Benign**, **Malignant**, or **Normal** (these labels can be customized based on your dataset).

## Features

- Loads an image, preprocesses it (resizes, normalizes), and runs it through the model for prediction.
- Displays the image alongside the predicted class and confidence score.
- Easy to use and modify for different image classification tasks.

## Prerequisites

Before you start, make sure you have the following dependencies installed:

- TensorFlow
- NumPy
- Matplotlib

To install the required libraries, run:

```bash
pip install tensorflow numpy matplotlib
```

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/image-classification-cnn.git
   cd image-classification-cnn
   ```

2. **Load and use your pre-trained model:**

   Ensure your model is already trained and loaded in the `model` variable in the script.

3. **Run the prediction script:**

   ```python
   import numpy as np
   from tensorflow.keras.preprocessing import image
   import matplotlib.pyplot as plt

   # Class labels (replace with your actual class names)
   class_labels = ['benign', 'malignant', 'normal']

   def predict_image(img_path):
       # Load and preprocess the image
       img = image.load_img(img_path, target_size=(150, 150))  # Resize to match model input size
       img_array = image.img_to_array(img)  # Convert to numpy array
       img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
       img_array /= 255.0  # Normalize the image

       # Make the prediction
       prediction = model.predict(img_array)

       # Get the class with the highest probability
       predicted_class = np.argmax(prediction, axis=1)[0]
       predicted_label = class_labels[predicted_class]

       # Print and show the prediction
       print(f"Predicted: {predicted_label} (Confidence: {np.max(prediction)*100:.2f}%)")

       # Display the image
       plt.imshow(img)
       plt.title(f"Prediction: {predicted_label}")
       plt.axis('off')
       plt.show()

   # Example usage:
   img_path = 'path_to_your_test_image.jpg'  # Replace with the actual path to your image
   predict_image(img_path)
   ```

4. **Modify the class labels (if needed):**

   If you're using different classes in your model, update the `class_labels` array accordingly.

5. **Predict an image:**

   You can predict any image by providing the correct file path to the `img_path` variable.

   ```python
   img_path = 'path_to_your_test_image.jpg'  # Replace with your image path
   predict_image(img_path)
   ```

## Output

The script will output:

- The predicted class.
- Confidence score of the prediction.
- A visual display of the image with the prediction as the title.

## Example

```
Predicted: Benign (Confidence: 92.35%)
```

![Predicted Image](path_to_image)

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improving this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---
