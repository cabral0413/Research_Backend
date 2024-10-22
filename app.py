from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image, ImageOps
from flask_cors import CORS
import io
import os

app = Flask(__name__)
CORS(app)

# Load the models once during startup
yolo_model = YOLO('models/best.pt')  # YOLO model for object detection and cropping
classification_model = YOLO('models/bestc.pt')  # Model for classification

CONFIDENCE_THRESHOLD = 0.5  # Set a confidence threshold for valid detections

@app.route('/classify', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        img = Image.open(file.stream).convert("RGB")  # Ensure it's in RGB format

        # Run YOLO for object detection
        detection_results = yolo_model(img)

        # Debugging: Print detection results
        print("Detection Results:", detection_results)

        # Check if any detections are made
        if len(detection_results) == 0 or len(detection_results[0].boxes) == 0:
            return jsonify({'message': 'No objects detected in the image.'}), 200

        # Filter to keep only the most confident detection above threshold
        best_box = max(detection_results[0].boxes, key=lambda box: box.conf)
        if best_box.conf < CONFIDENCE_THRESHOLD:
            return jsonify({'message': 'No valid snake detected. Please upload a clearer image.'}), 200

        box_coords = best_box.xyxy[0].tolist()  # Convert to list of coordinates
        print(f"Cropping coordinates: {box_coords}")  # Debugging: Print cropping coordinates

        cropped_img = img.crop(box_coords)  # Crop using bounding box coordinates

        # Add padding to the cropped image to prevent distortion
        padded_img = ImageOps.pad(cropped_img, (224, 224), method=Image.Resampling.LANCZOS)

        # Perform classification on the padded image
        classification_results = classification_model(padded_img)

        # Process the classification results
        predictions = []
        for result in classification_results:
            top_class = result.names[result.probs.top1]
            top_confidence = result.probs.top1conf.item()

            # If the classification confidence is below threshold, reject the result
            if top_confidence < CONFIDENCE_THRESHOLD:
                return jsonify({
                    'message': 'No valid snake detected. Please upload a clearer image.',
                    'class': top_class,
                    'probability': "{:.2%}".format(top_confidence)
                }), 200

            # Determine venom status
            venom_status = get_venom_status(top_class)

            # Format the probability as a percentage with two decimal points
            formatted_prob = "{:.2%}".format(top_confidence)

            # Append to predictions
            predictions.append({
                'class': top_class,
                'probability': formatted_prob,
                'venom_status': venom_status
            })

        return jsonify({'predictions': predictions}), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'An error occurred during processing.', 'details': str(e)}), 500

def get_venom_status(class_name):
    venom_status_map = {
        'Common Indian Krait': 'Venomous',
        'Python': 'Non-venomous',
        'Hump Nosed Viper': 'Venomous',
        'Green Vine Snake': 'Non-venomous',
        'Russells Viper': 'Venomous',
        'Indian Cobra': 'Venomous'
    }
    return venom_status_map.get(class_name, 'Unknown')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    