import os
import cv2
import numpy as np
import re
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from rembg import remove
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
from ultralytics import YOLO

app = Flask(__name__)

# Core configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure Tesseract path gracefully
DEFAULT_TESSERACT_PATH = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
if os.path.exists(DEFAULT_TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT_PATH

# Helper to resolve file paths dynamically
def get_path(filename):
    direct_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(direct_path):
        return direct_path
    pan_subfolder_path = os.path.join(BASE_DIR, 'PAN', filename)
    if os.path.exists(pan_subfolder_path):
        return pan_subfolder_path
    return direct_path

# Load YOLOv8 model safely
MODEL_PATH = get_path('best (5).pt')
TEMPLATE_PATH = get_path('pan.jpg')

model = None
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
else:
    print(f"Warning: YOLOv8 model weights not found at {MODEL_PATH}")

# Core logic functions
def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def remove_background(image_path):
    input_image = Image.open(image_path)
    output_image = remove(input_image)
    output_np = np.array(output_image)
    
    if output_np.shape[2] == 4:
        trans_mask = output_np[:, :, 3] == 0
        bgr = cv2.cvtColor(output_np, cv2.COLOR_RGBA2BGR)
        bgr[trans_mask] = [255, 255, 255] # Mask transparent to white
        return bgr
    return output_np

def get_detected_objects(image, threshold=0.4):
    if model is None:
        return []
    
    results = model(image)[0]
    detected_objects = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            detected_objects.append({
                'class': results.names[int(class_id)],
                'confidence': float(score),
                'coordinates': [int(x1), int(y1), int(x2), int(y2)],
                'relative_coordinates': {
                    'x1': float(x1 / image.shape[1]),
                    'y1': float(y1 / image.shape[0]),
                    'x2': float(x2 / image.shape[1]),
                    'y2': float(y2 / image.shape[0])
                }
            })
    return detected_objects

def normalize_bbox(bbox_coords, width, height):
    return [
        bbox_coords[0] / width,
        bbox_coords[1] / height,
        bbox_coords[2] / width,
        bbox_coords[3] / height
    ]

def compare_layouts(user_objects, template_objects, user_w, user_h, temp_w, temp_h):
    if not user_objects or not template_objects:
        return 0.0

    similar_count = 0
    for u_obj in user_objects:
        for t_obj in template_objects:
            if u_obj['class'] == t_obj['class']:
                # Compute Cosine Similarity between normalized bounding boxes
                u_norm = normalize_bbox(u_obj['coordinates'], user_w, user_h)
                t_norm = normalize_bbox(t_obj['coordinates'], temp_w, temp_h)
                
                sim = cosine_similarity([u_norm], [t_norm])[0][0]
                if sim > 0.90:
                    similar_count += 1
                    break
    
    # Return similarity ratio
    return float(similar_count / len(template_objects)) if template_objects else 0.0

def extract_ocr_data(image):
    extracted_text = {}
    
    # Try calling Tesseract OCR
    try:
        raw_text = pytesseract.image_to_string(image, config='--psm 6')
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        
        extracted_text['raw'] = lines
        
        # Regex search for Indian PAN Card pattern (5 uppercase letters, 4 digits, 1 uppercase letter)
        pan_regex = r'[A-Z]{5}\d{4}[A-Z]'
        pan_matches = re.findall(pan_regex, raw_text)
        if pan_matches:
            extracted_text['pan_number'] = pan_matches[0]
            
        # Extract potential DOB (DD/MM/YYYY)
        dob_regex = r'\b\d{2}/\d{2}/\d{4}\b'
        dob_matches = re.findall(dob_regex, raw_text)
        if dob_matches:
            extracted_text['dob'] = dob_matches[0]

        # Extract name heuristics
        for line in lines:
            if 'INCOME TAX' in line.upper() or 'GOVT' in line.upper() or 'INDIA' in line.upper():
                continue
            # Usually Name is one of the early uppercase blocks
            if re.match(r'^[A-Z\s]+$', line) and len(line) > 3 and 'pan_number' not in extracted_text:
                if 'name' not in extracted_text:
                    extracted_text['name'] = line
                elif 'father_name' not in extracted_text:
                    extracted_text['father_name'] = line

    except Exception as e:
        print(f"OCR Error: {e}")
        extracted_text['error'] = "Tesseract OCR engine is not configured locally."
        
    return extracted_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filename = secure_filename(file.filename)
    original_save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_' + filename)
    file.save(original_save_path)

    # 1. Normalize image file format to RGB via Pillow
    try:
        img_pillow = Image.open(original_save_path)
        img_pillow = img_pillow.convert('RGB')
        normalized_filename = filename.rsplit('.', 1)[0] + '.jpg'
        normalized_path = os.path.join(app.config['UPLOAD_FOLDER'], 'normalized_' + normalized_filename)
        img_pillow.save(normalized_path, 'JPEG')
    except Exception as e:
        return jsonify({'error': f'Failed to process file format: {str(e)}'}), 500

    # Load OpenCV images
    original_cv = cv2.imread(normalized_path)
    h, w, _ = original_cv.shape

    # 2. Preprocess / Background removal
    try:
        bg_removed = remove_background(normalized_path)
        bg_removed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'bgremoved_' + normalized_filename)
        cv2.imwrite(bg_removed_path, bg_removed)
    except Exception as e:
        print(f"Background removal error: {e}")
        bg_removed = original_cv.copy()
        bg_removed_path = normalized_path # fallback

    # Calculate brightness values
    brightness_original = calculate_brightness(original_cv)
    brightness_processed = calculate_brightness(bg_removed)

    # 3. YOLOv8 element detection
    detected_objects = get_detected_objects(original_cv)
    
    # Draw boxes for UI rendering
    detected_cv = original_cv.copy()
    for obj in detected_objects:
        coords = obj['coordinates']
        cv2.rectangle(detected_cv, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 3)
        cv2.putText(detected_cv, f"{obj['class']} ({int(obj['confidence'] * 100)}%)", 
                    (coords[0], coords[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    detected_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + normalized_filename)
    cv2.imwrite(detected_path, detected_cv)

    # 4. Cosine similarity layout checking against template
    similarity_score = 0.0
    is_valid_pan = False
    
    if os.path.exists(TEMPLATE_PATH) and model is not None:
        template_cv = cv2.imread(TEMPLATE_PATH)
        temp_h, temp_w, _ = template_cv.shape
        template_objects = get_detected_objects(template_cv)
        similarity_score = compare_layouts(detected_objects, template_objects, w, h, temp_w, temp_h)
        # We consider it a structural match if layout similarity exceeds 50%
        if similarity_score >= 0.50:
            is_valid_pan = True
    else:
        # Fallback heuristic if template isn't available
        if any(obj['class'] == 'panNo' for obj in detected_objects):
            is_valid_pan = True
            similarity_score = 0.75

    # 5. Extract OCR Data
    ocr_results = extract_ocr_data(bg_removed)
    
    # Extra validation check for PAN structure
    extracted_pan = ocr_results.get('pan_number', None)
    if extracted_pan:
        is_valid_pan = True
    else:
        # Check if we got any class label matching 'panNo' with high confidence
        pan_box = next((obj for obj in detected_objects if obj['class'] == 'panNo'), None)
        if pan_box and pan_box['confidence'] > 0.60:
            is_valid_pan = True

    # 6. Compute SSIM index (if template is available and shapes match/can be aligned)
    ssim_val = 0.0
    if os.path.exists(TEMPLATE_PATH):
        try:
            template_gray = cv2.cvtColor(cv2.imread(TEMPLATE_PATH), cv2.COLOR_BGR2GRAY)
            user_gray = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2GRAY)
            # Resize user to template size for metrics
            user_gray_resized = cv2.resize(user_gray, (template_gray.shape[1], template_gray.shape[0]))
            ssim_val, _ = ssim(template_gray, user_gray_resized, full=True)
            ssim_val = float(ssim_val)
        except Exception as e:
            print(f"SSIM error: {e}")

    # Format result payload
    response_data = {
        'original_url': f'/static/uploads/original_{filename}',
        'normalized_url': f'/static/uploads/normalized_{normalized_filename}',
        'bg_removed_url': f'/static/uploads/bgremoved_{normalized_filename}',
        'detected_url': f'/static/uploads/detected_{normalized_filename}',
        'brightness': {
            'original': brightness_original,
            'processed': brightness_processed
        },
        'layout_similarity': similarity_score,
        'ssim_score': ssim_val,
        'is_valid_pan': is_valid_pan,
        'detected_classes': [obj['class'] for obj in detected_objects],
        'detected_objects': detected_objects,
        'ocr': ocr_results
    }

    return jsonify(response_data)

if __name__ == '__main__':
    # Runs the local development server
    app.run(debug=True, port=5000)
