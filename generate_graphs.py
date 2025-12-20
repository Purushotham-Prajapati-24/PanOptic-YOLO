import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend for automated graph generation
import matplotlib.pyplot as plt
from rembg import remove
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, 'static', 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

# Helper to find file
def get_path(filename):
    direct_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(direct_path):
        return direct_path
    sub_path = os.path.join(BASE_DIR, 'PAN', filename)
    if os.path.exists(sub_path):
        return sub_path
    return direct_path

def main():
    print("Starting automated graph & output generation...")
    input_path = get_path('pan.jpg')
    model_path = get_path('best (5).pt')
    
    if not os.path.exists(input_path):
        print(f"Error: Template image not found at {input_path}")
        return
        
    print(f"Loading template from: {input_path}")
    image = cv2.imread(input_path)
    
    # 1. Grayscale Histogram Plot
    print("Generating grayscale histogram...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    plt.figure(figsize=(8, 5))
    plt.style.use('dark_background')
    plt.plot(hist, color='#10b981', linewidth=2.5)
    plt.title('Grayscale Pixel Intensity Distribution', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Pixel Value (0 = Black, 255 = White)', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(ASSETS_DIR, 'histogram.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Grayscale histogram saved.")

    # 2. Background isolation using rembg
    print("Removing background...")
    try:
        pillow_img = Image.open(input_path)
        output_pillow = remove(pillow_img)
        bg_removed = np.array(output_pillow)
        if bg_removed.shape[2] == 4:
            trans_mask = bg_removed[:, :, 3] == 0
            bgr = cv2.cvtColor(bg_removed, cv2.COLOR_RGBA2BGR)
            bgr[trans_mask] = [255, 255, 255]
            bg_removed = bgr
    except Exception as e:
        print(f"Rembg background removal failed, falling back to original image: {e}")
        bg_removed = image.copy()
        
    cv2.imwrite(os.path.join(ASSETS_DIR, 'bg_removed.jpg'), bg_removed)
    print("Background-removed card saved.")

    # 3. HSV Profile Plotting
    print("Generating Hue & Saturation channel mappings...")
    hsv_image = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2HSV)
    hue, saturation, _ = cv2.split(hsv_image)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0b0f19')
    
    for ax in axes:
        ax.set_facecolor('#0b0f19')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        
    axes[0].imshow(hue, cmap='hsv')
    axes[0].set_title('Hue Channel (Color Frequency)', fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(saturation, cmap='plasma')
    axes[1].set_title('Saturation Channel (Purity)', fontsize=11, fontweight='bold')
    axes[1].axis('off')
    
    # Convert BGR of bg_removed to RGB for matplotlib
    bg_removed_rgb = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2RGB)
    axes[2].imshow(bg_removed_rgb)
    axes[2].set_title('Isolated RGB Reference', fontsize=11, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'hsv_profile.png'), dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("HSV profile saved.")

    # 4. Adaptive Thresholding and Contours Only
    print("Generating contour maps...")
    gray_bg = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_bg, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_contour_area = 1000
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    contour_img = bg_removed.copy()
    if filtered_contours:
        cv2.drawContours(contour_img, filtered_contours, -1, (0, 255, 0), 5)
        
    cv2.imwrite(os.path.join(ASSETS_DIR, 'contours.jpg'), contour_img)
    print("Contour maps saved.")

    # 5. YOLOv8 Element Detections
    if os.path.exists(model_path):
        print("Running YOLOv8 element detection...")
        yolo_model = YOLO(model_path)
        frame = cv2.resize(image, (420, 640))
        results = yolo_model(frame)[0]
        
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > 0.4:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
                            
        cv2.imwrite(os.path.join(ASSETS_DIR, 'yolov8_detections.jpg'), frame)
        print("YOLOv8 element detection output saved.")
    else:
        print("Warning: YOLOv8 model weights not found. Skipping detection output.")

    print("\nAll assets generated successfully!")

if __name__ == '__main__':
    main()
