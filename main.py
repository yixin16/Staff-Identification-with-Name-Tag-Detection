import cv2
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModelWithProjection
import torch
import os
import numpy as np
from datetime import datetime


# --- ⚙️ CONFIGURATION ---
VIDEO_PATH = 'sample.mp4'
YOLO_MODEL_PATH = 'yolo11n.pt'
NAMETAG_CROP_PATH = 'nametag.jpg'

OUTPUT_DIR = 'smart_adaptive_output'
THUMBNAIL_DIR = os.path.join(OUTPUT_DIR, 'staff_thumbnails')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)

# --- TEXT & VISUAL EVIDENCE ---
TEXT_PROMPTS = [
    "a person with a small white rectangle on their chest",
    "a person with no badge or tag on their clothes"
]

try:
    nametag_template_image = Image.open(NAMETAG_CROP_PATH).convert("RGB")
except FileNotFoundError:
    print(f"❌ ERROR: Cropped nametag image '{NAMETAG_CROP_PATH}' not found.")
    exit()

# --- THRESHOLDS & WEIGHTS ---
PERSON_CONFIDENCE_THRESHOLD = 0.4
TEXT_WEIGHT = 0.3
IMAGE_WEIGHT = 0.7
STAFF_CONFIDENCE_THRESHOLD = 15.0

# Lower value means more frames will be considered "blurry".
BLUR_THRESHOLD = 100.0

def variance_of_laplacian(image):
    """Compute the Laplacian of the image and then return the focus measure,
       which is simply the variance of the Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# --- INITIALIZE MODELS ---
print("🚀 Starting Smart Adaptive Zero-Shot System...")
person_detector = YOLO(YOLO_MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

with torch.no_grad():
    template_inputs = clip_processor(images=nametag_template_image, return_tensors="pt").to(device)
    template_features = clip_model.get_image_features(**template_inputs)
    template_features /= template_features.norm(p=2, dim=-1, keepdim=True)

# --- VIDEO SETUP ---
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width, frame_height, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
output_video_path = os.path.join(OUTPUT_DIR, 'smart_adaptive_output.mp4')
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

print("\nProcessing video... Press 'q' on the display window to quit.")
# --- MAIN LOOP ---
frame_number = 0
while cap.isOpened():
    success, original_frame = cap.read()
    if not success: break
    frame_number += 1
    
    # === 🌟 NEW: SMART ADAPTIVE PRE-PROCESSING LOGIC 🌟 ===
    clarity_score = variance_of_laplacian(original_frame)
    
    # Decide whether to apply the enhancement pipeline
    if clarity_score < BLUR_THRESHOLD:
        # --- DEBUG PRINT ---
        print(f"Frame {frame_number}: Low quality detected (Clarity: {clarity_score:.2f}). Applying full enhancement pipeline.")
        
        # Run the full restoration pipeline
        denoised = cv2.fastNlMeansDenoisingColored(original_frame, None, 10, 10, 7, 15)
        hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        enhanced_v = clahe.apply(v)
        enhanced_hsv = cv2.merge([h, s, enhanced_v])
        contrast_enhanced = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        gaussian_blur = cv2.GaussianBlur(contrast_enhanced, (9, 9), 10.0)
        processed_frame = cv2.addWeighted(contrast_enhanced, 1.5, gaussian_blur, -0.5, 0)
    else:
        # --- DEBUG PRINT ---
        print(f"Frame {frame_number}: High quality frame (Clarity: {clarity_score:.2f}). Skipping enhancement.")
        processed_frame = original_frame
    # ======================================================

    detections = person_detector.predict(processed_frame, classes=[0], conf=PERSON_CONFIDENCE_THRESHOLD, verbose=False)

    # The rest of the detection/identification loop remains the same
    for box in detections[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_crop = processed_frame[y1:y2, x1:x2]
        if person_crop.size == 0: continue

        person_image = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
        inputs = clip_processor(text=TEXT_PROMPTS, images=person_image, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            text_outputs = clip_model(**inputs)
            logits_per_image_text = text_outputs.logits_per_image[0]
            patch_feats = vision_model(pixel_values=inputs['pixel_values'], output_hidden_states=True).last_hidden_state[:, 1:, :]
            patch_proj = vision_model.visual_projection(patch_feats)
            patch_proj /= patch_proj.norm(dim=-1, keepdim=True)
            similarities = (patch_proj @ template_features.T).squeeze(0)
            best_patch_similarity, best_patch_index = torch.max(similarities, 0)
            logit_scale = clip_model.logit_scale.exp()

        text_score = (logits_per_image_text[0] - logits_per_image_text[1]).item()
        image_score = (best_patch_similarity * logit_scale).item()
        final_score = (text_score * TEXT_WEIGHT) + (image_score * IMAGE_WEIGHT)

        if final_score > STAFF_CONFIDENCE_THRESHOLD:
            label = f"Staff ({final_score:.1f})"
            color = (0, 255, 0)
            thumb_crop = original_frame[y1:y2, x1:x2]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            thumb_path = os.path.join(THUMBNAIL_DIR, f"staff_{timestamp}.jpg")
            cv2.imwrite(thumb_path, thumb_crop)

            GRID_SIZE, PATCH_SIZE_224 = 7, 32
            row, col = best_patch_index.item() // GRID_SIZE, best_patch_index.item() % GRID_SIZE
            ph, pw, _ = person_crop.shape
            scale_w, scale_h = pw / 224.0, ph / 224.0
            p_x1, p_y1 = int(col * PATCH_SIZE_224 * scale_w), int(row * PATCH_SIZE_224 * scale_h)
            p_x2, p_y2 = int((col+1) * PATCH_SIZE_224 * scale_w), int((row+1) * PATCH_SIZE_224 * scale_h)
            cv2.rectangle(processed_frame, (x1 + p_x1, y1 + p_y1), (x1 + p_x2, y1 + p_y2), (0, 255, 255), 2)
        else:
            label = "Person"
            color = (255, 0, 0)

        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(processed_frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Smart Adaptive Pipeline - Live Detection", processed_frame)
    out.write(processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n'q' pressed. Exiting...")
        break

cap.release()
out.release()
cv2.destroyAllWindows()


print("\n✅ Processing complete.")

# --- FINAL REPORT ---
print(f"\n✅ Video saved to: {output_video_path}")
print("="*50)
print("📝 ZERO-SHOT AI EVALUATION REPORT")
print("="*50)