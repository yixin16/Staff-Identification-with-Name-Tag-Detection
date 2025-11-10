# 🧠 Smart Adaptive Zero-Shot Staff Identification System

An **advanced AI surveillance system** designed to identify staff members in real-time video footage using a **single sample name tag image** — with **no retraining required**.  
The system integrates **YOLOv8** for person detection, **OpenAI CLIP** for zero-shot identification, and an **adaptive video enhancement pipeline** for robust performance under varying video conditions.

---

## 🚀 Key Features

### 🔹 True Zero-Shot Identification
- No model retraining or fine-tuning required.  
- To detect a new staff name tag, simply replace `nametag.jpg`.  
- Powered by CLIP’s generalized world knowledge for universal visual and textual understanding.

### 🔹 Explainable AI (XAI) for Transparency
- Every positive identification highlights the **exact patch** (yellow rectangle) that triggered the match.  
- Users can **see what the AI sees**, improving explainability and trust.

### 🔹 Smart Adaptive Frame Enhancement
- Automatically detects **blurry or low-quality frames** using Laplacian variance.  
- Applies **denoising, contrast enhancement, and sharpening** only when needed to save resources.

### 🔹 Configurable & Modular
- Key parameters (thresholds, weights, file paths) are centralized at the top of the script.  
- Easy to tune and adapt to new environments without changing the logic.

### 🔹 Automated Evidence Logging
- Each positively identified staff member is saved as a cropped **thumbnail** under `staff_thumbnails/`.  
- Facilitates audit trails, security reviews, and model explainability.

---

## 🧩 System Architecture

```
            ┌──────────────────────────────┐
            │  Adaptive Frame Preprocessing│
            │ (Enhance blurry frames only) │
            └──────────────┬───────────────┘
                           │
                           ▼
                ┌────────────────────┐
                │  YOLOv8 Detection   │
                │ (Person detection)  │
                └──────────┬──────────┘
                           │
                           ▼
             ┌───────────────────────────┐
             │  OpenAI CLIP Verification │
             │  • Visual Patch Matching  │
             │  • Textual Semantic Match │
             └───────────┬───────────────┘
                         ▼
                ┌─────────────────────┐
                │ Weighted Fusion     │
                │ → Staff / Non-Staff │
                └─────────────────────┘
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yixin16/Staff-Identification-with-Name-Tag-Detection.git
cd Staff-Identification-with-Name-Tag-Detection
```

### 2️⃣ Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

> 💡 **Note:** A CUDA-enabled GPU is highly recommended for real-time processing performance.

---

## 🧠 How It Works

### Step 1: Prepare Inputs
- Place your **video file** in the root directory and name it:
  ```
  sample.mp4
  ```
- Crop a **sample name tag** image and save it as:
  ```
  nametag.jpg
  ```

### Step 2: Run the System
```bash
python main.py
```

### Step 3: Observe the Results
- A display window will open showing live detection results.  
- Press **`q`** to quit processing.
- Output includes:
  - `smart_adaptive_output.mp4` → Annotated video  
  - `staff_thumbnails/` → Cropped staff detections  
  - On-screen logs → Frame clarity, enhancement status, and decisions  

---

## 🧪 Core Components

| Component | Description |
|------------|-------------|
| **YOLOv8** | Detects all people in each frame. |
| **CLIP Model** | Performs dual verification via visual patch similarity and semantic textual understanding. |
| **Adaptive Frame Enhancement** | Dynamically enhances frames based on clarity score using Laplacian variance. |
| **Explainability Layer** | Highlights the most similar image patch (yellow box) to justify decisions. |
| **Evidence Logging** | Saves thumbnails of positively identified staff for later review. |

---

## 🧮 Key Parameters (Configurable)

| Parameter | Description | Default |
|------------|-------------|----------|
| `PERSON_CONFIDENCE_THRESHOLD` | Minimum YOLO confidence for person detection | `0.4` |
| `STAFF_CONFIDENCE_THRESHOLD` | Minimum weighted score to confirm "Staff" | `15.0` |
| `TEXT_WEIGHT` / `IMAGE_WEIGHT` | Weighting between CLIP textual and visual similarity | `0.3 / 0.7` |
| `BLUR_THRESHOLD` | Laplacian variance cutoff for detecting low-quality frames | `100.0` |

---

## 🧷 Output Files

| File / Folder | Description |
|----------------|-------------|
| `smart_adaptive_output.mp4` | Final processed video with detection overlays |
| `smart_adaptive_output/` | Directory containing video output |
| `smart_adaptive_output/staff_thumbnails/` | Cropped thumbnails of identified staff members |
| `nametag.jpg` | Reference sample for zero-shot identification |

---

## 🧠 Example Scenario

1. Security footage (`sample.mp4`) is provided.
2. System detects all persons using YOLOv8.
3. Each person’s cropped image is analyzed by CLIP:
   - **Visual match:** Compares each patch to the uploaded name tag.
   - **Text match:** Evaluates if the person fits the prompt "a person with a small white rectangle on their chest".
4. Both scores are fused; if above threshold → person is identified as **Staff**.
5. The system marks:
   - Green box → Staff  
   - Blue box → Non-staff  
   - Yellow box → Matched patch (visual proof)

---

## 📊 Sample Log Output

```
🚀 Starting Smart Adaptive Zero-Shot System...
Using device: cuda

Processing video... Press 'q' to quit.
Frame 12: Low quality detected (Clarity: 85.23). Applying full enhancement pipeline.
Frame 13: High quality frame (Clarity: 156.41). Skipping enhancement.

✅ Processing complete.
✅ Video saved to: smart_adaptive_output/smart_adaptive_output.mp4
```

---

## 🧰 Tech Stack

- **Python 3.10+**
- **OpenCV**
- **PyTorch**
- **Ultralytics YOLOv8**
- **OpenAI CLIP (ViT-B/32)**
- **Transformers**
- **Pillow, NumPy, datetime**

---

> *“Proof, not prediction — explainable AI for the real world.”*
