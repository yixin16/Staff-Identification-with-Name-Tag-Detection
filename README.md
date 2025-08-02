Smart Adaptive Zero-Shot Staff Identification System
This project is an advanced AI surveillance system engineered to identify staff members in video footage in real-time. It uses a state-of-the-art, multi-stage pipeline to handle real-world challenges like variable video quality and the need for explainable decisions. The system's core strength is its ability to identify staff using just a single sample image of a name tag, without any need for model retraining.

Key Features
True Zero-Shot Capability: The system requires no re-training or fine-tuning. To identify a different name tag, one only needs to replace the nametag.jpg file. This is made possible by the generalized world knowledge embedded within the CLIP model.

Explainable AI (XAI) for User Trust: The system does not just provide a label; it provides proof. By drawing a yellow rectangle on the exact patch that produced the highest visual similarity score, it shows the user why it made a decision, making the results transparent and verifiable.

Configurable & Adaptable: Key parameters (THRESHOLDS, WEIGHTS, PATHS) are centralized at the top of the script, allowing for easy tuning and adaptation to different operational environments without altering the core logic.

Automated Evidence Collection: The system automatically saves a cropped thumbnail of every positively identified staff member to the staff_thumbnails directory, creating a log for security review.

How It Works
The system operates on an intelligent three-stage pipeline:

Adaptive Frame Pre-Processing: It first analyzes each video frame for clarity. Low-quality or blurry frames are automatically put through an enhancement process (denoising, contrast adjustment, sharpening), while high-quality frames are passed through directly to save processing time.

Person Detection (YOLOv8): A lightweight YOLOv8 model detects all instances of people in the frame, drawing bounding boxes around them.

Multi-Modal Identification (OpenAI CLIP): For each detected person, a sophisticated two-part verification occurs:

Visual Match: The person's image is broken into patches, and each patch is compared to the nametag.jpg template to find a direct visual match.

Textual Match: The person's image is compared to semantic descriptions (e.g., "a person with a white rectangle on their chest") to provide contextual understanding.

A weighted score from both checks determines if the person is "Staff".

Setup & Installation
Clone the repository:

git clone [https://github.com/yixin16/staff-detection-system.git](https://github.com/yixin16/Staff-Identification-with-Name-Tag-Detection)
cd staff-detection-system

Install dependencies: It is recommended to use a virtual environment.

pip install -r requirements.txt

(If a requirements.txt is not available, install the following manually):

pip install torch torchvision torchaudio ultralytics transformers opencv-python numpy Pillow

Note: For optimal performance, a CUDA-enabled GPU is highly recommended.

Usage
Place your video file named sample.mp4 in the root directory of the project.

Provide a name tag sample: Crop an image of the name tag you want to detect and save it as nametag.jpg in the root directory.

Run the main script:

python main.py


An output window will appear showing the live detection. Press 'q' to quit the process.
