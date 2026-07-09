# Person Photo Validator

Lightweight photo validation pipeline using:

- **SCRFD 2.5G ONNX** for face detection
- **MediaPipe Pose** for body landmarks
- Rule-based quality checks for blur, brightness, contrast, crop, and body visibility

It accepts:

- `.jpg`
- `.jpeg`
- `.png`
- `.heic`
- `.heif`

It returns one of:

- `acceptable`
- `manual_verification`
- `rejected`

---

## What it checks

1. Is there a face/person?
2. Is exactly one face detected?
3. Is the image clear enough?
4. Is lighting acceptable?
5. Is the face not too close or too small?
6. Is more than head and shoulders visible using pose landmarks?
7. Is the image a real photograph, or is it suspicious of being AI-generated/synthetic?

---

## Project structure

```text
person_photo_validator/
├── app.py
├── config.py
├── download_model.py
├── requirements.txt
├── README.md
├── models/
│   ├── scrfd_2.5g_bnkps.onnx          # Face detector model (ONNX)
│   └── ai_detector_mobilenetv3.tflite # Optional AI suspicion model (TFLite)
├── src/
│   ├── image_loader.py
│   ├── face_detector_scrfd.py
│   ├── body_pose_checker.py
│   ├── body_crop_checker.py
│   ├── quality_checker.py
│   ├── ai_detector.py                # MobileNetV3 TFLite classifier
│   ├── geometry.py
│   ├── decision_engine.py
│   └── validator.py
├── training/                         # Model training pipelines
│   ├── train_ai_detector.py          # MobileNetV3 Keras trainer
│   ├── convert_to_tflite.py          # Keras to TFLite converter
│   └── ai_detector_dataset/          # Local training datasets (ignored in git)
├── test_images/
│   ├── acceptable/
│   ├── manual_verification/
│   └── rejected/
└── outputs/
    └── results.json
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

---

## Add SCRFD model

Place the model here:

```text
models/scrfd_2.5g_bnkps.onnx
```

Use a SCRFD 2.5G KPS/BNKPS ONNX model. The KPS version is preferred because it includes face keypoints.

You can manually download it from a trusted model source and rename it to:

```text
scrfd_2.5g_bnkps.onnx
```

Or use:

```bash
python download_model.py --url "PASTE_DIRECT_ONNX_URL_HERE"
```

---

## Train and Add AI Detector Model (Optional)

To enable detection of synthetic/AI-generated images, prepare a dataset and train the MobileNetV3Small binary classifier.

### 1. Prepare Dataset
Organize real and synthetic images inside the `training/ai_detector_dataset/real-vs-fake/` folder:
* `train/real/` (Real photos for training)
* `train/fake/` (Synthetic/AI-generated images for training)
* `valid/real/` (Real photos for validation)
* `valid/fake/` (Synthetic/AI-generated images for validation)

### 2. Train the Model
Run the Keras training script. Use the `--limit` flag to limit how many images per class to load for local training (to complete quickly on a local CPU/GPU):
```bash
python training/train_ai_detector.py --dataset-dir training/ai_detector_dataset/real-vs-fake --limit 2000 --epochs-init 10 --epochs-fine 5
```
*This will train the custom classification head, fine-tune the final layers, and save the output to `models/ai_detector_mobilenetv3.keras`.*

### 3. Convert Model to TFLite
Convert and optimize the Keras model to a lightweight TFLite binary:
```bash
python training/convert_to_tflite.py --input models/ai_detector_mobilenetv3.keras --output models/ai_detector_mobilenetv3.tflite --quantize
```
*Once the file `models/ai_detector_mobilenetv3.tflite` is created, the validator automatically loads it and runs the AI generated image checks.*

---

## Run on one image

```bash
python app.py test_images/acceptable/sample.jpg --pretty
```

Output example:

```json
{
  "flag": "acceptable",
  "reasons": [
    "One clear face detected",
    "Pose landmarks show more than head and shoulders"
  ],
  "warnings": [],
  "scores": {
    "image_width": 900,
    "image_height": 1200,
    "blur_score": 155.82,
    "brightness": 128.44,
    "contrast": 49.31,
    "face_count": 1,
    "body_status": "more_than_head_shoulders",
    "body_confidence": 0.9,
    "face_height_ratio": 0.21,
    "space_below_face_ratio": 0.62,
    "face_confidence": 0.93,
    "ai_suspicion_score": 0.1824
  }
}
```

## Run on a folder (JSON or CSV output)

You can process all images in a folder at once and save the results as either a JSON file or a CSV spreadsheet based on the file extension of your `--output` path.

**Export as JSON:**
```bash
python app.py test_images --output outputs/results.json
```

**Export as CSV:**
```bash
python app.py test_images --output outputs/results.csv
```

---

## Main thresholds

Edit `config.py`.

```python
face_confidence_min = 0.65
blur_reject_threshold = 60.0
blur_manual_threshold = 100.0
brightness_min = 40.0
brightness_max = 220.0
max_face_height_ratio = 0.45
min_space_below_face_reject = 0.30
min_space_below_face_manual = 0.45
pose_min_visibility = 0.50

# AI suspicion thresholds
ai_suspicion_threshold_manual = 0.50
ai_suspicion_threshold_strict = 0.75
ai_suspicion_threshold_reject = 0.90
```

---

## Meaning of flags

### acceptable

Returned when:

- exactly one clear face is detected
- image quality is good
- pose landmarks show more than head and shoulders
- AI generated suspicion score is low (< 0.50)

### manual_verification

Returned when:

- face is detected but confidence is low
- image is slightly blurry
- lighting/contrast is questionable
- body visibility is unclear
- multiple faces are detected
- AI generated suspicion score is medium or high (0.50 to 0.90)

### rejected

Returned when:

- no face is detected
- image is too blurry
- face is too close
- only head/shoulders are visible
- body landmarks do not support upper-body visibility
- AI generated suspicion score is extremely high (>= 0.90)

---

## Notes

- First test with at least 100 images per category.
- Tune thresholds using your real business examples.
- For production, log `scores` and `debug` fields so you can understand why a photo was rejected.
- For strict onboarding/KYC-style validation, keep `manual_verification` as a safe middle class instead of forcing uncertain photos into accepted/rejected.
