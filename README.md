# Dental Caries Detection

This project detects dental diseases from panoramic X-ray images. It has two components: an image classifier that labels a cropped tooth image, and a YOLO-based object detector that finds and labels disease regions directly on a full X-ray. Both are served through a live Streamlit demo and a FastAPI backend.

## About the Project

Built on the **DENTEX dataset** (MICCAI 2023 challenge), which contains panoramic dental X-rays with quadrant, tooth-number, and diagnosis annotations, covering 4 disease categories:
- Caries
- Deep Caries
- Impacted
- Periapical Lesion

## Approach 1: Classification (MobileNetV2)

Individual teeth are cropped from full X-rays using bounding-box annotations, then classified into one of the 4 disease categories.

- **Model:** MobileNetV2 (transfer learning, pretrained on ImageNet), with a custom 4-class classification head
- **Class imbalance handling:** the dataset is heavily imbalanced (Caries: 2189 images vs. Periapical Lesion: 158 images). Multiple class-weighting strategies were systematically tested

**Known limitation:** minority classes (particularly Periapical Lesion, with only 158 training images) have lower recall than majority classes. This is a known limitation of the dataset size and is documented as an area for future improvement (e.g., more data collection, further augmentation strategies).

## Approach 2: Detection (YOLOv8)

Rather than requiring teeth to be pre-cropped, a YOLOv8 model is trained to directly locate disease regions on a full panoramic X-ray and draw a bounding box around each one, labeled with the disease type and confidence score.

- **Model:** YOLOv8n, fine-tuned from pretrained COCO weights on the DENTEX annotations (599 train / 106 val images, converted from COCO-style bounding boxes to YOLO format)
- **Training:** 50 epochs, 640px input, trained locally on Apple Silicon (MPS backend)
- **Results:** mAP50 = 0.478, mAP50-95 = 0.31 overall. Per-class recall: Impacted 0.91, Caries 0.41, Periapical Lesion 0.32, Deep Caries 0.32 (reflects the same class imbalance noted above)

## Live Demo

Two Streamlit apps let you upload an X-ray and see predictions instantly:

```bash
streamlit run app.py             # classifier demo (single cropped tooth)
streamlit run app_detection.py   # detector demo (full X-ray, bounding boxes)
```

## API

A FastAPI backend exposes both models over HTTP:

```bash
uvicorn main:app --reload --port 8001
```

- `POST /predict` — classifier: returns the predicted disease label for a cropped tooth image
- `POST /detect` — detector: returns a list of detections, each with class, confidence, and bounding box coordinates

Interactive docs available at `/docs` once the server is running.

## Running with Docker

```bash
docker build -t caries-detection .
docker run -p 8000:8000 caries-detection
```

## Project Structure

```
├── main.py                 # FastAPI app (classifier + detector endpoints)
├── app.py                  # Streamlit demo — classifier
├── app_detection.py        # Streamlit demo — detector
├── convert_to_yolo.py      # converts DENTEX COCO annotations to YOLO format
├── train_yolo.py           # trains the YOLOv8 detector
├── data.yaml               # YOLO dataset config
├── caries_model.pth        # trained classifier weights
├── models/best.pt          # trained YOLO detector weights
└── requirements.txt
```

## Future Improvements

- Address class imbalance for the detector (currently mitigated informally by dataset size only)
- Combine both approaches: use the detector to locate teeth, then the classifier to confirm disease type
