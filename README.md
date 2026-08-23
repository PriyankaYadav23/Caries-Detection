 <!--  -->
 # Dental Caries Detection using Transfer Learning

This project detects dental diseases from panoramic X-ray images using a deep learning model built on PyTorch. Individual teeth are cropped from full X-rays and classified into one of 4 disease categories.

## About the Project

Built on the **DENTEX dataset** (MICCAI 2023 challenge), which contains panoramic dental X-rays with quadrant, tooth-number, and diagnosis annotations.

This project uses the fully-labeled subset — X-rays annotated with quadrant, tooth enumeration, and diagnosis classes. Individual teeth are cropped using bounding-box coordinates from the annotations, then classified into 4 disease categories:
- Caries
- Deep Caries
- Impacted
- Periapical Lesion

## Approach

- **Model:** MobileNetV2 (transfer learning, pretrained on ImageNet), with a custom 4-class classification head
- **Class imbalance handling:** the dataset is heavily imbalanced (Caries: 2189 images vs. Periapical Lesion: 158 images). Multiple class-weighting strategies were systematically tested



## Known Limitations

Minority classes (particularly Periapical Lesion, with only 158 training images) have lower recall than majority classes. This is a known limitation of the dataset size and is documented as an area for future improvement (e.g., more data collection, further augmentation strategies).