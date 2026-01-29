# ARNet
Deepfake model detect (ARNet)


# ARNet: Adaptive Resampling Network for Video-Based Binary Classification

ARNet is a deep learning framework designed to address *class imbalance* in *video-based binary classification tasks, with a focus on **facial analysis and deepfake detection*.  
The framework introduces an *adaptive resampling strategy* that balances classes by controlling the number of frames extracted per video, while preserving all video sources.

---

## 📌 Key Features

•⁠  ⁠*Adaptive Frame Resampling* to handle class imbalance  
•⁠  ⁠*Video-Level Preservation* (no video is discarded)  
•⁠  ⁠*CNN-Based Pipeline*, compatible with AlexNet-style backbones  
•⁠  ⁠*Binary Classification Support*  
•⁠  ⁠*Imbalance-Aware Evaluation Metrics* (ROC, AUC, G-Mean)

---

## 🧠 Methodology

ARNet follows three main stages:

1.⁠ ⁠*Frame Extraction*  
   Frames are sampled from videos according to a configurable policy.

2.⁠ ⁠*Adaptive Undersampling*  
   - The class distribution is analyzed at the training stage.  
   - When imbalance is detected, the number of frames extracted from videos belonging to the *majority class* is reduced.  
   - Videos from the *minority class* retain a higher number of frames.  
   - This balances the dataset at the frame level while maintaining all video-level sources.

3.⁠ ⁠*Model Training*  
   Extracted frames are used to train a CNN-based binary classifier.

---


## 🏗️ Model Architecture

ARNet is architecture-agnostic. A typical configuration includes:

•⁠  ⁠Input: RGB facial frames  
•⁠  ⁠Backbone: AlexNet (or similar CNN)  
•⁠  ⁠Fully Connected Layers  
•⁠  ⁠Output: Sigmoid activation for binary classification  
