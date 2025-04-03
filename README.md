# Early-Stage Skin Cancer Detection using Deep Learning

This project focuses on building a robust image classification pipeline to detect seven types of skin lesions using the **HAM10000 dataset**. We explore various machine learning models and deep learning architectures, including **Logistic Regression, Neural Networks, a custom CNN**, and **ResNet-50 with Transfer Learning**, to compare their performance and applicability to real-world medical imaging.

---

## 📊 Project Objectives
- Classify dermatoscopic images into 7 lesion types.
- Evaluate multiple models from simple baselines to state-of-the-art CNNs.
- Explore the impact of data imbalance, augmentation, and transfer learning.

---

## 📈 Models Compared

| Model                     | Accuracy | Strengths                                      | Limitations                          |
|--------------------------|----------|------------------------------------------------|--------------------------------------|
| Logistic Regression      | ~67%     | Simple, interpretable baseline                 | Poor on image data                   |
| Fully Connected NN       | ~70%     | Learns basic patterns                          | Ignores spatial structure            |
| Custom CNN               | ~77%     | Captures spatial features, deep layers         | Some overfitting, limited generalization |
| **ResNet-50 (Best)**     | **~88%** | Pretrained, strong generalization, fine-tuned  | Needs more memory, still biased toward majority class |

---

## 📊 Dataset: HAM10000

- Source: [Kaggle - HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- 10,015 dermatoscopic images
- 7 lesion classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`

---

## 🚀 Technologies Used
- **Python, TensorFlow/Keras** for model development
- **scikit-learn** for metrics and evaluation
- **matplotlib & seaborn** for data visualization
- **Google Colab Pro with A100 GPU** for training acceleration

---

## 💼 Project Highlights
- Implemented baseline logistic regression and neural networks.
- Developed a custom 3-block CNN with BatchNormalization and Dropout.
- Leveraged **ResNet-50** pretrained on ImageNet with a custom head.
- Applied data augmentation to handle class imbalance.
- Used callbacks like **ModelCheckpoint** and **EarlyStopping**.


---

## 📊 Results Snapshot
- **Best Validation Accuracy:** ~87% (ResNet-50)
- **Key Takeaway:** ResNet-50 offers strong generalization and robustness, while custom CNNs and simpler models struggle with subtle visual distinctions.

---

## 👤 Author
**Sridevi Bommidi**  
MS in Data Science | AI for Healthcare Enthusiast  
[LinkedIn](#) | [Portfolio](#) | [Email](#)

---
