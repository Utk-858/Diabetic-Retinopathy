# 🩺 Diabetic Retinopathy Detection using Deep Learning

This project uses a Convolutional Neural Network (CNN) trained on retina fundus images to detect and classify Diabetic Retinopathy severity. The model is deployed as an interactive web application using **Streamlit**.

---

## 🎯 Project Goal

To build a computer-aided screening system that can automatically detect Diabetic Retinopathy (DR) early and assist ophthalmologists in diagnosis.

---

## 🔬 Classification Categories

The model classifies DR severity into the following **5 levels**:

| Label | Category |
|------|----------|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

---

## 🧠 Model Architecture

- Framework: **PyTorch**
- Model Type: **Custom CNN**
- Input Image Size: **224 × 224**
- Output: **5 classes**
- Dataset Preprocessing: Resize + Normalization

---

## 🧪 Dataset

Gaussian filtered retina images from Kaggle:

> Diabetic Retinopathy 224×224 Gaussian Filtered Dataset

---

## 🚀 Deployment

The trained model is deployed using Streamlit.

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open in browser:

📌 http://localhost:8501/

---

## 📁 Project Structure

```
📂 Diabetic-Retinopathy
├── app.py               # Streamlit App
├── model.pth            # Trained PyTorch Model
├── labels.json          # Class label mappings
├── requirements.txt     # Project dependencies
└── README.md
```

---

## 🏗 How it Works

1. Upload a retina fundus image
2. Model preprocesses image to 224×224
3. CNN predicts DR class probability
4. App displays:
   - Final predicted category
   - Confidence scores

---

## 🛠 Future Improvements

- Add Grad-CAM based heatmaps 🔥
- Use transfer learning models like EfficientNet
- Include Explainable AI for clinical support
- Improve accuracy with more data augmentation

---

## 👨‍💻 Author

**Utkarsh Bansal**
**Shulin Agarwal**

---

## ⭐ Contribute

Feel free to fork this repository, raise issues, and submit pull requests to enhance the model or UI.

