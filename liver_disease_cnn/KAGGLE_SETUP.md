# Kaggle Dataset Setup Instructions

## 🔑 Step 1: Get Kaggle API Credentials

1. **Go to Kaggle**: https://www.kaggle.com/account
2. **Login** to your account (create one if needed)
3. **Scroll down** to the "API" section
4. **Click "Create New Token"** - this downloads `kaggle.json`
5. **Place the file** in the correct location:
   - **Windows**: `C:\Users\{username}\.kaggle\kaggle.json`
   - **Mac/Linux**: `~/.kaggle/kaggle.json`

## 📊 Recommended Liver Disease Datasets

### 🏥 Medical Image Datasets (Best for Deep Learning):

1. **Liver Tumor Segmentation**
   - Dataset: `andrewmvd/liver-tumor-segmentation`
   - Size: ~2GB
   - Contains: CT images with liver tumors
   - **BEST CHOICE** for image classification

2. **CT Medical Images**
   - Dataset: `mohamedhanyyy/ct-medical-images`
   - Size: ~1.5GB
   - Contains: Various CT scan images

3. **Medical Image Datasets**
   - Dataset: `kmader/medical-image-datasets`
   - Size: ~500MB
   - Contains: Various medical imaging data

### 📋 Clinical Data Datasets (For reference):

4. **Indian Liver Patient Records**
   - Dataset: `uciml/indian-liver-patient-records`
   - Size: Small (~1MB)
   - Contains: Clinical lab values (not images)

5. **Liver Disease Prediction**
   - Dataset: `jeevannagaraj/liver-disease-prediction`
   - Size: Very small (~500KB)
   - Contains: Clinical features (not images)

## 🚀 Quick Download Commands

Once you have `kaggle.json` set up:

```bash
# Download liver tumor dataset (RECOMMENDED)
kaggle datasets download -d andrewmvd/liver-tumor-segmentation -p downloads --unzip

# Download CT medical images
kaggle datasets download -d mohamedhanyyy/ct-medical-images -p downloads --unzip

# Download clinical data
kaggle datasets download -d uciml/indian-liver-patient-records -p downloads --unzip
```

## 🎯 What to Do After Download:

1. **Organize images** by diagnosis into:
   ```
   data/train/normal/
   data/train/cirrhosis/
   data/train/liver_cancer/
   data/train/fatty_liver/
   data/train/hepatitis/
   ```

2. **Run the setup script**: `python setup_dataset.py`

3. **Start training**: `jupyter lab notebooks/liver_disease_classification.ipynb`

## 💡 Alternative: Browse Kaggle Manually

Visit these links to explore datasets:
- https://www.kaggle.com/search?q=liver+disease+images
- https://www.kaggle.com/search?q=medical+imaging+liver
- https://www.kaggle.com/search?q=ct+scan+liver

## 🏥 Other Medical Dataset Sources

If Kaggle doesn't work:
- **Medical Segmentation Decathlon**: http://medicaldecathlon.com/
- **LiTS Challenge**: https://competitions.codalab.org/competitions/17094
- **CHAOS Challenge**: https://chaos.grand-challenge.org/
- **NIH Clinical Center**: https://www.nih.gov/about-nih/what-we-do/nih-almanac/clinical-center
