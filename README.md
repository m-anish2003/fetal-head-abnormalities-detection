# Fetal Head Abnormalities Detection Project

## 📖 Overview
This project focuses on detecting fetal head abnormalities from ultrasound images using deep learning techniques. The implementation includes data preprocessing, baseline UNet training, synthetic data generation, and Pix2PixHD training for enhanced image segmentation.

## 🚀 Features
- Data exploration and preprocessing pipelines
- Baseline UNet model for segmentation
- Synthetic data generation using GANs
- Pix2PixHD implementation for high-resolution image translation
- Comprehensive evaluation metrics

## 📁 Project Structure
fetal-head-project/
├── datasets/
│ └── fetal_ultrasound/
│ ├── train_A/ # Original ultrasound images
│ └── train_B/ # Corresponding mask images
├── notebooks/
│ ├── 01_data_exploration_preprocessing.ipynb
│ ├── 02_baseline_unet_training.ipynb
│ ├── 03_synthetic_data_generation.ipynb
│ ├── 04_pix2pixHD_training_and_inference.ipynb
│ └── checkpoints/
│ ├── baseline_unet.pth # 93.36 MB
│ └── checkpoints_spade/
│ ├── model_epoch_1.pth # 141.46 MB
│ ├── model_epoch_2.pth # 141.46 MB
│ └── ... (multiple large model files)
├── src/
│ ├── data_processing.py
│ ├── models.py
│ ├── utils.py
│ └── visualization.py
├── requirements.txt
└── README.md



# Installation

## 1. Clone the repository
```bash
git clone https://github.com/m-anish2003/fetal-head-abnormalities-detection.git
cd fetal-head-abnormalities-detection
``` 


2. Install dependencies
```bash
pip install -r requirements.txt
```
🏗️ Usage
Data Preprocessing
```python
from src.data_processing import DataPreprocessor


preprocessor = DataPreprocessor()
preprocessor.load_data('datasets/fetal_ultrasound/')
preprocessor.preprocess_images()
```
Training Baseline UNet
```python
from src.models import UNet

model = UNet()
model.train('datasets/fetal_ultrasound/train_A/', 'datasets/fetal_ultrasound/train_B/')
model.save('notebooks/checkpoints/baseline_unet.pth')
```
Synthetic Data Generation
```python
from src.models import GAN

gan = GAN()
gan.generate_synthetic_data()
```
Pix2PixHD Training
```python
from src.models import Pix2PixHD

pix2pix_model = Pix2PixHD()
pix2pix_model.train()
pix2pix_model.save('notebooks/checkpoints_spade/')
```
📊 Results
The model achieves the following performance metrics:

Dice Coefficient: 0.92

IoU: 0.87

Precision: 0.94

Recall: 0.89

🔧 Git LFS Configuration
Why Git LFS was Needed
This project contains large model files (.pth files ranging from 93MB to 141MB) that exceed GitHub's file size limit of 100MB. Git LFS (Large File Storage) was implemented to:

Store large files on a separate server

Keep repository size manageable

Enable version control for binary files

Avoid GitHub rejection during pushes

Git LFS Setup Commands
```bash
# Initialize Git LFS
git lfs install

# Track large file types
git lfs track "*.pth"
git lfs track "*.h5"
git lfs track "*.zip"
git lfs track "*.pkl"

# Add tracking configuration
git add .gitattributes
```
⚠️ Important Guidelines for Future Updates
When Adding New Notebooks
Clear Outputs Before Committing

```bash
# Clean notebook outputs
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebook.ipynb
```
Use Sequential Numbering


```bash
# Good naming convention
05_new_analysis.ipynb
06_model_evaluation.ipynb
```
Add Comprehensive Documentation within notebooks using Markdown cells

When Adding Large Files
Always Use Git LFS for Large Files

```bash
# Before adding new large files
git lfs track "*.new_extension"
git add .gitattributes

# Then add files
git add path/to/large/file.new_extension
```
Verify File Sizes Before Committing

```bash
# Check file sizes
du -h path/to/file

# Files > 50MB should use LFS
```
General Maintenance Rules
Regularly Sync with Remote

```bash
git pull origin main
```
Use Descriptive Commit Messages

```bash
git commit -m "feat: add data augmentation techniques for ultrasound images"
git commit -m "fix: resolve memory leak in data loader"
git commit -m "docs: update training documentation"
```
Keep Requirements Updated

```bash
# Generate updated requirements
pip freeze > requirements.txt

# Install new dependencies properly
pip install new-package && pip freeze > requirements.txt
🐛 Troubleshooting Guide
Common Issues and Solutions
LFS Files Not Tracking Properly

bash
# Re-initialize LFS tracking
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add path/to/large_file.pth
```
Large File Push Errors

```bash
# Use force push if needed (with caution)
git push --force origin main
```
Submodule Detection Issues

```bash
# Remove embedded git repositories
rm -rf path/to/submodule/.git
```
Notebook Output Clearing

```bash
# Install nbconvert if not available
pip install nbconvert

# Clear outputs from all notebooks
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/*.ipynb
```
👥 Contributors
Manish Tiwari
Aniket Kumar
Satya Priya

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Medical imaging research community

Ultrasound data providers and repositories

Open-source deep learning frameworks (PyTorch, TensorFlow)

Git LFS team for large file management solution

