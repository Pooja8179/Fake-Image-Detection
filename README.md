🖼️ Fake Image Detection

This project focuses on detecting fake or manipulated images (such as deepfakes) using deep learning techniques. The model is built with Convolutional Neural Networks (CNNs) 🧠 to analyze image features and classify them as ✅ real or ❌ fake.

✨ Features

🔄 Image preprocessing including resizing and normalization

🎯 Feature extraction using CNN layers

🤖 Training a deep learning model for fake vs real classification

📊 Evaluation using accuracy, precision, recall, and F1-score

📈 Visualization of training and testing results

📂 Project Structure

📁 dataset/ – contains image data (not uploaded due to large size, external link provided)

🗂️ models/ – trained model files

💻 src/ – source code for preprocessing, training, and evaluation

📜 requirements.txt – dependencies

📝 README.md – project documentation

📦 Dataset

The dataset is large (around 3 GB) and cannot be stored directly in GitHub.
📥 Download it from an external link (Google Drive / Kaggle / Hugging Face).

⚙️ Installation & Setup

⬇️ Clone the repository.

📦 Install required Python libraries from requirements.txt.

📂 Download the dataset and place it inside the dataset folder.

🚀 Training

The model can be trained using the training script with adjustable parameters like epochs, batch size, and learning rate.

🏆 Results

The model achieves good performance in classifying images as real or fake.
✅ Accuracy, 🎯 Precision, 🔄 Recall, and 🏅 F1-score are reported after training.

🔮 Future Work

📊 Use larger and more diverse datasets for better generalization

🧩 Experiment with advanced architectures such as Vision Transformers

🌐 Deploy the model as a real-time web application



# Environment Setup

Make sure Anaconda is installed and launch anaconda prompt and navigate to root directory in the anaconda prompt

create venv

```shell
conda create -n "DeepFakeImageDetection" python=3.10
```

Activate

```shell
conda activate DeepFakeImageDetection 
```

In order to set your environment up to run the code here, first install all requirements:

```shell
pip install -r requirements.txt
```

run the app.py file 

```shell
python app.py
```

Once you see this url - http://127.0.0.1:5000/ in logs, open it in browser.


Now your setup is ready.
