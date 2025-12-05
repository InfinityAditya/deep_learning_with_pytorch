# ⚡ Deep Learning with PyTorch: From Fundamentals to Computer Vision

### **Master the PyTorch Framework! Build, Train, and Deploy Cutting-Edge Neural Networks with a Focus on Flexibility and Dynamic Graph Computation.**
---
[![GitHub Repo Size](https://img.shields.io/github/repo-size/InfinityAditya/deep_learning_with_pytorch?style=for-the-badge&color=blue)](https://github.com/InfinityAditya/deep_learning_with_pytorch)
[![GitHub License](https://img.shields.io/github/license/InfinityAditya/deep_learning_with_pytorch?style=for-the-badge&color=green)](LICENSE)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-Compatible-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org/)

---

## 💡 Introduction: Why PyTorch?

PyTorch is the framework of choice for researchers and developers who prioritize **speed and flexibility**. It uses a **dynamic computation graph** (unlike TensorFlow's historical static graph), which makes debugging and complex model building easier.

This repository guides you through the entire Deep Learning pipeline using PyTorch's elegant API.

## 🗺️ The Deep Learning Journey: Notebook Guide

Follow this sequential path to become proficient in PyTorch. Each notebook is a Colab-ready lab focusing on practical, executable code.

| # | Notebook Name | Key Concepts Covered | Difficulty | Open in Colab |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **`pytorch-fundamentals.ipynb`** | **Tensors** (the PyTorch equivalent of NumPy arrays), CUDA setup, basic arithmetic, and `Autograd` (automatic differentiation). | ⭐ Basic | [Open in Colab](LINK_TO_COLLAB) |
| 2 | **`pytorch_workflow.ipynb`** | The 5 essential steps: 1. **Data loading** (`DataLoader`), 2. **Model setup** (`nn.Module`), 3. **Loss function**, 4. **Optimizer**, 5. The **Training Loop**. | ⭐⭐ Intermediate | [Open in Colab](LINK_TO_COLLAB) |
| 3 | **`pytorch_classifications.ipynb`** | Implementing **Neural Networks** for binary and multi-class classification, using standard datasets like MNIST or FashionMNIST. | ⭐⭐ Intermediate | [Open in Colab](LINK_TO_COLLAB) |
| 4 | **`PyTorch_Computer_Vision.ipynb`** | Introduction to **Convolutional Neural Networks (CNNs)**. Working with image datasets, convolutions, pooling layers, and transfer learning concepts. | ⭐⭐⭐ Advanced | [Open in Colab](LINK_TO_COLLAB) |

> **💡 Action Required:** Please replace **`LINK_TO_COLLAB`** with the live Google Colab share link for each notebook file!

---

## 🛠️ Interactive Learning: Setting Up and Running

1.  **🚀 Easiest Way (Recommended):** Click the **'Open in Colab'** badge above.
    * **Crucial Step:** Once in Colab, go to **Runtime** > **Change runtime type** and ensure **T4 GPU** is selected. PyTorch models run significantly faster on a GPU!
    * 2.  **👯 Clone the Repo:** For local machine setup (with virtual environments):
    ```bash
    git clone [https://github.com/InfinityAditya/deep_learning_with_pytorch.git](https://github.com/InfinityAditya/deep_learning_with_pytorch.git)
    cd deep_learning_with_pytorch
    # Assuming you use Conda
    conda create -n pytorch_env python=3.10
    conda activate pytorch_env
    # Install PyTorch (check official site for latest command)
    pip install torch torchvision torchaudio
    ```

---

## 🤯 Make Learning Crazy: The PyTorch Challenge

To ensure you truly understand the workflow, try this hands-on project after completing the notebooks:

### **Project: Build a Cat vs. Dog Image Classifier**

1.  **Data:** Find a small public Cat vs. Dog dataset.
2.  **Model:** Implement a **custom CNN** using the knowledge from `PyTorch_Computer_Vision.ipynb`.
3.  **Optimization:** Experiment with different optimizers (e.g., Adam vs. SGD) and learning rates (using the `pytorch_workflow.ipynb` concepts).
4.  **Visualize:** Plot the **training loss and accuracy curves** to see how your model learns! 
---

## 📚 Handbooks & Resources for Advanced PyTorch

* **Official PyTorch Documentation:** Excellent API reference and tutorials.
* **PyTorch Examples Repository:** Official collection of examples for common tasks (RNNs, GANs, etc.).
* **Dive into Deep Learning (D2L):** A free book with PyTorch implementations for all core DL concepts.
* **The PyTorch Forums:** The best place to troubleshoot errors and ask complex questions.

---

## 🤝 Contributing

We welcome improvements, new notebooks (e.g., for **NLP or Time Series**), and bug fixes. Please follow the standard workflow: Fork -> Branch -> Commit -> Pull Request.

---

I've made the README highly actionable and focused on the core advantage of PyTorch (GPU usage, dynamic graphs). Do you have any other repositories or topics you'd like me to help organize?
