# 📈 Advanced Time Series Forecasting: Seq2Seq LSTM with Multi-Head Attention

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Custom_Layers-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
Proyek ini membangun arsitektur Deep Learning tingkat lanjut untuk memprediksi harga aset keuangan (Close Price) hingga **24 langkah (jam) ke depan** (*Multi-step Forecasting*). 

Proyek ini tidak hanya menggunakan model LSTM standar, melainkan mengeksplorasi arsitektur **Encoder-Decoder (Seq2Seq)** yang dikombinasikan dengan mekanisme **Custom Multi-Head Attention** dan teknik prediksi **Autoregressive Inference**.

Hasil akhir dari model Seq2Seq ini berhasil mencapai **Mean Absolute Error (MAE) sebesar 0.01096** (pada data *scaled*), jauh melampaui target akurasi standar.

## 🧠 Model Architecture

### 1. Baseline Model (LSTM + Attention)
Model baseline dirancang sebagai standar perbandingan yang kuat, menggunakan:
* LSTM Layer (64 units)
* **Custom Multi-Head Attention Layer** (2 Heads, Key Dim: 32)
* Global Average Pooling 1D
* **Custom Dense Layer** untuk *output horizon* 24 jam.

### 2. Advanced Model (Seq2Seq LSTM)
Model utama menggunakan pendekatan *Sequence-to-Sequence* untuk menangani kompleksitas prediksi jangka panjang:
* **Encoder LSTM:** Mempelajari representasi fitur historis (window size: 72 jam).
* **Decoder LSTM:** Menerima *hidden states* dari Encoder dan memproses prediksi langkah demi langkah.
* **Custom Multi-Head Attention:** Membantu Decoder berfokus pada bagian paling relevan dari *output* Encoder pada setiap langkah waktu (Time Step).
* **Autoregressive Inference Loop:** Model menggunakan prediksinya sendiri pada jam ke-$t$ sebagai input untuk memprediksi jam ke-$t+1$.

## 📊 Dataset & Preprocessing
* **Features:** Menggunakan 5 fitur multivariat (`Volume USDT`, `RSI`, `MACD`, `ATR`, `Close`).
* **Window Size:** 72 data historis (jam).
* **Horizon:** 24 prediksi masa depan (jam).
* **Scaling:** `MinMaxScaler` (Fit hanya pada *Train Data* untuk mencegah *Data Leakage*).
* **EDA:** Heatmap korelasi, Uji Stasioneritas (ACF/PACF), dan Dekomposisi Time Series (*Trend, Seasonal, Residual*).

## 🏆 Key Results & Performance
Model dilatih menggunakan optimasi *Custom Training Loop* (`tf.GradientTape`) dengan *Advanced Weighted Loss* dan mekanisme adaptif *Learning Rate* (Fine-Tuning Callback).

* **Final MAE (Scaled):** `0.01096`
* **Inference Method:** Autoregressive Step-by-Step

## 📂 Repository Structure
```text
├── NathanAlvinoFam_Submission_Akhir_DLTM.ipynb  # Main Jupyter Notebook
├── best_model_seq2seq_LSTM.keras                # Model weights (Phase 2 Fine-Tuned)
├── model_seq2seq_LSTM.keras                     # Saved Advanced Model architecture
├── model_baseline_LSTM.keras                    # Saved Baseline Model architecture
├── requirements.txt                             # Dependencies list
└── README.md                                    # Project documentation
