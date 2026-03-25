# 🚀 Rancang Bangun Sistem Kendali Adaptif L1 pada ROV  
## Berbasis Optimasi Algoritma Genetika & Analisis Kestabilan Lyapunov

Kerangka simulasi komprehensif berbasis Python untuk *Remotely Operated Vehicle* (ROV) yang mengimplementasikan:

- **Sistem Kendali Adaptif L1**
- **Optimasi Algoritma Genetika (GA)**
- **Analisis Kestabilan Lyapunov**

---

## 🎯 Rumusan Masalah

Penelitian ini berfokus pada:

1. **Kinerja kendali Adaptif L1**  
   Bagaimana sistem mempertahankan *trajectory tracking* di bawah ketidakpastian dan gangguan arus laut?

2. **Optimasi parameter menggunakan GA**  
   Bagaimana GA melakukan *auto-tuning* parameter:
   - $\omega_c$ (bandwidth filter)  
   - $\gamma$ (adaptation gain)  
   - $k$ (feedback gain)  

   untuk meminimalkan error (ITAE / ISE)?

3. **Validasi kestabilan Lyapunov**  
   Apakah sistem stabil secara asimtotik berdasarkan fungsi energi Lyapunov?

---

## 🌟 Fitur Utama

| Fitur | Deskripsi |
|------|----------|
| **Dinamika ROV 6-DOF** | Model berbasis persamaan Fossen (massa, Coriolis, drag) |
| **L1 Adaptive Control** | Robust terhadap gangguan dan dinamika tak termodel |
| **Genetic Algorithm (GA)** | Auto-tuning parameter kontrol |
| **Analisis Lyapunov** | Validasi kestabilan real-time |
| **GUI PyQt5** | Visualisasi interaktif |
| **Export JSON** | Simpan data simulasi |

---

## 📁 Struktur Proyek

```
roc_l1_ga_project/
│
├── controller/
│   └── l1_adaptive.py
│
├── gui/
│   └── main_window.py
│
├── optimizer/
│   ├── ga_tuner.py
│   └── plotting.py
│
├── plant/
│   ├── environment.py
│   └── rov_dynamics.py
│
├── utils/
│   └── plotting.py
│
├── main.py
└── requirements.txt
```

---

## ⚙️ Instalasi

### 1. Clone repository

```bash
git clone https://github.com/gmsaw/roc_l1_ga_project.git
cd roc_l1_ga_project
```

### 2. Virtual Environment

```bash
python -m venv venv

# Linux / MacOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Cara Menjalankan

```bash
python main.py
```

---

## 🖥️ Panduan GUI

| Komponen | Fungsi |
|----------|--------|
| Start / Pause / Stop | Kontrol simulasi |
| Parameter Tuning | Manual atau via GA |
| Trajectory | Step, Smooth, Sine |
| Environment | Simulasi arus laut |
| Lyapunov Tab | Analisis kestabilan |
| Export JSON | Simpan data |

---

## 🧮 Landasan Matematis

### 1. Fungsi Lyapunov

Fungsi energi sistem:

$$
V(t) = \frac{1}{2} e(t)^T K_p e(t) + \frac{1}{2} v(t)^T M v(t)
$$

Dimana:

- $e(t)$ = error posisi  
- $v(t)$ = kecepatan  
- $K_p$ = gain kontrol  
- $M$ = matriks massa  

Kondisi kestabilan:

$$
\frac{dV(t)}{dt} \leq 0
$$

---

### 2. Hukum Adaptasi (L1 Adaptive Law)

$$
\frac{d\sigma}{dt} = \Gamma \cdot \text{Proj}(\sigma, -\tilde{x})
$$

Dimana:

- $\Gamma$ = adaptation gain  
- $\tilde{x}$ = estimation error  
- $\text{Proj}$ = operator pembatas  

---

## 📊 Metrik Evaluasi

| Metrik | Formula | Deskripsi |
|--------|--------|----------|
| ITAE | $ \int_0^T t |e(t)| dt $ | Penalti error lama |
| ISE | $ \int_0^T |e(t)|^2 dt $ | Penalti error besar |
| IAE | $ \int_0^T |e(t)| dt $ | Total error |

---

## 👤 Penulis

**Gede Mahendra Sastra Adhi Wiguna**  
Program Studi Fisika, Universitas Udayana  