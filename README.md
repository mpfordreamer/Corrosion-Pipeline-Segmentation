# **Proyek Segmentasi Gambar: Deteksi Kelas pada Citra Pipa Korosi dengan DeepLabV3+**

## **Ringkasan Proyek**
![image](https://github.com/dicodingacademy/assets/main/project-dummy/ccdef54bd74b209d8495c643b97b0a88.png)
Proyek ini bertujuan untuk membangun dan melatih model *deep learning* **DeepLabV3+** untuk tugas segmentasi gambar. Model ini dirancang untuk mengklasifikasikan setiap piksel dalam sebuah citra ke dalam kelas-kelas yang telah ditentukan (*background*, *asset*, *corrosion*). Dengan memanfaatkan arsitektur *encoder-decoder* yang kuat dan *Atrous Spatial Pyramid Pooling* (ASPP), model ini mampu menangkap konteks multi-skala untuk menghasilkan prediksi segmentasi yang akurat.

### **Tujuan Proyek:**

1.  **Membangun Model Segmentasi**: Mengimplementasikan arsitektur DeepLabV3+ dengan *backbone* ResNet50 untuk melakukan klasifikasi piksel demi piksel.
2.  **Mencapai Akurasi Tinggi**: Melatih model untuk mencapai metrik evaluasi yang optimal seperti *Intersection over Union* (IoU) dan *Dice Coefficient*.
3.  **Visualisasi Hasil**: Menghasilkan *mask* prediksi yang secara visual dapat membedakan antar kelas dalam citra uji.

### **Cakupan Proyek:**

*   Preprocessing data gambar dan *mask ground truth*.
*   Implementasi dan pelatihan model DeepLabV3+ menggunakan TensorFlow dan Keras.
*   Evaluasi kinerja model pada data uji menggunakan metrik segmentasi standar.
*   Visualisasi hasil prediksi untuk analisis kualitatif.

### **Persiapan Data**

#### **Dataset:**

Dataset yang digunakan terdiri dari gambar asli dan *mask ground truth* yang sesuai, diorganisir dalam struktur folder berikut:
```
/DATASET
    /original
        ORI_001.png
        ...
    /ground_truth
        GT_001.png
        ...
```
**Detail Dataset:**
-   **Total Data**: 216 pasang gambar dan *mask*.
-   **Format**: PNG
-   **Ukuran Gambar**: Diseragamkan menjadi 256x256 piksel.
-   **Mode Warna**: RGB

Pastikan *environment* Anda sudah terinstal dengan *library* yang diperlukan:
```bash
pip install tensorflow numpy opencv-python matplotlib seaborn openpyxl
```

#### **Proses Persiapan Data:**
1.  **Normalisasi & Standarisasi**: Semua gambar dan *mask* diubah ukurannya menjadi 256x256 dan dinormalisasi.
2.  **Pembagian Data**: Dataset dibagi menjadi data latih (80%), validasi (10%), dan uji (10%) secara acak.
3.  **One-Hot Encoding**: *Mask ground truth* (RGB) diubah menjadi format *one-hot encoded* dengan 3 kelas untuk digunakan sebagai target pelatihan.
    *   **Kelas 0**: Merah `(255, 0, 0)`
    *   **Kelas 1**: Biru `(0, 0, 255)`
    *   **Kelas 2**: Hijau `(0, 255, 0)`

### **Modeling**

Model yang dibangun adalah **DeepLabV3+**, sebuah arsitektur *state-of-the-art* untuk segmentasi semantik.

*   **Encoder**: Menggunakan **ResNet50** yang telah dilatih pada ImageNet untuk mengekstraksi fitur dari gambar input.
*   **Atrous Spatial Pyramid Pooling (ASSP)**: Blok ini digunakan untuk menangkap konteks pada berbagai skala tanpa mengurangi resolusi spasial, yang krusial untuk segmentasi objek berukuran berbeda.
*   **Decoder**: Menggabungkan fitur dari *encoder* dengan fitur dari ASSP, lalu melakukan *upsampling* untuk menghasilkan *mask* segmentasi dengan resolusi yang sama seperti gambar asli.
*   **Fungsi Aktivasi Output**: Menggunakan `softmax` untuk menghasilkan probabilitas kelas untuk setiap piksel.

### **Evaluation**

Kinerja model dievaluasi menggunakan metrik standar untuk tugas segmentasi:

*   **Intersection over Union (IoU)**: Mengukur tumpang tindih antara *mask* prediksi dan *ground truth*.
*   **Dice Coefficient**: Mirip dengan IoU, metrik ini juga mengukur tumpang tindih dan sangat umum digunakan dalam segmentasi.
*   **F1-Score**: Memberikan skor tunggal yang menyeimbangkan *precision* dan *recall*.
*   **Confusion Matrix**: Memberikan gambaran visual tentang performa klasifikasi untuk setiap kelas.

Berikut adalah ringkasan hasil evaluasi performa model pada data uji:

| Class                 | IoU (Rata-rata) | Dice (Rata-rata) | F1-Score (Rata-rata) |
|-----------------------|-----------------|------------------|----------------------|
| Corrosion (Merah)     | XX.XX%          | XX.XX%           | XX.XX%               |
| Asset (Biru)          | XX.XX%          | XX.XX%           | XX.XX%               |
| Background (Hijau)    | XX.XX%          | XX.XX%           | XX.XX%               |
| **Overall**           | **XX.XX%**      | **XX.XX%**       | **XX.XX%**           |

*(Catatan: Metrik di atas adalah placeholder. Isi dengan hasil evaluasi model Anda setelah pelatihan selesai)*

### **Arsitektur & Deployment**

Model DeepLabV3+ yang telah dilatih diintegrasikan ke dalam sistem aplikasi untuk penggunaan praktis melalui arsitektur berbasis layanan (*microservices*).

*   **Backend (FastAPI)**: Backend API dibangun menggunakan **FastAPI** (Python) untuk melayani model *machine learning*. Endpoint ini bertanggung jawab untuk menerima gambar, memprosesnya dengan model DeepLabV3+, dan mengembalikan hasil segmentasi.
*   **Frontend (ASP.NET)**: Aplikasi *web* yang menghadap pengguna (antarmuka) dikembangkan menggunakan **ASP.NET**. Pengguna dapat mengunggah gambar melalui antarmuka ini, yang kemudian akan dikirim ke backend FastAPI untuk diproses.

**Alur Kerja Sistem:**
1.  Pengguna mengakses aplikasi web ASP.NET dan mengunggah gambar.
2.  Frontend ASP.NET mengirimkan gambar ke endpoint API FastAPI.
3.  Backend FastAPI menerima gambar, melakukan *preprocessing*, dan memberikannya ke model DeepLabV3+ untuk prediksi.
4.  Model menghasilkan *mask* segmentasi.
5.  FastAPI mengembalikan *mask* hasil prediksi ke aplikasi ASP.NET.
6.  Frontend menampilkan gambar asli beserta hasil segmentasinya kepada pengguna.

### **Conclusion**

Proyek ini berhasil mengimplementasikan model **DeepLabV3+** untuk tugas segmentasi gambar multikelas. Model ini menunjukkan kemampuan yang baik dalam membedakan antara kelas *corrosion*, *asset*, dan *background* pada level piksel. Arsitektur DeepLabV3+ dengan *backbone* ResNet50 terbukti efektif untuk tugas ini, didukung oleh teknik seperti ASSP yang mampu menangkap informasi kontekstual yang kaya.

#### **Potensi Aplikasi**

Model segmentasi seperti ini memiliki aplikasi luas di berbagai industri, antara lain:
-   **Inspeksi Infrastruktur**: Mendeteksi korosi atau kerusakan pada jembatan, pipa, atau struktur logam lainnya secara otomatis.
-   **Manufaktur**: Mengidentifikasi cacat produk pada lini produksi.
-   **Analisis Medis**: Segmentasi organ atau jaringan abnormal dari citra medis seperti CT scan atau MRI.
-   **Monitoring Lingkungan**: Memetakan jenis tutupan lahan dari citra satelit.
