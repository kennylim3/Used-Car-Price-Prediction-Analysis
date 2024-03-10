# Laporan Proyek Machine Learning - Kenny Lim

## Domain Proyek

Industri otomotif terus berkembang, dan dalam beberapa tahun terakhir, pasar mobil bekas menjadi semakin penting. Banyak konsumen yang memilih 
untuk membeli mobil bekas karena faktor ekonomi dan nilai investasi yang lebih baik. Saat ini, jika konsumen ingin menjual mobil mereka, maka 
mereka harus membawa mobil mereka ke bengkel perusahaan masing-masing atau membuat janji untuk mendapatkan perkiraan harga. Proses ini melibatkan banyak waktu dan sumber daya. Dengan teknologi machine learning, kita dapat membuat model untuk memprediksi harga mobil bekas berdasarkan sejumlah fitur tertentu. Model machine learning yang dapat memprediksi harga mobil bekas tentunya akan berdampak pada beberapa keuntungan, seperti efisiensi dalam keputusan, peningkatan pengalaman pembeli, dan optimalisasi penjualan.

**Referensi**: Gajera, P., Gondaliya, A., & Kavathiya, J. (2021). Old car price prediction with machine learning. Int. Res. J. Mod. Eng. Technol. Sci, 3, 284-290.


## Business Understanding
### Problem Statements
- Harga mobil bekas sangat bervariasi dan sulit diprediksi dengan akurasi tinggi.
- Pembeli sering mengalami kesulitan dalam menilai apakah harga yang ditawarkan adil atau tidak.
- Penjual kesulitan menentukan harga yang bersaing untuk mobil bekas mereka.

### Goals
- Meningkatkan akurasi prediksi harga mobil bekas dengan menggunakan model machine learning.
- Memberikan panduan harga yang lebih transparan dan akurat kepada pembeli dan penjual.
- Meningkatkan efisiensi transaksi dan kepuasan pelanggan di pasar mobil bekas.

### Solution Statements
- Menggunakan beberapa algoritma, antara lain K-Nearest Neighbor, Random Forest, dan Adaptive Boosting.
- Membandingkan model dengan metrik MSE (Mean Squared Error) untuk memilih model dengan performa terbaik.

## Data Understanding
Data yang akan digunakan pada proyek ini adalah dataset Used Car Price Prediction yang diambil dari Kaggle. Dataset tersebut berisi 6019 records data mobil bekas.
Sumber: https://www.kaggle.com/datasets/colearninglounge/used-cars-price-prediction

### Variabel-variabel pada Used Car Price Prediction dataset adalah sebagai berikut:
- Name: nama mobil
- Location: Lokasi di mana mobil tersebut dijual atau terdaftar.
- Year: tahun pembuatan mobil
- Kilometers_Driven: jumlah kilometer yang sudah ditempuh mobil
- Fuel_Type: jenis bahan bakar yang digunakan (Diesel, Petrol, CNG, LPG, Electric)
- Transmission: jenis transmisi mobil (Manual, Automatic)
- Onwer_Type: Tipe kepemilikan mobil (First, Second, Third, Fourth & Above)
- Mileage: konsumsi bahan bakar dalam kilometer per liter
- Engine: kapasitas mesin dalam CC
- Power: tenaga mesin dalam bhp
- Seats: jumlah kursi
- New_Price: harga mobil kondisi baru
- Price: harga mobil

### Exploratory Data Analysis
Melalui teknik visualisasi dan EDA, didapatkan insight-insight berikut.
- Location memiliki pengaruh terhadap harga mobil, di mana Coimbatore adalah lokasi dengan rata-rata harga tertinggi, sedangkan 
Jaiput dan Kolkata memiliki rata-rata harga terendah.
- Harga rata-rata mobil bekas cenderung lebih rendah seiring dengan semakin lamanya tahun pembuatannya.
- Mobil dengan bahan bakar diesel memiliki rata-rata harga tertinggi, diikuti Petrol, CNG, dan LPG.
- Mobil transmisi automatic cenderung lebih mahal daripada transmisi manual.
- Semakin banyak perpindahan kepemilikan mobil, semakin rendah harganya.
- Jumlah kilometer dan jumlah kursi tidak berpengaruh signifikan pada harga mobil.
- Engine dan power berkorelasi positif dengan harga mobil.
- Mileage berkorelasi negatif dengan harga mobil.

## Data Preparation
Beberapa hal yang dilakukan dalam tahap preparation adalah sebagai berikut.
- Encoding Fitur Kategori
Proses encoding fitur kategori dilakukan menggunakan teknik one-hot encoding. Proses ini dilakukan untuk mengonversi fitur kategorikal menjadi bentuk numerik. Encoding membantu mengatasi masalah representasi fitur kategorikal dalam bentuk yang dapat dipahami oleh model.
- Reduksi Dimensi dengan PCA
Proses PCA (Principal Component Analysis) dilakukan untuk mereduksi dimensi fitur. Hal ini akan mempercepat waktu pelatihan model dengan mengurangi jumlah fitur, mengatasi multikolinearitas, dan memperbaiki masalah overfitting pada model.
- Train Test Split
Proses ini dilakukan dengan membagi dataset menjadi data latih dan data uji menggunakan train_test_split dari scikit-learn. 
- Standardisasi
Proses standardisasi menyelaraskan skala fitur dengan mengurangkan mean dan membagi oleh deviasi standar. Standardisasi akan Memastikan semua fitur memiliki skala yang serupa, sehingga tidak ada fitur yang mendominasi yang lain.

## Modeling
Untuk pembuatan model, beberapa algoritma yang digunakan adalah K-Nearest Neighbor (dengan 10 jumlah tetangga), Random Forest (dengan n_estimators = 50, max_depth = 16, random_state=55, n_jobs=-1), dan Adaptive Boosting (dengan learning_rate=0.05, random_state=55).

**K-Nearest Neighbor**

Kelebihan algoritma K-Nearest Neighbor, yaitu:
- Sederhana dan Intuitif: Konsepnya mudah dipahami dan diimplementasikan.
- Non-Parametrik: KNN tidak membuat asumsi terhadap distribusi data, sehingga dapat menangani data yang kompleks dan tidak terstruktur.
- Cocok untuk Data Multiclass: KNN dapat digunakan untuk masalah klasifikasi multiclass tanpa perlu penyesuaian khusus.
Kekurangan algoritma K-Nearest Neighbor, yaitu:
- Komputasi yang Mahal: Proses pengambilan keputusan memerlukan perhitungan jarak ke setiap titik data, yang bisa menjadi mahal untuk dataset besar.
- Sensitif terhadap Outlier: Outlier dapat memiliki pengaruh besar pada hasil prediksi.
- Harus Menyimpan Seluruh Dataset: Model harus menyimpan seluruh dataset pelatihan, yang dapat memakan banyak memori untuk dataset besar.

**Random Forest**

Kelebihan algoritma Random Forest, yaitu:
- Ketangguhan terhadap Overfitting: Random Forest memiliki kemampuan yang baik untuk mengatasi overfitting karena membangun banyak pohon yang beragam.
- Mampu Menangani Data yang Tidak Seimbang: Random Forest dapat memberikan hasil yang baik pada dataset yang tidak seimbang.
- Fitur Importance: Memberikan informasi tentang pentingnya setiap fitur dalam membuat prediksi.
Kekurangan algoritma Random Forest, yaitu:
- Sulit diinterpretasi: Random Forest lebih sulit diinterpretasi daripada model linear sederhana.
- Komputasi yang Mahal: Melibatkan pelatihan sejumlah besar pohon, yang dapat memakan waktu dan sumber daya komputasi.
- Tidak Cocok untuk Data Runtuh (Drift): Random Forest dapat menghadapi masalah saat diterapkan pada data yang berubah secara dinamis.

**Adaptive Boosting**

Kelebihan algoritma Adaptive Boosting, yaitu:
- Ketangguhan terhadap Overfitting: Seperti Random Forest, AdaBoost cenderung memiliki ketangguhan terhadap overfitting.
- Mampu Menangani Data yang Tidak Seimbang: Cocok untuk menangani masalah klasifikasi dengan kelas minoritas.
- Menggunakan Model Lemah: Dapat menggunakan model lemah (misalnya, decision stump) dan meningkatkan performanya.
Kekurangan algoritma Adaptive Boosting, yaitu:
- Sensitif terhadap Noise: Rentan terhadap noise dan outlier dalam data.
- Hyperparameter perlu diperhatikan: Sensitif terhadap konfigurasi hyperparameter yang tidak optimal.
- Komputasi yang Mahal: Meskipun lebih cepat daripada beberapa algoritma kompleks, AdaBoost masih memerlukan banyak waktu pelatihan.

**Model terbaik yang dipilih sebagai solusi adalah model random forest. Hal ini dikarenakan model random forest menghasilkan MSE (Mean Squared Error) yang paling rendah di antara ketiga model sehingga ketika diuji hasil prediksinya, sering kali model random forest memberikan hasil yang lebih mendekati dibanding 2 model lainnya.**

## Evaluation
**Metrik Evaluasi yang Digunakan: Mean Squared Error (MSE)**
MSE adalah metrik evaluasi yang digunakan untuk mengukur sejauh mana perbedaan antara nilai yang diprediksi oleh model dan nilai sebenarnya. Dalam konteks prediksi harga mobil bekas, MSE mengukur rata-rata dari kuadrat selisih antara harga yang diprediksi dan harga sebenarnya.

Formula MSE dan Cara Kerjanya:
- MSE menghitung rata-rata dari kuadrat perbedaan antara nilai prediksi dan nilai sebenarnya.
- Semakin kecil MSE, semakin dekat prediksi dengan nilai sebenarnya.
- MSE memberikan "hukuman" lebih besar untuk perbedaan yang besar, membuatnya cocok untuk kasus prediksi harga.

Hasil dari ketiga model yang dibuat menunjukkan hasil yang cukup baik dengan MSE yang rendah, di mana model KNN memiliki MSE train sebesar 0.028183 dan test sebesar 0.042867, random forest memiliki MSE train sebesar 0.004026 dan test sebesar 0.027424, dan model adaboost memiliki MSE train sebesar 0.045626 dan test sebesar 0.05815. Ketiga model menunjukkan hasil yang baik dan dapat dikatakan good fit.


