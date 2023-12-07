Submission 1: Water Quality Classification

Nama: Aditya Atallah

username: ahdithya

| |   |
|-|--|
|Dataset| [Water Potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability)  |
| Masalah |Manusia membutuhkan air untuk mempertahankan hidrasi tubuh. Air membantu menjaga keseimbangan cairan  dalam tubuh, mengangkut nutrisi ke sel-sel serta membantu proses metabolisme tubuh, sehingga air sangat memiliki pangaruh besar pada tubuh manusia. Air juga dapat menjadi racun pada seseorang jika air yang dikonsumsi tidak memiliki kualitas yang baik. Air yang kurang baik dapat menyebabkan berbagai penyakit pada manusia seperti keracunan, diare, saluran pernafasan dan hal lainnya. Bahkan air yang kurang baik memiliki kandungan tertentu dapat menyebabkan gangguan saraf. Air yang kurang baik juga tidak hanya mempengaruhi tubuh manusia tapi juga alam seperti _Eutrofikasi_. Untuk itu, penting sekali menjaga kualitas air yang akan dikonsumsi atau digunakan, tidak hanya untuk tubuh manusia tapi juga ekosistem alam.|
| Solusi |Para penguji telah melakukan eksperimen terhadap air, pada setiap air memiliki kandungan tertentu. Kandungan ini diantaranya pH air, kekerasan air (_calcium and magnesium salts_),  Kekeruhan air, _Trihalometana_ air dan masih banyak hal lainnya. Kandungan air ini digunakan untuk menentukan kualitas air yang baik atau tidak. Sehingga pada solusi yang digunakan adalah penerapan **Klasifikasi Kualitas Air** berdasarkan kandungan air dengan pendekatan machine learning.|
| Metode Pengelolahan | Data yang terkandung memiliki tipe numerik dan pada feature label memiliki tipe numerik dengan nilai 0 dan 1. Pada pengelolahan data akan melakukan normalisasi data untuk menyamakan nilai data antara 0 hingga 1 tanpa kehilangan nilai sebenarnya.|
| Arsitektur Model | Model menggunakan konsep Jaringan saraf tiruan dengan memiliki beberapa Layer yaiitu layer input dengan 256 _node_ menggunakan _activation relu_, dua hidden layer dengan 64 _node_  dan 16 _node_ menggunakan _activation relu_ dan layer output dengan 1 _node_ menggunakan _activation sigmoid_. pada _compile_ menggunakan _optimizer_ _Adam_, _loss_ nya _Binary_ _Crossentropy_ dan untuk melihat _matrics_ berupa _binary accuracy_|
| Metrics Evaluasi | Metrics Evaluasi yang digunakan pada klasifikasi kualitas air  yaitu _Accuracy_, _AUC_, _Binary Accuracy_, _Precision_ dan _recall_|
| Evaluasi |  Hasil pengujian model yaitu pada metrik _accuracy_ bernilai 0.678, Metrik _AUC_ bernilai 0.68, Metrik _Binary Accuracy_ 0.678, _recall_ bernilai 0.364 dan _precision_ bernilai 0.632  |
|Opsi Deployment|Proyek Klasifikasi Kualitas Air ini akan dideploy menggunakan platform `Railway`, yaitu platform yang menyediakan layanan gratis untuk mendeploy sebuah proyek dengan cara _Platform as a Service_ (PaaS).|
|Web App| [Water-Classification](https://water-classification.up.railway.app/](https://water-classification.up.railway.app/)|
|Monitoring| Pemantauan sistem ini menggunakan Prometheus dan Grafana. Proses pemantauan hanya mengevaluasi permintaan yang diterima oleh sistem. Pada pemantauan  yang dilakukan yaitu melihat perubahan jumlah request yang dilakukan kepada sistem dan menampilkan setiap status pada request yang dilakukan|



