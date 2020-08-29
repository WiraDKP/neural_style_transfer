# Neural Style Transfer with PyTorch
![Neural Style Transfer Najwa Barli](assets/style_transfer_result.png)

# Starter Guide
## Step 1: Download materi
- Klik disini untuk [Download ZIP](https://codeload.github.com/WiraDKP/neural_style_transfer/zip/master), atau
- Bagi yang familiar dengan git, boleh menggunakan clone
    ```
    git clone https://github.com/WiraDKP/neural_style_transfer.git
    ```

## Step 2: Instalasi Miniconda
### **Windows user**
- Download miniconda untuk Python 3.7
    - Klik link ini untuk download: [Miniconda Windows 64-bit](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)
    - Note: skip step ini apabila kamu sudah menggunakan Anaconda sebelumnya. Walau demikian, saya akan jelaskan alasan kenapa kamu sebaiknya menggunakan miniconda nanti di course ini.

- Install miniconda
    - Ketika ada pilihan `install for`, pilih `Just Me (recommended)`
    - Untuk `Advanced Options`, silahkan centang `Register Anaconda as my default Python 3.7`
    - Tunggu hingga instalasi selesai

- Jalankan `Anaconda Prompt`

### **Mac user**
- Download miniconda untuk Python 3.7
    - Klik link ini untuk download: [Miniconda Mac OS X 64-bit](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.pkg)
    - Note: skip step ini apabila kamu sudah menggunakan Anaconda sebelumnya. Walau demikian, saya akan jelaskan alasan kenapa kamu sebaiknya menggunakan miniconda nanti di course ini.

- Install miniconda
    - Install tanpa mengubah opsi apapun
    - Tunggu hingga instalasi selesai

- Jalankan terminal

### **Linux user**
- Download miniconda untuk Python 3.7
    - Klik link ini untuk download: [Miniconda Linux 64-bit](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)
    - Note: skip step ini apabila kamu sudah menggunakan Anaconda sebelumnya. Walau demikian, saya akan jelaskan alasan kenapa kamu sebaiknya menggunakan miniconda nanti di course ini.
    
- Install miniconda
    - jalankan terminal
    - Install miniconda menggunakan command berikut
        ```
        bash Miniconda3-latest-Linux-x86_64.sh
        ```
    - Ketik `yes` untuk agree dengan license nya, kemudian `yes` lagi untuk `prepend miniconda install location to PATH`
    - Tunggu hingga instalasi selesai
    
- hanya untuk memastikan, tutup dan buka terminal lagi

## Step 3: Instalasi Jupyter 
- Kita akan install 2 hal di base environment
    ```
    conda install --name base jupyter nb_conda_kernels
    ```

## Step 4: Instalasi Environment
- Change directory `cd` ke folder kerja ini
    ```
    cd neural_style_transfer/
    ```
- Jalankan command ini untuk menginstall environment `style_transfer`
    ```
    conda env create -f environment.yml
    ```

## Step 5: Memastikan environment terinstall dengan baik
- Jalankan command berikut untuk mengecek instalasi dan ikuti instruksi yang dihasilkan
    ```
    python check_installation.py
    ```
- Jika sudah aman, maka akan muncul keterangan berikut, dan kita bisa mulai belajar. Semangat!
    ```
    ✓ jupyter telah terinstall dengan baik
    ✓ nb_conda_kernels telah terinstall dengan baik
    ✓ Environment style_trasnfer terdeteksi
    ✓ Package telah terinstall dengan baik di dalam environment style_trasnfer
    ✓ Instalasi berjalan dengan baik. Selamat belajar!
    ```

# Image Reference
- [Logo Peringatan Hari Ulang Tahun Ke-75 Kemerdekaan Republik Indonesia Tahun 2020](https://www.setneg.go.id/view/index/peringatan_hari_ulang_tahun_ke_75_kemerdekaan_republik_indonesia_tahun_2020)
- [Fruit Seller oleh Barli Sasmitawinata](http://4.bp.blogspot.com/-BUoRG9l23c0/URWlrjxEsQI/AAAAAAAADvY/19m7brBMQEI/s1600/Fruit+Saller_tumb.jpg)
- [Lukisan oleh Delsy Syamsumar](http://artkimianto.blogspot.com/2010/02/pelukis-legendaris-delsy-syamsumar.html)
http://harajukushina.blogspot.com/2014/02/gambar-lukisan-karya-seni-barli.html
- Foto Najwa Shihab - IG: [@najwashihab](https://www.instagram.com/najwashihab/?hl=en)
- [Doki-doki Literature Club Sketh](https://knowyourmeme.com/photos/1305762-doki-doki-literature-club)
