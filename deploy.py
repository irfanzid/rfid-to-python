import torch
import cv2
import time
# import pytesseract
import re
import numpy as np
import easyocr
import serial
import os
from datetime import datetime


##### Mendefinisikan variabel global
EASY_OCR = easyocr.Reader(['en']) ### inisialisasi easyocr
OCR_TH = 0.2




### -------------------------------------- function untuk menjalankan deteksi ---------------------------------------------------------
def detectx (frame, model):
    frame = [frame]
    print(f"Detecting. . . ")
    results = model(frame)
    # results.show()
    # print( results.xyxyn[0])
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates

### ------------------------------------ untuk plot BBox and results --------------------------------------------------------
def plot_boxes(results, frame,classes):

    """
    --> function ini mengambil results, frame and classes
    --> results: mengandung labels dan prediksi koordinat oleh model pada frame yang diberikan
    --> classes: mengandung strings labels

    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"Total {n} deteksi. . . ")
    print(f"Looping keseluruhan deteksi. . . ")


    ### looping keseluruhan deteksi
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.65: ### threshold value untuk deteksi. Kita membuang seluruh value di bawah threshold
            print(f"Extracting koordinat BBox . . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## koordinat BBOx
            text_d = classes[int(labels[i])]
            # cv2.imwrite("./output/dp.jpg",frame[int(y1):int(y2), int(x1):int(x2)])

            coords = [x1,y1,x2,y2]

            plate_num = recognize_plate_easyocr(img = frame, coords= coords, reader= EASY_OCR, region_threshold= OCR_TH)


            # if text_d == 'mask':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
            cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## untuk text label background
            cv2.putText(frame, f"{plate_num}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)

            # cv2.imwrite("./output/np.jpg",frame[int(y1)-25:int(y2)+25, int(x1)-25:int(x2)+25])




    return frame



#### ---------------------------- function untuk mengenali plat nomor --------------------------------------


# function untuk mengenali plat nomor menggunakan Tesseract OCR
def recognize_plate_easyocr(img, coords,reader,region_threshold):
    xmin, ymin, xmax, ymax = coords
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)] ### mengcrop gambar plat dari keseluruham gambar


    ocr_result = reader.readtext(nplate)



    text = filter_text(region=nplate, ocr_result=ocr_result, region_threshold= region_threshold)
    merged_text = text[0]
    
    # Hapus spasi ekstra di akhir jika diperlukan
    merged_text = merged_text.strip()

    # Menghapus spasi
    merged_text = merged_text.replace(' ', '')

    # Menghapus titik
    merged_text = merged_text.replace('.', '')
    
    # Menghapus koma
    merged_text = merged_text.replace(',', '')
    
    # Membuat seluruh huruf uppercase
    merged_text = merged_text.upper()

    print(f"Hasil pendeteksian ocr:\n{merged_text}")

    if len(text) ==1:
        text = text[0].upper()
    return text


### untuk filter out deteksi yang salah

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    
    plate = [] 
    print(ocr_result)
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate





### ---------------------------------------------- Main function -----------------------------------------------------

print(f"Loading model... ")
## loading the custom trained model
# model =  torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt',force_reload=True) ## jika anda ingin mendowload git repoand kemudian menjalankan deteksi
model =  torch.hub.load('./yolov5-master', 'custom', source ='local', path='best.pt',force_reload=True) ### repo disimpan secara lokal

classes = model.names ### class names merupakan string format




### --------------- untuk deteksi pada gambar --------------------

# Inisialisasi koneksi serial dengan Arduino
print("------------------------------")
print(f"Silahkan Tempelkan kartu . . .")
ser = serial.Serial('COM9', 9600)  # Ganti COM dengan port serial yang sesuai

# Inisialisasi webcam
cap = cv2.VideoCapture(0)  # Ganti nomor kamera jika perlu

# Buat direktori untuk menyimpan foto
output_dir = ['gambar_plat', 'gambar_ocr']  # Ganti dengan nama direktori yang Anda inginkan
for items in output_dir:
    os.makedirs(items, exist_ok=True)

while True:
    data = ser.readline().decode().strip()  # Membaca data dari Arduino
    trig_cam = bool(data)
    if data:
        print("------------------------")
        print("Data dari Arduino:", data)
        
        print("Mengambil gambar Webcam")
        ret, frame = cap.read()
            
        # Dapatkan timestamp untuk nama file yang unik
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Simpan foto untuk kebutuhan computer vision
        ocr_name = "plat_cap.jpg"
        cv2.imwrite(ocr_name, frame)
            
        # Simpan foto dengan nama unik berdasarkan timestamp
        img_name = os.path.join('gambar_plat', f"plat_cap_{timestamp}.jpg")
        cv2.imwrite(img_name, frame)
        print("Foto telah disimpan sebagai", img_name)

            
        # Tampilkan gambar dengan OpenCV
        cv2.imshow("Webcam", frame)

        # Proses easyocr
        print(f"Mendeteksi tulisan: {ocr_name}")
        img_out_name = f"./output/result_{ocr_name.split('/')[-1]}"

        frame = cv2.imread(ocr_name) ### reading the image
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                
        results = detectx(frame, model = model) ### DETECTION HAPPENING HERE    

        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        frame = plot_boxes(results, frame,classes = classes)

        while True:
            # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            cv2.imshow("img_ocr", frame)

            # Simpan foto dengan nama unik berdasarkan timestamp
            img_ocr = os.path.join('gambar_ocr', f"plat_cap_{timestamp}.jpg")
            cv2.imwrite(img_ocr, frame)
            print("Foto telah disimpan sebagai", img_ocr)

            print("----------------------------------")

             # Tunggu hingga tombol 'q' ditekan atau selama 5 detik
            key = cv2.waitKey(5000)  # Tunggu selama 5 detik
            break

    print("Tekan tombol 'q' untuk keluar")
    print("Silahkan tempelkan kartu......")
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break  # keluar dari while loop


# Tutup webcam dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()
           

    
