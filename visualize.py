import ast
import cv2
import numpy as np
import pandas as pd

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=3, line_length_x=50, line_length_y=50):
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Köşelerden geçen kısa “L” çizgileri
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

# CSV'den verileri alıyoruz
results = pd.read_csv('./test_interpolated.csv')

# Video yükleme
video_path = 'vid.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

license_plate = {}
car_ids = np.unique(results['car_id'])

# Her araca ait en yüksek license_number_score olan plakayı belirleyip kırpıyoruz
for car_id in car_ids:
    max_score = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    best_row = results[(results['car_id'] == car_id) & 
                       (results['license_number_score'] == max_score)].iloc[0]

    license_plate[car_id] = {
        'license_crop': None,
        'license_plate_number': best_row['license_number']
    }

    # En yüksek puanlı plakayı yakaladığımız frame'e gidip orada plaka görselini kırp
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_row['frame_nmr'])
    ret, frame = cap.read()

    bbox_str = best_row['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
    x1, y1, x2, y2 = ast.literal_eval(bbox_str)

    # Plakayı yeniden boyutlandırırken 400 yerine 200 piksel yükseklik
    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 200 / (y2 - y1)), 200))

    license_plate[car_id]['license_crop'] = license_crop

frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Videodaki her kareyi oku, araç/plaka çizimlerini ekle
while True:
    ret, frame = cap.read()
    frame_nmr += 1
    if not ret:
        break

    df_ = results[results['frame_nmr'] == frame_nmr]
    for row_indx in range(len(df_)):
        car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(
            df_.iloc[row_indx]['car_bbox']
            .replace('[ ', '[')
            .replace('   ', ' ')
            .replace('  ', ' ')
            .replace(' ', ',')
        )

        # Aracın etrafına (yeşil) çerçeve çizimi
        draw_border(frame,
                    (int(car_x1), int(car_y1)),
                    (int(car_x2), int(car_y2)),
                    (0, 255, 0),
                    thickness=3,
                    line_length_x=50,
                    line_length_y=50)

        # Plaka dikdörtgeni (kırmızı)
        x1, y1, x2, y2 = ast.literal_eval(
            df_.iloc[row_indx]['license_plate_bbox']
            .replace('[ ', '[')
            .replace('   ', ' ')
            .replace('  ', ' ')
            .replace(' ', ',')
        )
        cv2.rectangle(frame,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      (0, 0, 255),
                      3)

        # Kırpılmış plaka resmini ve text'i yerleştirelim
        car_id = df_.iloc[row_indx]['car_id']
        license_crop = license_plate[car_id]['license_crop']
        plate_text = license_plate[car_id]['license_plate_number']
        H, W, _ = license_crop.shape

        # Yazı boyutu ve kalınlığı
        font_scale = 1
        font_thickness = 2

        try:
            # Plaka görselini aracın üzerindeki boşluğa yerleştiriyoruz
            frame[int(car_y1) - H - 100 : int(car_y1) - 100,
                  int((car_x2 + car_x1 - W) / 2) : int((car_x2 + car_x1 + W) / 2), :] = license_crop

            # Plaka text'inin altına/beyaz zemin yapıyoruz
            frame[int(car_y1) - H - 200 : int(car_y1) - H - 100,
                  int((car_x2 + car_x1 - W) / 2) : int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

            # Plaka yazısını çiz
            (text_width, text_height), _ = cv2.getTextSize(plate_text,
                                                           cv2.FONT_HERSHEY_SIMPLEX,
                                                           font_scale,
                                                           font_thickness)

            cv2.putText(frame,
                        plate_text,
                        (int((car_x2 + car_x1 - text_width) / 2),
                         int(car_y1 - H - 150 + (text_height / 2))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0),
                        font_thickness)

        except Exception as e:
            # Yeri dar kalırsa hata alabilir; atlayıp devam ediyoruz
            pass

    # Değişiklikleri videoya yaz
    out.write(frame)

    # İsterseniz anlık kontrol için ekran gösterimi yapabilirsiniz
    # (komutları yoruma almanızda sakınca yok)
    # show_frame = cv2.resize(frame, (1280, 720))
    # cv2.imshow('frame', show_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

out.release()
cap.release()
# cv2.destroyAllWindows()
