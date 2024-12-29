
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # TkAgg, genellikle en uyumlu backend'dir.
import os

# Ana klasör yolu (dataset_folder değişkeni path değerini içermelidir)
dataset_folder = 'C:/Users/yusuf/.cache/kagglehub/datasets/naderabdalghani/iam-handwritten-forms-dataset/versions/1/data'

# Tüm klasörleri ve resimleri listeleme
image_files = []
a=0
for root, _, files in os.walk(dataset_folder):
    for file in files:
        if file.endswith(".png") :
            image_files.append(os.path.join(root, file))


print(f"{len(image_files)} adet görüntü bulundu.")


def detect_lines_and_split(image):
    # Görüntüyü yükleme (grayscale format)

    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)


    # Görüntüyü ters çevirme (çizgiler beyaz olacak)
    inverted_image = cv2.bitwise_not(image)

    # Kenar tespiti
    edges = cv2.Canny(inverted_image, threshold1=100, threshold2=200, apertureSize=3)

    # Kontur bulma
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Yatay çizgileri tespit etme
    lines = []
    for contour in contours:
        # Çizginin bounding box koordinatları
        x, y, w, h = cv2.boundingRect(contour)

        # Çizgileri yükseklik ve genişlik kriterine göre filtreleme
        aspect_ratio = w / h
        if h < 5 or aspect_ratio <15:  # İnce ve uzun olanları seç
            continue
        lines.append(y)

    # Çizgileri sıralama
    lines = sorted(lines)

    # Çizgiler arasındaki bölümleri ayırma
    label_part = image[lines[0]:lines[1], :] if len(lines) > 1 else None
    handwritten_part = image[lines[1]:lines[2], :] if len(lines) > 2 else None

    return label_part, handwritten_part, lines

output_label_dir = "output/labels"
output_handwritten_dir = "output/handwritten"
os.makedirs(output_label_dir, exist_ok=True)
os.makedirs(output_handwritten_dir, exist_ok=True)

for idx, file in enumerate(image_files):
    label_part, handwritten_part, _ = detect_lines_and_split(file)

    if label_part is not None:
        cv2.imwrite(f"{output_label_dir}/image{idx}.png", label_part)
    if handwritten_part is not None:
        cv2.imwrite(f"{output_handwritten_dir}/image{idx}.png", handwritten_part)

print("Etiket ve el yazısı bölümleri ayrıldı ve kaydedildi.")


i = image_files[1006]
print(i)
# Görüntüyü işleme
label_image, handwritten_image, detected_lines = detect_lines_and_split(i)




original_image = cv2.imread(i)
for y in detected_lines:
    cv2.line(original_image, (0, y), (original_image.shape[1], y), (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title("Tespit Edilen Çizgiler")
plt.axis("off")
plt.show()







