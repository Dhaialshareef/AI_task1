from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os


np.set_printoptions(suppress=True)


if not os.path.exists("keras_model.h5"):
    raise FileNotFoundError("ملف النموذج 'keras_model.h5' غير موجود!")
model = load_model("keras_model.h5", compile=False)

if not os.path.exists("labels.txt"):
    raise FileNotFoundError("ملف التصنيفات 'labels.txt' غير موجود!")
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]


image_path = input("يرجى إدخال مسار الصورة: ").strip()
if not os.path.exists(image_path):
    raise FileNotFoundError(f"ملف الصورة '{image_path}' غير موجود!")


image = Image.open(image_path).convert("RGB")


size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)


image_array = np.asarray(image)


normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1


data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
data[0] = normalized_image_array


prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]


print(f"الفئة المتوقعة: {class_name}")
print(f"درجة الثقة: {confidence_score:.2f}")
