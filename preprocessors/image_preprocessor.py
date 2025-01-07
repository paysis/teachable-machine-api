from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np

class ImagePreprocessor:
    @staticmethod
    def preprocess(file):
        try:
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            img = Image.open(file).convert("RGB")
            img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS) # resize and crop from the center
            
            img_array = np.asarray(img)
            normalized_img_array = (img_array.astype(np.float32) / 127.5) - 1

            data[0] = normalized_img_array
            return data
        except UnidentifiedImageError:
            raise ValueError("Geçersiz veya bozuk bir resim dosyası")