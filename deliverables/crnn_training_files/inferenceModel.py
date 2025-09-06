import cv2
import numpy as np
import pandas as pd
from collections import Counter
from io import StringIO
from tqdm import tqdm

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

from autocorrect import Speller


def letterbox_image(image, target_size):
    """
    Resize image to fit inside target_size while keeping aspect ratio.
    Adds white padding to fill the rest.
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255  # white background
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return padded


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, model_path: str, val_csv_path: str, *args, **kwargs):
        super().__init__(model_path=model_path, *args, **kwargs)

        self.spell = Speller(lang='en')  # Uses a simple English dictionary

    def correct_spelling(self, text: str) -> str:
        words = text.lower().split()
        corrected = [self.spell(word) for word in words]
        corrected_text = " ".join(corrected)
        print(f"🔎 Raw: '{text}' → Corrected: '{corrected_text}'")
        return corrected_text


    def predict(self, image: np.ndarray) -> str:
        input_size = self.input_shapes[0][1:3][::-1]  # (width, height)

        # Step 1: Letterbox to match model input
        letterboxed_image = letterbox_image(image, input_size)
        cv2.imwrite("model_input_view.png", letterboxed_image)
        print("📐 Saved letterboxed image as 'model_input_view.png'")

        # Step 2: Ensure RGB if grayscale
        if len(letterboxed_image.shape) == 2 or letterboxed_image.shape[2] == 1:
            letterboxed_image = cv2.cvtColor(letterboxed_image, cv2.COLOR_GRAY2RGB)

        # Step 4: Convert to float32 AFTER enhancement
        image = letterboxed_image.astype(np.float32)
        image_pred = np.expand_dims(image, axis=0)  # Add batch dim

        # Step 5: Run inference
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]
        text = ctc_decoder(preds, self.metadata["vocab"])[0]

        return text

def case_insensitive_cer(prediction: str, label: str) -> float:
    return get_cer(prediction.lower(), label.lower())
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Paths to model and CSV
    model_path = "/content/drive/MyDrive/Colab Notebooks/crnnSentences54onward/Models/model.onnx"
    val_csv_path = "/content/drive/MyDrive/Colab Notebooks/crnnSentences54onward/Models/val.csv"

    # Load model with SymSpell
    model = ImageToWordModel(model_path=model_path, val_csv_path=val_csv_path)

    # 🔁 Try with a single image
    test_image_path = "/content/drive/MyDrive/Colab Notebooks/crnnSentences54onward/bb.png"
    test_image = cv2.imread(test_image_path)

    if test_image is None:
        print(f"❌ Failed to load test image: {test_image_path}")
    else:
        raw_text = model.predict(test_image)
        corrected_text = model.correct_spelling(raw_text)

        print(f"\n🔤 Raw Prediction: {raw_text}")
        print(f"✅ Corrected Prediction: {corrected_text}")

        # 🖼️ Show the image
        cv2.imwrite("output_image.png", test_image)
        print("🖼️ Saved test image as 'output_image.png'")



    # # Load CSV and start evaluation
    # df = pd.read_csv(val_csv_path, header=None).values.tolist()

    # cer_scores = []

    # for image_path, label in tqdm(df):
    #     image_path = image_path.replace("\\", "/")
    #     image = cv2.imread(image_path)

    #     if image is None:
    #         print(f"❌ Could not load image: {image_path}")
    #         continue

    #     # Predict and correct
    #     raw_prediction = model.predict(image)
    #     corrected_prediction = model.correct_spelling(raw_prediction)

    #     cer = case_insensitive_cer(corrected_prediction, label)
    #     print(f"✅ Image: {image_path} | Label: {label} | Prediction: {corrected_prediction} | CER: {cer:.4f}")

    #     cer_scores.append(cer)

    # if cer_scores:
    #   avg_cer = np.mean(cer_scores)
    #   print(f"\n📊 Average Case-Insensitive CER: {avg_cer:.4f}")
