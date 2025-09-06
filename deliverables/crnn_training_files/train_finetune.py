# This script mirrors `train.py` but incorporates the custom Malaysian handwriting dataset
# into the training pipeline for fine-tuning the CRNN model.
import os
import tensorflow as tf
import tf2onnx
from tqdm import tqdm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage

from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CERMetric, WERMetric

from model import train_model
from configs import ModelConfigs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ {len(gpus)} GPU(s) detected.")
    for gpu in gpus:
        print("GPU name:", gpu.name)
else:
    print("❌ No GPU found.")
# Path to IAM Sentences dataset (download manually)
sentences_txt_path = os.path.join("Datasets", "IAM_Sentences", "ascii", "sentences.txt")
sentences_folder_path = os.path.join("Datasets", "IAM_Sentences", "sentences")

dataset, vocab, max_len = [], set(), 0
with open(sentences_txt_path, "r") as f:
    words = f.readlines()

for line in tqdm(words):
    if line.startswith("#"):
        continue

    line_split = line.strip().split(" ")
    if line_split[2] == "err":
        continue

    folder1 = line_split[0][:3]
    folder2 = "-".join(line_split[0].split("-")[:2])
    file_name = line_split[0] + ".png"
    label = line_split[-1].rstrip("\n").replace("|", " ")

    rel_path = os.path.join(sentences_folder_path, folder1, folder2, file_name)
    if not os.path.exists(rel_path):
        print(f"File not found: {rel_path}")
        continue

    dataset.append([rel_path, label])
    vocab.update(label)
    max_len = max(max_len, len(label))
import csv

# Load custom dataset
custom_images_path = os.path.join("Datasets", "handwriting", "images")
custom_labels_path = os.path.join("Datasets", "handwriting", "labels.csv")

custom_dataset = []
with open(custom_labels_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_path = os.path.join(custom_images_path, row['img_path'])
        label = row['text'].strip().lower()

        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}")
            continue

        custom_dataset.append([image_path, label])
        vocab.update(label)
        max_len = max(max_len, len(label))

# Merge with IAM
# To give your handwriting dataset more weight
dataset += custom_dataset * 20  

# Model configuration
configs = ModelConfigs()
configs.vocab = "".join(sorted(vocab))
print("🔤 Vocab used in this training run:")
print(repr(configs.vocab))  # 👈 this shows the full exact vocab
configs.max_text_length = max_len
configs.save()

# Create data provider
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=True),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
    ],
)

# Split dataset
train_data_provider, val_data_provider = data_provider.split(split=0.9)
train_data_provider.num_workers = configs.train_workers
val_data_provider.num_workers = configs.train_workers
# Data augmentation
train_data_provider.augmentors = [
    RandomBrightness(),
    RandomErodeDilate(),
    RandomRotate(),         # <— Add if not already
    RandomSharpen(),
]

# Rebuild and compile model
model = train_model(
    input_dim=(configs.height, configs.width, 3),
    output_dim=len(configs.vocab),
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
    loss=CTCloss(),
    metrics=[
        CERMetric(vocabulary=configs.vocab),
        WERMetric(vocabulary=configs.vocab)
    ],
    run_eagerly=False
)

# Load weights from previous training
h5_model_path = os.path.join(configs.model_path, "model0.0460.h5")
model.load_weights(h5_model_path)

# Callbacks
early_stopping = EarlyStopping(
    monitor="val_CER",
    mode="min",  
    patience=10,
    restore_best_weights=True
)
checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5", monitor="val_CER", verbose=1, save_best_only=True, mode="min")
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(log_dir=f"{configs.model_path}/logs", update_freq=1)
ReduceLROnPlateau= ReduceLROnPlateau(monitor="val_CER", factor=0.9, patience=5, min_lr=1e-6)

# Continue training from epoch 38
model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,  
    initial_epoch=58,           
    callbacks=[early_stopping, checkpoint, trainLogger, ReduceLROnPlateau, tb_callback]
)

# Export final model
saved_model_path = os.path.join(configs.model_path, "saved_model")
model.export(saved_model_path)

train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))