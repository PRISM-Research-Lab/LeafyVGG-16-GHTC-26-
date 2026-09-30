import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

import time

# =========================
# PATHS
# =========================
base_path = r"C:/WVSU_Documents/Scripts/Tomato_Disease_DetectionV1/Tomato_Lettuce_Dataset/Variant-a(Multiclass Classification)"
train_path = os.path.join(base_path, "train")
test_path  = os.path.join(base_path, "test")

# =========================
# CLASS NAMES
# =========================
class_names = sorted(os.listdir(train_path))
print("Classes:", class_names)

# =========================
# TEST DATA
# =========================
test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

test_gen = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=class_names,
    shuffle=False
)

# =========================
# LOAD MODEL
# =========================
# model = tf.keras.models.load_model("best_resnet50_50Epoch.keras")
model = tf.keras.models.load_model("C:/WVSU_Documents/Scripts/Tomato_Disease_DetectionV1/pretrained_models/best_vgg16_250Epoch.keras")
print("Model loaded")

# =========================
# LOAD ALL DATA
# =========================
test_gen.reset()

imgs, lbls = [], []

for _ in range(len(test_gen)):
    x, y = next(test_gen)
    imgs.append(x)
    lbls.append(y)

all_images = np.concatenate(imgs, axis=0)
all_labels = np.concatenate(lbls, axis=0)

y_true = np.argmax(all_labels, axis=1)

print("Total samples:", len(y_true))

# =========================
# CLEAN ACCURACY
# =========================
clean_pred = np.argmax(model.predict(all_images, verbose=0), axis=1)
clean_acc = accuracy_score(y_true, clean_pred)

print("\n===== CLEAN ACCURACY =====")
print(f"Accuracy : {accuracy_score(y_true, clean_pred):.4f}")
print(f"Precision: {precision_score(y_true, clean_pred, average='macro'):.4f}")
print(f"Recall   : {recall_score(y_true, clean_pred, average='macro'):.4f}")
print(f"F1-score : {f1_score(y_true, clean_pred, average='macro'):.4f}")

print("\nClassification Report (Clean):")
print(classification_report(y_true, clean_pred, target_names=class_names))

print("\nConfusion Matrix (Clean):")
print(confusion_matrix(y_true, clean_pred))

# =====================================================
# 🔥 FGSM (BATCH SAFE)
# =====================================================
def fgsm_attack(model, images, labels, epsilon, batch_size=16):

    adv_all = []

    for i in range(0, len(images), batch_size):

        x = tf.convert_to_tensor(images[i:i+batch_size], dtype=tf.float32)
        y = tf.convert_to_tensor(labels[i:i+batch_size], dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x)

            preds = model(x, training=False)

            loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(y, preds)
            )

        grad = tape.gradient(loss, x)

        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)

        adv = x + epsilon * tf.sign(grad)

        adv_all.append(adv.numpy())

    return np.concatenate(adv_all, axis=0)

# =====================================================
# 🔥 PGD (BATCH SAFE)
# =====================================================
def pgd_attack(model, images, labels,epsilon, alpha, steps, batch_size=16):

    adv_final = []

    for i in range(0, len(images), batch_size):

        orig = tf.convert_to_tensor(images[i:i+batch_size], dtype=tf.float32)
        adv = tf.identity(orig)
        y = tf.convert_to_tensor(labels[i:i+batch_size], dtype=tf.float32)

        for _ in range(steps):

            with tf.GradientTape() as tape:
                tape.watch(adv)

                preds = model(adv, training=False)

                loss = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(y, preds)
                )

            grad = tape.gradient(loss, adv)

            grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)

            adv = adv + alpha * tf.sign(grad)

            adv = tf.clip_by_value(adv, orig - epsilon, orig + epsilon)

        adv_final.append(adv.numpy())

    return np.concatenate(adv_final, axis=0)

# =========================
# FULL DATASET EVALUATION
# =========================
# FGSM parameters
epsilon_fgsm=[0.005, 0.01, 0.03, 0.05] # prove as many parameter here
#PGD parameters
epsilon_pgd=0.05
alpha=0.005
steps=10

for eps in epsilon_fgsm:

    adv = fgsm_attack(model, all_images, all_labels, eps)

    y_pred_fgsm = np.argmax(model.predict(adv, verbose=0), axis=1)

    acc = accuracy_score(y_true, y_pred_fgsm)

    print("\n===== FGSM RESULTS =====")
    print("Epsilon:",eps)
    print(f"Accuracy : {accuracy_score(y_true, y_pred_fgsm) :.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_fgsm, average='macro'):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred_fgsm, average='macro'):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred_fgsm, average='macro'):.4f}")

    print("\nClassification Report (FGSM):")
    print(classification_report(y_true, y_pred_fgsm, target_names=class_names))
    print("\nConfusion Matrix (FGSM):")
    print(confusion_matrix(y_true, y_pred_fgsm))

# =========================
# PGD EVALUATION
# =========================

adv_pgd = pgd_attack(model, all_images, all_labels,epsilon_pgd, alpha, steps)

y_pred_pgd = np.argmax(model.predict(adv_pgd, verbose=0), axis=1)

# print("PGD Accuracy:", pgd_acc)

print("\n===== PGD RESULTS =====")
print(f"Epsilon: {epsilon_pgd}, Alpha: {alpha}, Steps: {steps}")
print(f"Accuracy : {accuracy_score(y_true, y_pred_pgd):.4f}")
print(f"Precision: {precision_score(y_true, y_pred_pgd, average='macro'):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred_pgd, average='macro'):.4f}")
print(f"F1-score : {f1_score(y_true, y_pred_pgd, average='macro'):.4f}")

print("\nClassification Report (PGD):")
print(classification_report(y_true, y_pred_pgd, target_names=class_names))

print("\nConfusion Matrix (PGD):")
print(confusion_matrix(y_true, y_pred_pgd))

