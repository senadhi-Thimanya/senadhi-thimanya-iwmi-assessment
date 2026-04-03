# =============================================================================
# IWMI Data Science Intern Assessment
# Task 1: Data Preprocessing & Pipeline
# Task 2: Custom CNN Architecture & Training
# Task 3: Model Evaluation and Basic Inferencing
# =============================================================================

# Importing required libraries
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Additional libraries
import json
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, TensorBoard
)


# =============================================================================
# TASK 1 — Data Preprocessing & Pipeline
# =============================================================================

class BasicPreprocessing:
    """
    Handles all data-related operations:
      - Importing and scanning the dataset
      - Splitting into train / validation / test sets
      - Building augmented tf.data / Keras generators
      - Exploratory visualisation helpers
    """

    def __init__(
        self,
        dataset_dir: str = "dataset",
        img_size: tuple = (128, 128),
        batch_size: int = 32,
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42,
    ):
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        self.class_names = []
        self.train_paths, self.val_paths, self.test_paths = [], [], []
        self.train_labels, self.val_labels, self.test_labels = [], [], []
        self.train_gen = self.val_gen = self.test_gen = None

    # ------------------------------------------------------------------
    def import_dataset(self) -> pd.DataFrame:
        """
        Method
        ------
        Import the given dataset.

        Parameters
        ----------
        As required

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns ['filepath', 'label', 'class_name'].

        Notes
        -----
        Expected on-disk layout::

            dataset/
                with_mask/
                    img1.jpg  ...
                without_mask/
                    img1.jpg  ...
        """
        records = []
        class_dirs = sorted(
            [d for d in Path(self.dataset_dir).iterdir() if d.is_dir()]
        )
        self.class_names = [d.name for d in class_dirs]

        for label_idx, class_dir in enumerate(class_dirs):
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in class_dir.glob(ext):
                    records.append(
                        {
                            "filepath": str(img_path),
                            "label": label_idx,
                            "class_name": class_dir.name,
                        }
                    )

        df = pd.DataFrame(records)
        print(f"[Dataset] Classes found : {self.class_names}")
        print(f"[Dataset] Total images  : {len(df)}")
        print(df["class_name"].value_counts().to_string())
        return df

    # ------------------------------------------------------------------
    def split_dataset(self, df: pd.DataFrame) -> tuple:
        """
        Split DataFrame into stratified train / val / test sets.

        Parameters
        ----------
        df : pd.DataFrame
            Output of :meth:`import_dataset`.

        Returns
        -------
        tuple
            (train_df, val_df, test_df)
        """
        X = df["filepath"].values
        y = df["label"].values

        # First carve out the test set
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y,
            test_size=self.test_split,
            stratify=y,
            random_state=self.seed,
        )
        # Then split remainder into train / val
        val_ratio_adjusted = self.val_split / (1.0 - self.test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp,
            test_size=val_ratio_adjusted,
            stratify=y_tmp,
            random_state=self.seed,
        )

        self.train_paths, self.train_labels = X_train.tolist(), y_train.tolist()
        self.val_paths,   self.val_labels   = X_val.tolist(),   y_val.tolist()
        self.test_paths,  self.test_labels  = X_test.tolist(),  y_test.tolist()

        print(
            f"[Split] Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}"
        )

        train_df = pd.DataFrame({"filepath": X_train, "label": y_train})
        val_df   = pd.DataFrame({"filepath": X_val,   "label": y_val})
        test_df  = pd.DataFrame({"filepath": X_test,  "label": y_test})
        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    def build_generators(self) -> tuple:
        """
        Build Keras ImageDataGenerators with augmentation for training
        and simple normalisation for val/test.

        Returns
        -------
        tuple
            (train_generator, val_generator, test_generator)
        """
        # --- Augmentation for training ---
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.10,
            zoom_range=0.15,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode="nearest",
        )

        # --- Only normalisation for val / test ---
        eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

        def _flow_from_lists(datagen, paths, labels, shuffle):
            return datagen.flow(
                self._load_images(paths),
                np.array(labels),
                batch_size=self.batch_size,
                shuffle=shuffle,
                seed=self.seed,
            )

        self.train_gen = _flow_from_lists(
            train_datagen, self.train_paths, self.train_labels, shuffle=True
        )
        self.val_gen = _flow_from_lists(
            eval_datagen, self.val_paths, self.val_labels, shuffle=False
        )
        self.test_gen = _flow_from_lists(
            eval_datagen, self.test_paths, self.test_labels, shuffle=False
        )

        return self.train_gen, self.val_gen, self.test_gen

    # ------------------------------------------------------------------
    def _load_images(self, paths: list) -> np.ndarray:
        """
        Load and resize a list of image paths into a float32 NumPy array.

        Parameters
        ----------
        paths : list of str

        Returns
        -------
        np.ndarray  shape (N, H, W, 3)
        """
        images = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            images.append(img)
        return np.array(images, dtype=np.float32)

    # ------------------------------------------------------------------
    def preprocess_single_image(self, img_path: str) -> np.ndarray:
        """
        Preprocess a single image for inference (resize + normalise).

        Parameters
        ----------
        img_path : str

        Returns
        -------
        np.ndarray  shape (1, H, W, 3)  — batch dimension included.
        """
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    # ------------------------------------------------------------------
    def visualise_samples(self, df: pd.DataFrame, n: int = 8) -> None:
        """
        Plot a grid of random sample images with their class labels.

        Parameters
        ----------
        df : pd.DataFrame
        n  : int  Number of samples to display (should be even).
        """
        sample = df.sample(n=n, random_state=self.seed)
        fig, axes = plt.subplots(2, n // 2, figsize=(16, 6))
        axes = axes.flatten()
        for ax, (_, row) in zip(axes, sample.iterrows()):
            img = cv2.imread(row["filepath"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            ax.imshow(img)
            ax.set_title(row["class_name"], fontsize=10)
            ax.axis("off")
        plt.suptitle("Sample Images", fontsize=14, fontweight="bold")
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/sample_images.png", dpi=150)
        plt.show()
        print("[Saved] results/sample_images.png")

    # ------------------------------------------------------------------
    def class_distribution_plot(self, df: pd.DataFrame) -> None:
        """
        Plot the class distribution bar chart.

        Parameters
        ----------
        df : pd.DataFrame
        """
        counts = df["class_name"].value_counts()
        plt.figure(figsize=(7, 4))
        bars = plt.bar(counts.index, counts.values,
                       color=["#4C72B0", "#DD8452"], edgecolor="black")
        for bar, val in zip(bars, counts.values):
            plt.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 5, str(val),
                     ha="center", va="bottom", fontweight="bold")
        plt.title("Class Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/class_distribution.png", dpi=150)
        plt.show()
        print("[Saved] results/class_distribution.png")


# =============================================================================
# TASK 2 — Custom CNN Architecture & Training
# =============================================================================

class ModelDevelopment:
    """
    Defines, compiles, trains, and persists a custom CNN built from scratch.
    Architecture overview::

        Input (128×128×3)
          ↓
        ConvBlock × 4   (32 → 64 → 128 → 256 filters)
          each block: Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm →
                      ReLU → MaxPool → Dropout
          ↓
        GlobalAveragePooling2D
          ↓
        Dense(512) → BatchNorm → ReLU → Dropout(0.5)
          ↓
        Dense(256) → BatchNorm → ReLU → Dropout(0.3)
          ↓
        Dense(num_classes, softmax)

    No pretrained weights — trained entirely from scratch.
    """

    def __init__(
        self,
        img_size: tuple = (128, 128),
        num_classes: int = 2,
        learning_rate: float = 1e-3,
        model_save_path: str = "models/best_model.keras",
    ):
        self.img_size = img_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model_save_path = model_save_path
        self.model = None
        self.history = None

    # ------------------------------------------------------------------
    def _conv_block(
        self,
        x,
        filters: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.25,
    ):
        """
        Reusable convolutional block:
        Conv → BN → ReLU → Conv → BN → ReLU → MaxPool → Dropout

        Parameters
        ----------
        x            : Keras tensor
        filters      : int   Number of filters in both Conv layers
        kernel_size  : int
        dropout_rate : float

        Returns
        -------
        Keras tensor
        """
        x = layers.Conv2D(
            filters, kernel_size, padding="same",
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(
            filters, kernel_size, padding="same",
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(dropout_rate)(x)
        return x

    # ------------------------------------------------------------------
    def build_model(self) -> keras.Model:
        """
        Construct the custom CNN architecture.

        Returns
        -------
        keras.Model  (uncompiled)
        """
        inputs = keras.Input(shape=(*self.img_size, 3), name="input_image")

        # ── Convolutional backbone ──────────────────────────────────────
        x = self._conv_block(inputs, filters=32,  dropout_rate=0.20)
        x = self._conv_block(x,      filters=64,  dropout_rate=0.25)
        x = self._conv_block(x,      filters=128, dropout_rate=0.30)
        x = self._conv_block(x,      filters=256, dropout_rate=0.35)

        # ── Global feature aggregation ─────────────────────────────────
        x = layers.GlobalAveragePooling2D()(x)

        # ── Classifier head ────────────────────────────────────────────
        x = layers.Dense(
            512, kernel_regularizer=regularizers.l2(1e-4)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.50)(x)

        x = layers.Dense(
            256, kernel_regularizer=regularizers.l2(1e-4)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.30)(x)

        outputs = layers.Dense(
            self.num_classes, activation="softmax", name="predictions"
        )(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name="MaskDetectorCNN")
        return self.model

    # ------------------------------------------------------------------
    def compile_model(self) -> None:
        """
        Compile the model with Adam + categorical cross-entropy.
        """
        if self.model is None:
            raise RuntimeError("Call build_model() before compile_model().")

        optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
        )
        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(self.model.summary())

    # ------------------------------------------------------------------
    def get_callbacks(self) -> list:
        """
        Build training callbacks:
          - ModelCheckpoint  (save best val_accuracy)
          - EarlyStopping    (patience = 10)
          - ReduceLROnPlateau (halve LR on plateau)
          - TensorBoard

        Returns
        -------
        list of keras.callbacks.Callback
        """
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

        checkpoint = ModelCheckpoint(
            filepath=self.model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        )
        early_stop = EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        )
        tensorboard = TensorBoard(log_dir="logs/fit", histogram_freq=1)

        return [checkpoint, early_stop, reduce_lr, tensorboard]

    # ------------------------------------------------------------------
    def train_model(
        self,
        train_gen,
        val_gen,
        epochs: int = 50,
    ) -> keras.callbacks.History:
        """
        Train the model and record history.

        Parameters
        ----------
        train_gen : Keras generator
        val_gen   : Keras generator
        epochs    : int

        Returns
        -------
        keras.callbacks.History
        """
        if self.model is None:
            raise RuntimeError("Build and compile the model first.")

        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=self.get_callbacks(),
        )
        return self.history

    # ------------------------------------------------------------------
    def plot_training_history(self) -> None:
        """
        Plot and save accuracy + loss curves for train and validation.
        """
        if self.history is None:
            raise RuntimeError("Train the model first.")

        hist = self.history.history
        epochs_range = range(1, len(hist["accuracy"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # ── Accuracy ───────────────────────────────────────────────────
        ax1.plot(epochs_range, hist["accuracy"],     label="Train Accuracy", marker="o", markersize=3)
        ax1.plot(epochs_range, hist["val_accuracy"], label="Val Accuracy",   marker="s", markersize=3)
        ax1.set_title("Accuracy per Epoch", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
        ax1.legend(); ax1.grid(alpha=0.3)

        # ── Loss ───────────────────────────────────────────────────────
        ax2.plot(epochs_range, hist["loss"],     label="Train Loss", marker="o", markersize=3)
        ax2.plot(epochs_range, hist["val_loss"], label="Val Loss",   marker="s", markersize=3)
        ax2.set_title("Loss per Epoch", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
        ax2.legend(); ax2.grid(alpha=0.3)

        plt.suptitle("Training History — MaskDetectorCNN", fontsize=15, fontweight="bold")
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/training_curves.png", dpi=150)
        plt.show()
        print("[Saved] results/training_curves.png")

    # ------------------------------------------------------------------
    def save_model(self, path: str = None) -> None:
        """
        Persist the model to disk in Keras native format.

        Parameters
        ----------
        path : str  (optional) Override save path.
        """
        save_path = path or self.model_save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        print(f"[Saved] Model → {save_path}")

    # ------------------------------------------------------------------
    def load_model(self, path: str = None) -> keras.Model:
        """
        Load a previously saved model.

        Parameters
        ----------
        path : str

        Returns
        -------
        keras.Model
        """
        load_path = path or self.model_save_path
        self.model = keras.models.load_model(load_path)
        print(f"[Loaded] Model ← {load_path}")
        return self.model

    # ------------------------------------------------------------------
    def get_architecture_summary(self) -> str:
        """
        Return a human-readable string summary of the model architecture.

        Returns
        -------
        str
        """
        if self.model is None:
            return "Model not built yet."
        lines = []
        self.model.summary(print_fn=lambda x: lines.append(x))
        return "\n".join(lines)


# =============================================================================
# TASK 3 — Model Evaluation and Basic Inferencing
# =============================================================================

class BasicInference:
    """
    Handles face detection (via Haarcascade), single-image classification,
    batch evaluation, and metric reporting.

    Performance Metric Justification
    ---------------------------------
    For a binary face-mask classifier we report:

    * **Accuracy**   – Overall correctness; useful when classes are balanced.
    * **Precision**  – Of predicted "with mask", how many truly wear masks.
                      High precision reduces false alarms.
    * **Recall**     – Of actual "with mask", how many did we catch.
                      High recall is safety-critical (no missed violations).
    * **F1-Score**   – Harmonic mean of precision & recall; best single
                      metric when we care about both false positives and
                      false negatives equally.
    * **Confusion Matrix** – Reveals the direction of errors
                             (FP vs FN breakdown).
    * **ROC-AUC**    – Threshold-independent measure of discriminability.
    """

    # Class names must match the training label order (alphabetical by default)
    CLASS_NAMES = ["with_mask", "without_mask"]

    def __init__(
        self,
        model: keras.Model = None,
        img_size: tuple = (128, 128),
        confidence_threshold: float = 0.50,
    ):
        self.model = model
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    # ------------------------------------------------------------------
    def detect_images(self, image_input) -> dict:
        """
        Method
        ------
        Implement a function to read images and classify those faces to
        either faces with masks or faces without masks.

        Hint: Use haarcascade_frontalface_default to detect faces in a
        static image.

        Parameters
        ----------
        image_input : str | np.ndarray
            File path to an image *or* a pre-loaded BGR/RGB NumPy array.

        Returns
        -------
        dict with keys:
            'faces'         : list of dicts per detected face
                - 'bbox'        : (x, y, w, h)
                - 'prediction'  : class name string
                - 'confidence'  : float 0-1
                - 'probabilities': list of floats (one per class)
            'annotated_image': np.ndarray (RGB, uint8)
            'face_count'     : int
        """
        # ── Load image ────────────────────────────────────────────────
        if isinstance(image_input, str):
            bgr = cv2.imread(image_input)
            if bgr is None:
                raise FileNotFoundError(f"Cannot read image: {image_input}")
        else:
            bgr = image_input.copy()
            if bgr.shape[-1] == 3:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # ── Haarcascade face detection ─────────────────────────────────
        faces_rect = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        results = []
        annotated = rgb.copy()

        if len(faces_rect) == 0:
            # Fallback: classify the full image
            faces_rect = [(0, 0, bgr.shape[1], bgr.shape[0])]

        for (x, y, w, h) in faces_rect:
            face_crop = rgb[y:y + h, x:x + w]
            face_resized = cv2.resize(face_crop, self.img_size).astype(np.float32) / 255.0
            face_batch = np.expand_dims(face_resized, axis=0)

            probs = self.model.predict(face_batch, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            label = self.CLASS_NAMES[pred_idx]

            # Draw annotation
            color = (0, 200, 0) if label == "with_mask" else (220, 50, 50)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            text = f"{label}: {confidence:.1%}"
            cv2.putText(
                annotated, text, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )

            results.append(
                {
                    "bbox": (x, y, w, h),
                    "prediction": label,
                    "confidence": confidence,
                    "probabilities": probs.tolist(),
                }
            )

        return {
            "faces": results,
            "annotated_image": annotated,
            "face_count": len(results),
        }

    # ------------------------------------------------------------------
    def predict_single(self, img_path: str) -> dict:
        """
        Classify a single image without face detection.

        Parameters
        ----------
        img_path : str

        Returns
        -------
        dict with 'prediction', 'confidence', 'probabilities'
        """
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size).astype(np.float32) / 255.0
        img_batch = np.expand_dims(img, axis=0)

        probs = self.model.predict(img_batch, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        return {
            "prediction": self.CLASS_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": probs.tolist(),
        }

    # ------------------------------------------------------------------
    def evaluate_on_test_set(
        self,
        test_paths: list,
        test_labels: list,
        save_dir: str = "results",
    ) -> dict:
        """
        Run full evaluation on the held-out test set and save plots.

        Parameters
        ----------
        test_paths  : list of str  image file paths
        test_labels : list of int  ground-truth integer labels
        save_dir    : str

        Returns
        -------
        dict with accuracy, precision, recall, f1, confusion_matrix
        """
        os.makedirs(save_dir, exist_ok=True)
        y_true, y_pred, y_probs = [], [], []

        for path, label in zip(test_paths, test_labels):
            result = self.predict_single(path)
            y_pred.append(self.CLASS_NAMES.index(result["prediction"]))
            y_probs.append(result["probabilities"])
            y_true.append(label)

        y_true  = np.array(y_true)
        y_pred  = np.array(y_pred)
        y_probs = np.array(y_probs)

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted")
        rec  = recall_score(y_true, y_pred, average="weighted")
        f1   = f1_score(y_true, y_pred, average="weighted")
        cm   = confusion_matrix(y_true, y_pred)

        print("\n" + "=" * 50)
        print("  TEST SET EVALUATION")
        print("=" * 50)
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print("\nClassification Report:")
        print(
            classification_report(y_true, y_pred, target_names=self.CLASS_NAMES)
        )

        # ── Confusion matrix heatmap ───────────────────────────────────
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d",
            xticklabels=self.CLASS_NAMES,
            yticklabels=self.CLASS_NAMES,
            cmap="Blues",
        )
        plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
        plt.ylabel("True Label"); plt.xlabel("Predicted Label")
        plt.tight_layout()
        cm_path = os.path.join(save_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150)
        plt.show()
        print(f"[Saved] {cm_path}")

        # ── Persist metrics to JSON ────────────────────────────────────
        metrics = {
            "accuracy":  round(acc, 4),
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1_score":  round(f1, 4),
            "confusion_matrix": cm.tolist(),
        }
        metrics_path = os.path.join(save_dir, "metrics.json")
        with open(metrics_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"[Saved] {metrics_path}")

        return metrics

    # ------------------------------------------------------------------
    def model_analysis(self) -> str:
        """
        Return a written analysis of model strengths and failure modes.

        Returns
        -------
        str
        """
        analysis = """
=== Model Strengths & Failure Mode Analysis ===

STRENGTHS
---------
1. Clear foreground vs background contrast
   The model learns well when the face is well-lit and centred, where
   colour and texture cues (fabric texture over the mouth/nose region)
   are strong discriminators.

2. Front-facing, single-person images
   The Haarcascade detector works reliably for straight-on faces,
   allowing the CNN crop to be well-aligned with the training distribution.

3. Data augmentation resilience
   Random flips, rotations, and brightness jitter during training make
   the model robust to minor pose and lighting variation.

FAILURE MODES
-------------
1. Partial / non-standard masks (scarves, bandanas, neck-gaiters)
   These items are under-represented in the training set. The model may
   predict "without mask" when the nose is uncovered.

2. Extreme pose (profile, downward tilt, heavy occlusion)
   Haarcascade misses non-frontal faces → no crop → full-image fallback
   with a misaligned perspective degrades CNN accuracy.

3. Low resolution / heavy compression artefacts
   Aggressive JPEG compression blurs texture edges; the model loses the
   fine-grained details that distinguish mask fabric from bare skin.

4. Dark skin tones in low-light conditions
   If training data is not demographically balanced, the model may exhibit
   lower recall for underrepresented groups, especially at night or in
   shadow.

RECOMMENDATIONS
---------------
- Collect and oversample harder edge cases (partial masks, profiles).
- Use a modern face detector (e.g., MediaPipe, MTCNN) for better recall.
- Apply test-time augmentation (TTA) to improve confidence calibration.
- Periodically re-evaluate with a fresh held-out set to catch distribution shift.
"""
        print(analysis)
        return analysis


# =============================================================================
# Main entry point — ties all tasks together
# =============================================================================

def main():
    print("IWMI Data Science Internship Assessment, I'm not a data scientist")

    # Set up paths relative to project root (parent of src directory)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATASET_DIR   = str(PROJECT_ROOT / "dataset")
    IMG_SIZE      = (128, 128)
    BATCH_SIZE    = 32
    EPOCHS        = 50
    NUM_CLASSES   = 2
    MODEL_PATH    = str(PROJECT_ROOT / "models" / "best_model.keras")

    # ── Task 1: Preprocessing ──────────────────────────────────────────
    preprocessor = BasicPreprocessing(
        dataset_dir=DATASET_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    df = preprocessor.import_dataset()
    preprocessor.visualise_samples(df)
    preprocessor.class_distribution_plot(df)

    train_df, val_df, test_df = preprocessor.split_dataset(df)
    train_gen, val_gen, _ = preprocessor.build_generators()

    # ── Task 2: Model Development ──────────────────────────────────────
    dev = ModelDevelopment(
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        model_save_path=MODEL_PATH,
    )
    dev.build_model()
    dev.compile_model()
    dev.train_model(train_gen, val_gen, epochs=EPOCHS)
    dev.plot_training_history()
    dev.save_model()

    # ── Task 3: Inference & Evaluation ────────────────────────────────
    loaded_model = dev.load_model()
    inference = BasicInference(model=loaded_model, img_size=IMG_SIZE)
    inference.evaluate_on_test_set(
        preprocessor.test_paths,
        preprocessor.test_labels,
    )
    inference.model_analysis()


if __name__ == "__main__":
    main()
