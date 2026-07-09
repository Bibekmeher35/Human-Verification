"""Training script for the lightweight AI synthetic image detector using MobileNetV3Small.

Reads images from training/ai_detector_dataset/real/ and training/ai_detector_dataset/synthetic/
and saves the trained model to models/ai_detector_mobilenetv3.keras.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Set TF logging level to suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape: tuple[int, int, int]) -> keras.Model:
    """Builds MobileNetV3Small with a custom binary classification head."""
    base_model = keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling="avg",  # Applies Global Average Pooling directly
    )

    # Freeze base model layers
    base_model.trainable = False

    # Create classification head
    inputs = keras.Input(shape=input_shape)
    # MobileNetV3 expects inputs in range [-1, 1] or [0, 255] depending on model.
    # The Keras MobileNetV3 model has a built-in Rescaling layer or expects normalization.
    # To be safe and standard, we preprocess inputs to [0, 1] and let the model rescale internally.
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = base_model(x, training=False)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    return model


def get_image_files(directory: Path) -> list[Path]:
    """Finds all image files with supported extensions in the given directory."""
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return [p for p in directory.glob("**/*") if p.suffix.lower() in extensions]


def make_dataset(
    real_files: list[Path],
    fake_files: list[Path],
    img_size: int,
    batch_size: int,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Creates a tf.data.Dataset from lists of real and fake file paths."""
    file_paths = [str(p) for p in real_files + fake_files]
    # Real = 0.0, Fake/Synthetic = 1.0
    labels = [0.0] * len(real_files) + [1.0] * len(fake_files)

    path_ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    def load_and_preprocess(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image_bytes = tf.io.read_file(path)
        # decode_image handles JPEG/PNG/WEBP/BMP
        image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, [img_size, img_size])
        image.set_shape([img_size, img_size, 3])
        return image, tf.expand_dims(label, axis=-1)

    ds = path_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV3Small AI Generated Image Detector")
    parser.add_argument("--dataset-dir", default="training/ai_detector_dataset/real-vs-fake", help="Dataset root directory")
    parser.add_argument("--limit", type=int, default=2000, help="Max image count to load per class for training (-1 for no limit)")
    parser.add_argument("--epochs-init", type=int, default=10, help="Epochs to train head layers")
    parser.add_argument("--epochs-fine", type=int, default=5, help="Epochs for fine-tuning")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--image-size", type=int, default=224, help="Target image dimension (height and width)")
    parser.add_argument("--lr-init", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--lr-fine", type=float, default=1e-5, help="Fine-tuning learning rate")
    parser.add_argument("--output-model", default="models/ai_detector_mobilenetv3.keras", help="Path to save trained model")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    
    # Locate directories (support both flat structure and real-vs-fake train/valid structure)
    train_real_dir = dataset_path / "train" / "real"
    train_fake_dir = dataset_path / "train" / "fake"
    if not train_real_dir.exists():
        # Fallback to direct directories
        train_real_dir = dataset_path / "real"
        train_fake_dir = dataset_path / "synthetic"

    valid_real_dir = dataset_path / "valid" / "real"
    valid_fake_dir = dataset_path / "valid" / "fake"
    if not valid_real_dir.exists():
        # Fallback to train/validation split from same dir
        valid_real_dir = None
        valid_fake_dir = None

    if not train_real_dir.exists() or not train_fake_dir.exists():
        print(f"Error: Could not locate real/fake dataset folders under {dataset_path}")
        return

    print("Scanning dataset directories for images...")
    train_real_files = get_image_files(train_real_dir)
    train_fake_files = get_image_files(train_fake_dir)
    print(f"Discovered: {len(train_real_files)} real and {len(train_fake_files)} fake images for training.")

    if len(train_real_files) == 0 or len(train_fake_files) == 0:
        print("Error: No training images found. Please check paths and file formats.")
        return

    # Apply limits if requested
    if args.limit > 0:
        print(f"Applying limit: Loading max {args.limit} images per class for training.")
        # Shuffle files locally before slicing to get a diverse sample
        import random
        random.seed(42)
        random.shuffle(train_real_files)
        random.shuffle(train_fake_files)
        train_real_files = train_real_files[:args.limit]
        train_fake_files = train_fake_files[:args.limit]

    # Resolve validation files
    if valid_real_dir and valid_real_dir.exists() and valid_fake_dir.exists():
        valid_real_files = get_image_files(valid_real_dir)
        valid_fake_files = get_image_files(valid_fake_dir)
        print(f"Discovered: {len(valid_real_files)} real and {len(valid_fake_files)} fake images for validation.")
        if args.limit > 0:
            val_limit = max(1, int(args.limit * 0.2))
            random.shuffle(valid_real_files)
            random.shuffle(valid_fake_files)
            valid_real_files = valid_real_files[:val_limit]
            valid_fake_files = valid_fake_files[:val_limit]
            print(f"Applying limit: Loading max {val_limit} images per class for validation.")
    else:
        # Split train files to create validation files
        split_idx_real = int(len(train_real_files) * 0.8)
        split_idx_fake = int(len(train_fake_files) * 0.8)
        
        valid_real_files = train_real_files[split_idx_real:]
        train_real_files = train_real_files[:split_idx_real]
        
        valid_fake_files = train_fake_files[split_idx_fake:]
        train_fake_files = train_fake_files[:split_idx_fake]
        print(f"Created validation split: {len(train_real_files)} train, {len(valid_real_files)} val per class.")

    input_shape = (args.image_size, args.image_size, 3)

    print("Building datasets...")
    train_ds = make_dataset(train_real_files, train_fake_files, args.image_size, args.batch_size, shuffle=True)
    val_ds = make_dataset(valid_real_files, valid_fake_files, args.image_size, args.batch_size, shuffle=False)

    print("Building model structure...")
    model = build_model(input_shape)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr_init),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Phase 1: Train head layers
    print(f"\n--- Phase 1: Training classification head for {args.epochs_init} epochs ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_init,
    )

    # Phase 2: Fine-tune model
    if args.epochs_fine > 0:
        print(f"\n--- Phase 2: Fine-tuning top model layers for {args.epochs_fine} epochs ---")
        # Unfreeze the MobileNetV3 base model
        model.layers[2].trainable = True
        # Keep early layers frozen, unfreeze top 15 layers
        for layer in model.layers[2].layers[:-15]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.lr_fine),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs_fine,
        )

    # Save model
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving final model to {output_path}...")
    model.save(str(output_path))
    print("Training complete!")


if __name__ == "__main__":
    main()
