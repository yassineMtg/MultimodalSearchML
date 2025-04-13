# Milestone_4_test/tfx_pipeline/components/trainer.py

from typing import Any, Dict
import os
import tensorflow as tf
import tensorflow_transform as tft
import mlflow
import mlflow.tensorflow
from codecarbon import EmissionsTracker


def _input_fn(file_pattern, tf_transform_output, batch_size=32):
    def parse_tfrecord(example_proto):
        feature_spec = tf_transform_output.transformed_feature_spec()
        parsed_features = tf.io.parse_single_example(example_proto, feature_spec)
        label = parsed_features.pop('label')  # ESCI label: 0, 1, 2, 3
        return parsed_features, label

    files = tf.io.gfile.glob(file_pattern)
    dataset = tf.data.TFRecordDataset(files, compression_type="GZIP", num_parallel_reads=1)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=1)
    dataset = dataset.shuffle(500).batch(batch_size).prefetch(1)
    return dataset


def _build_keras_model(input_dim: int = 768) -> tf.keras.Model:
    inputs = {f"f{i}": tf.keras.Input(shape=(1,), name=f"f{i}") for i in range(input_dim)}
    x = tf.keras.layers.Concatenate()(list(inputs.values()))
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def run_fn(fn_args: Dict[str, Any]) -> None:
    # Track emissions
    tracker = EmissionsTracker(output_dir=fn_args.model_run_dir, log_level="error")
    tracker.start()

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, batch_size=32)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, batch_size=32)

    mlflow.set_experiment("milestone4_training_real_labels")
    mlflow.tensorflow.autolog(log_models=False)

    with mlflow.start_run():
        model = _build_keras_model()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(fn_args.model_run_dir, 'checkpoint_{epoch:02d}.h5'),
                save_weights_only=False,
                save_best_only=False,
                monitor='val_accuracy',
                verbose=1
            )
        ]

        model.fit(
            train_dataset,
            steps_per_epoch=fn_args.train_steps,
            validation_data=eval_dataset,
            validation_steps=fn_args.eval_steps,
            epochs=50,
            callbacks=callbacks
        )

        model.save(fn_args.serving_model_dir, save_format='tf')
        tracker.stop()
        tf.get_logger().info(f"✅ Model and emissions saved to: {fn_args.serving_model_dir}")
