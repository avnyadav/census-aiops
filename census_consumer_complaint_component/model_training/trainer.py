import tensorflow as tf
import tensorflow_hub as hub
from census_consumer_complaint_exception.exception import CensusConsumerException
import sys

import tensorflow_transform as tft
from census_consumer_complaint_component.feature_engineering.feature_engineering import transformed_name, \
    ONE_HOT_FEATURES, TEXT_FEATURES, LABEL_KEY


def _gzip_reader_fn(filenames):
    try:
        return tf.data.TFRecordDataset(filenames, compression_type="GZIP")
    except Exception as e:
        raise CensusConsumerException(e, sys) from e


def get_model():
    try:
        input_features = []

        for key, dim in ONE_HOT_FEATURES.items():
            input_features.append(
                tf.keras.Input(shape=(dim + 1), name=transformed_name(key))
            )

        input_texts = []

        for key in TEXT_FEATURES.keys():
            input_texts.append(
                tf.keras.Input(shape=(1,), name=transformed_name(key),
                               dtype=tf.string)
            )
        inputs = input_features + input_texts

        MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

        embed = hub.KerasLayer(MODULE_URL)
        reshaped_narrative = tf.reshape(input_texts[0], [-1])
        embed_narrative = embed(reshaped_narrative)
        deef_ff = tf.keras.layers.Reshape((512,), input_shape=(1, 512))(embed_narrative)

        deep = tf.keras.layers.Dense(256, activation="relu", )(deef_ff)
        deep = tf.keras.layers.Dense(64, activation="relu")(deep)
        deep = tf.keras.layers.Dense(16, activation="relu")(deep)

        wide_ff = tf.keras.layers.concatenate(input_features)

        wide = tf.keras.layers.Dense(16, activation="relu")(wide_ff)

        both = tf.keras.layers.concatenate([deep, wide])

        output = tf.keras.layers.Dense(1, activation="sigmoid")(both)

        keras_model = tf.keras.models.Model(inputs, output)
        keras_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                            loss="binary_crossentropy",
                            metrics=[
                                tf.keras.metrics.BinaryAccuracy(),
                                tf.keras.metrics.TruePositives()
                            ]
                            )
        return keras_model
    except Exception as e:
        raise CensusConsumerException(e, sys) from e


def input_fn(file_pattern, tf_transform_output, batch_size=32):
    try:
        transformed_feature_spec = (
            tf_transform_output.transformed_feature_spec().copy()
        )

        dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=file_pattern,
            batch_size=batch_size,
            features=transformed_feature_spec,
            reader=_gzip_reader_fn,
            label_key=transformed_name(LABEL_KEY))

        return dataset
    except Exception as e:
        raise CensusConsumerException(e, sys) from e


def get_serve_tf_examples_fn(model, tf_transform_output):
    try:
        model.tft_layer = tf_transform_output.transform_features_layer()

        @tf.function
        def serve_tf_example_fn(serialized_tf_example):
            feature_spec = tf_transform_output.raw_feature_spec()
            feature_spec.pop(LABEL_KEY)
            parsed_features = tf.io.parse_example(
                serialized_tf_example, feature_spec
            )

            transformed_features = model.tft_layer(parsed_features)
            outputs = model(transformed_features)

            return {'output': outputs}

        return serve_tf_example_fn
    except Exception as e:
        raise CensusConsumerException(e, sys) from e


def run_fn(fn_args):
    try:
        tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

        train_dataset = input_fn(fn_args.train_files, tf_transform_output)
        eval_dataset = input_fn(fn_args.eval_files, tf_transform_output)

        model = get_model()

        model.fit(
            train_dataset,
            steps_per_epoch=fn_args.train_steps,
            validation_data=eval_dataset,
            validation_steps=fn_args.eval_steps
        )

        signatures = {
            'serving_default': get_serve_tf_examples_fn(
                model, tf_transform_output
            ).get_concrete_function(
                tf.TensorSpec(
                    shape=[None],
                    dtype=tf.string,
                    name="examples"
                )
            )

        }

        model.save(fn_args.serving_model_dir,
                   save_format='tf',
                   signatures=signatures
                   )
    except Exception as e:
        raise CensusConsumerException(e, sys) from e
