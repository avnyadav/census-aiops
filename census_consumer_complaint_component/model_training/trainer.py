from typing import List, Text
from tfx import v1 as tfx
import tensorflow as tf
import tensorflow_hub as hub
from tfx_bsl.public import tfxio
from census_consumer_complaint_exception.exception import CensusConsumerException
import sys
from tensorflow_transform import TFTransformOutput
import tensorflow_transform as tft
from census_consumer_complaint_component.feature_engineering.feature_engineering import transformed_name, \
    ONE_HOT_FEATURES, TEXT_FEATURES, LABEL_KEY


def _gzip_reader_fn(filenames):
    try:
        return tf.data.TFRecordDataset(filenames, compression_type="GZIP")
    except Exception as e:
        raise CensusConsumerException(e, sys) from e


def _build_keras_model(tf_transform_output: TFTransformOutput) -> tf.keras.Model:
    try:
        feature_spec = tf_transform_output.transformed_feature_spec().copy()

        feature_spec.pop(transformed_name(LABEL_KEY))

        inputs = {}

        deep = None
        for key, spec in feature_spec.items():
            print(f"building model with column{key}")
            if key in list(map(lambda x: transformed_name(x), TEXT_FEATURES.keys())):
                continue
                print(f"Skipping column{key}")
                inputs[key] = tf.keras.Input(shape=(1,), name=transformed_name(key),
                                             dtype=tf.string)
                MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

                embed = hub.KerasLayer(MODULE_URL)
                reshaped_narrative = tf.reshape(inputs[key], [-1])
                embed_narrative = embed(reshaped_narrative)
                deef_ff = tf.keras.layers.Reshape((512,), input_shape=(1, 512))(embed_narrative)
                deep = tf.keras.layers.Dense(256, activation="relu", )(deef_ff)
                deep = tf.keras.layers.Dense(64, activation="relu")(deep)
                deep = tf.keras.layers.Dense(16, activation="relu")(deep)
                continue
            if isinstance(spec, tf.io.VarLenFeature):
                inputs[key] = tf.keras.layers.Input(
                    shape=[None], name=key, dtype=spec.dtype, sparse=True
                )
            elif isinstance(spec, tf.io.FixedLenFeature):
                inputs[key] = tf.keras.layers.Input(
                    shape=spec.shape or [1], name=key, dtype=spec.dtype, )
            else:
                raise ValueError('Spec type is not supported:', key, spec)

        wide_ff = tf.keras.layers.concatenate([inputs[column] for column in filter(
            lambda x: x not in transformed_name('Consumer complaint narrative'), feature_spec.keys())])

        wide = tf.keras.layers.Dense(16, activation="relu")(wide_ff)

        #both = tf.keras.layers.concatenate([deep, wide])
        output = tf.keras.layers.Dense(100, activation='relu')(wide)
        output = tf.keras.layers.Dense(70, activation='relu')(output)
        output = tf.keras.layers.Dense(50, activation='relu')(output)
        output = tf.keras.layers.Dense(20, activation='relu')(output)
        output = tf.keras.layers.Dense(1)(output)
        # output = tf.keras.layers.Concatenate()(tf.nest.flatten(inputs))
        # output = tf.keras.layers.Dense(100, activation='relu')(output)
        # output = tf.keras.layers.Dense(70, activation='relu')(output)
        # output = tf.keras.layers.Dense(50, activation='relu')(output)
        # output = tf.keras.layers.Dense(20, activation='relu')(output)
        # output = tf.keras.layers.Dense(1)(output)
        return tf.keras.Model(inputs=inputs, outputs=output)
    except Exception as e:
        raise CensusConsumerException(e, sys) from e


# def get_model():
#     try:
#         input_features = []
#
#         for key, dim in ONE_HOT_FEATURES.items():
#             input_features.append(
#                 tf.keras.Input(shape=(dim + 1), name=transformed_name(key))
#             )
#
#         input_texts = []
#
#         for key in TEXT_FEATURES.keys():
#             input_texts.append(
#                 tf.keras.Input(shape=(1,), name=transformed_name(key),
#                                dtype=tf.string)
#             )
#         inputs = input_features + input_texts
#
#         MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
#
#         embed = hub.KerasLayer(MODULE_URL)
#         reshaped_narrative = tf.reshape(input_texts[0], [-1])
#         embed_narrative = embed(reshaped_narrative)
#         deef_ff = tf.keras.layers.Reshape((512,), input_shape=(1, 512))(embed_narrative)
#
#         deep = tf.keras.layers.Dense(256, activation="relu", )(deef_ff)
#         deep = tf.keras.layers.Dense(64, activation="relu")(deep)
#         deep = tf.keras.layers.Dense(16, activation="relu")(deep)
#
#         wide_ff = tf.keras.layers.concatenate(input_features)
#
#         wide = tf.keras.layers.Dense(16, activation="relu")(wide_ff)
#
#         both = tf.keras.layers.concatenate([deep, wide])
#
#         output = tf.keras.layers.Dense(1, activation="sigmoid")(both)
#
#         keras_model = tf.keras.models.Model(inputs, output)
#         keras_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                             loss="binary_crossentropy",
#                             metrics=[
#                                 tf.keras.metrics.BinaryAccuracy(),
#                                 tf.keras.metrics.TruePositives()
#                             ]
#                             )
#         return keras_model
#     except Exception as e:
#         raise CensusConsumerException(e, sys) from e


def input_fn(file_pattern: List[Text],
             data_accessor: tfx.components.DataAccessor,
             tf_transform_output: tft.TFTransformOutput,
             batch_size: int = 200) -> tf.data.Dataset:
    """Generates features and label for tuning/training.

      Args:
        file_pattern: List of paths or patterns of input tfrecord files.
        data_accessor: DataAccessor for converting input to RecordBatch.
        tf_transform_output: A TFTransformOutput.
        batch_size: representing the number of consecutive elements of returned
          dataset to combine in a single batch

      Returns:
        A dataset that contains (features, indices) tuple where features is a
          dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    try:

        # dataset = tf.data.experimental.make_batched_features_dataset(
        #     file_pattern=file_pattern,
        #     batch_size=batch_size,
        #     features=transformed_feature_spec,
        #     reader=_gzip_reader_fn,
        #     label_key=transformed_name(LABEL_KEY))

        return data_accessor.tf_dataset_factory(
            file_pattern,
            tfxio.TensorFlowDatasetOptions(
                batch_size=batch_size,
                label_key=transformed_name(LABEL_KEY)
            ),
            tf_transform_output.transformed_metadata.schema
        )

    except Exception as e:
        raise CensusConsumerException(e, sys) from e


def run_fn(fn_args: tfx.components.FnArgs):
    try:
        tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

        train_dataset = input_fn(file_pattern=fn_args.train_files,
                                 data_accessor=fn_args.data_accessor,
                                 tf_transform_output=tf_transform_output, )
        eval_dataset = input_fn(file_pattern=fn_args.eval_files,
                                data_accessor=fn_args.data_accessor,
                                tf_transform_output=tf_transform_output, )

        model = _build_keras_model(tf_transform_output=tf_transform_output)
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=[tf.keras.metrics.BinaryAccuracy()]
                      )
        model.fit(
            train_dataset,
            steps_per_epoch=fn_args.train_steps,
            validation_data=eval_dataset,
            validation_steps=fn_args.eval_steps
        )
        export_serving_model(tf_transform_output=tf_transform_output,
                             model=model,
                             output_dir=fn_args.serving_model_dir)
    except Exception as e:
        raise CensusConsumerException(e, sys) from e


def _get_tf_examples_serving_signature(model, tf_transform_output):
    try:
        model.tft_layer_inference = tf_transform_output.transform_features_layer()

        @tf.function(
            input_signature=[tf.TensorSpec(shape=None, dtype=tf.string, name='examples')]
        )
        def serve_tf_example_fn(serialized_tf_example):
            raw_feature_spec = tf_transform_output.raw_feature_spec()
            print(raw_feature_spec)
            raw_feature_spec.pop(LABEL_KEY)
            raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
            transformed_features = model.tft_layer_inference(raw_features)
            outputs = model(transformed_features)
            return {'outputs': outputs}

        return serve_tf_example_fn

    except Exception as e:
        raise CensusConsumerException(e, sys) from e


def _get_transform_features_signature(model, tf_transform_output):
    try:
        model.tft_layer_eval = tf_transform_output.transform_features_layer()

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
        ])
        def transform_features_fn(serialized_tf_examples):
            raw_feature_spec = tf_transform_output.raw_feature_spec()
            raw_features = tf.io.parse_example(serialized_tf_examples, raw_feature_spec)
            transformed_features = model.tft_layer_eval(raw_features)
            return transformed_features

        return transform_features_fn

    except Exception as e:
        raise CensusConsumerException(e, sys) from e


def export_serving_model(tf_transform_output, model, output_dir):
    try:
        model.tft_layer = tf_transform_output.transform_features_layer()

        signatures = {
            "serving_default":
                _get_tf_examples_serving_signature(model=model,
                                                   tf_transform_output=tf_transform_output),
            "transform_features": _get_transform_features_signature(model, tf_transform_output),

        }

        model.save(output_dir, save_format="tf", signatures=signatures)
    except Exception as e:
        raise CensusConsumerException(e, sys) from e
