import sys
from census_consumer_complaint.exception.exception import CensusConsumerException
import tensorflow as tf
import tensorflow_transform as tft

coulmns = ['Company', 'Company public response', 'Company response to consumer',
           'Complaint ID', 'Consumer complaint narrative', 'Consumer consent provided?',
           'Consumer disputed?', 'Date received', 'Date sent to company', 'Issue',
           'Product', 'State', 'Sub-issue',
           'Sub-product', 'Submitted via', 'Tags', 'Timely response?', 'ZIP code']
ONE_HOT_FEATURES = {
    "Product": 11,
    "Sub-product": 45,
    "Company response to consumer": 5,
    "State": 60,
    "Issue": 90
}

LABEL_KEY = "Consumer disputed?"

BUCKET_FEATURES = {
    "ZIP code": 10
}

TEXT_FEATURES = {
    "Consumer complaint narrative": None

}


def transformed_name(key):
    try:
        return key + "_xf"
    except Exception as e:
        raise CensusConsumerException(e, sys) from e


def fill_in_missing(x):
    try:
        is_updated = False
        default_value = "" if x.dtype == tf.string else 0
        print(type(x))
        if type(x) == tf.SparseTensor:
            is_updated = True
            x = tf.sparse.to_dense(
                tf.SparseTensor(
                    x.indices, x.values, [x.dense_shape[0], 1]),
                default_value
            )
        print(f"Column: {x} has been updated: {is_updated}")

        return tf.squeeze(x, axis=1)
    except Exception as e:
        raise CensusConsumerException(e, sys) from e


def convert_num_to_one_hot(label_tensor, num_labels=2):
    try:
        one_hot_tensor = tf.one_hot(label_tensor, num_labels)
        return tf.reshape(one_hot_tensor, [-1, num_labels])
    except Exception as e:
        raise CensusConsumerException(e, sys) from e


@tf.function
def cast_target_column_to_integer(x):
    x = tf.strings.lower(fill_in_missing(x))
    return tf.cast(tf.equal(x, 'no'), tf.int64)


@tf.function
def convert_zip_code(zip_code):
    try:
        if zip_code == "":
            zip_code = "00000"
        zip_code = tf.strings.regex_replace(zip_code, r'X{0,5}', "0")
        zip_code = tf.strings.to_number(zip_code, tf.float32)
        return zip_code
    except Exception as e:
        print(e)
        raise CensusConsumerException(e, sys) from e


def preprocessing_fn(inputs):
    try:
        print(f"Columns: {inputs.keys()}")
        outputs = {}
        for key in ONE_HOT_FEATURES.keys():
            dim = ONE_HOT_FEATURES[key]
            index = tft.compute_and_apply_vocabulary(
                fill_in_missing(inputs[key]), top_k=dim + 1)
            outputs[transformed_name(key)] = convert_num_to_one_hot(index, num_labels=dim + 1)

        # for key, bucket_count in BUCKET_FEATURES.items():
        #     temp_feature = tft.bucketize(
        #         tft.bucketize(convert_zip_code(fill_in_missing(inputs[key])),
        #                       num_buckets=5,
        #                       )  # , num_buckets=10
        #     )

        # outputs[transformed_name(key)] = convert_num_to_one_hot(temp_feature, num_labels=bucket_count + 1)

        for key in TEXT_FEATURES.keys():
            outputs[transformed_name(key)] = fill_in_missing(inputs[key])
        # converting label categorical column into integer
        outputs[transformed_name(LABEL_KEY)] = inputs[LABEL_KEY]
        return outputs
    except Exception as e:
        raise CensusConsumerException(e, sys) from e