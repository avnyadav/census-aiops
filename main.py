from census_consumer_complaint_orchestrator import apache_beam_orchestrator
import tensorflow as tf
import sys
from census_consumer_complaint_exception.exception import CensusConsumerException
if __name__ == "__main__":
    try:
    #apache_beam_orchestrator.run_apache_dag_pipeline()

        path  =r"/home/avnish/census_consumer_project/census_consumer_complaint/census_consumer_complaint_data/census_consumer_complaint/artifact/Trainer/model/53/Format-Serving"
        model = tf.saved_model.load(export_dir=path)

        transform_fn = model.signatures["transform_features"]
        prediction_fn = model.signatures["serving_default"]

        # print(transform_fn)
        # print(prediction_fn)

        file_name="/home/avnish/census_consumer_project/census_consumer_complaint/census_consumer_complaint_data/census_consumer_complaint/artifact/RemoteZipCsvExampleGen/examples/65/Split-eval/data_tfrecord-00000-of-00001.gz"
        file_names = [file_name]

        raw_data = tf.data.TFRecordDataset(file_names)
        for da in raw_data.take(10):
            print(da)
        # predicted_data = prediction_fn(transform_fn(raw_data))
        # print(predicted_data)
    except Exception as e:
        print(e)