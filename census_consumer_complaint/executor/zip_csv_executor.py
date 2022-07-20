# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Avro based TFX example gen executor."""
import datetime
import os
from typing import Any, Dict

from absl import logging
import apache_beam as beam
import tensorflow as tf

from tfx.components.example_gen import utils
from tfx.components.example_gen.base_example_gen_executor import BaseExampleGenExecutor
from tfx.types import standard_component_specs
from zipfile import ZipFile
import csv
from collections import OrderedDict
from census_consumer_complaint.utils.utils import download_dataset

FILE_READ_FORMAT = "r"

COLUMNS = ['Date received', 'Product', 'Sub-product', 'Issue', 'Sub-issue', 'Consumer complaint narrative',
           'Company public response', 'Company', 'State', 'ZIP code', 'Tags', 'Consumer consent provided?',
           'Submitted via', 'Date sent to company', 'Company response to consumer', 'Timely response?',
           'Consumer disputed?', 'Complaint ID']

# COLUMNS = "pickup_community_area,fare,trip_start_month,trip_start_hour,trip_start_day,trip_start_timestamp,pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude,trip_miles,pickup_census_tract,dropoff_census_tract,payment_type,company,trip_seconds,dropoff_community_area,tips".split(
#     ",")
REMOTE_ZIP_FILE_URI = 'zip_file_uri'


def dict_to_example(element):
    print(element)
    utils.dict_to_example(element)


COLUMNS = None


def parse_file(element):
    global COLUMNS
    for line in csv.reader([element], quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
        if COLUMNS is None:
            COLUMNS = line
            line = ["" for data in line]
        return OrderedDict(zip(COLUMNS, line))


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _ZipToExample(  # pylint: disable=invalid-name
        pipeline: beam.Pipeline, exec_properties: Dict[str, Any],
        split_pattern: str) -> beam.pvalue.PCollection:
    """Read Avro files and transform to TF examples.

  Note that each input split will be transformed by this function separately.

  Args:
    pipeline: beam pipeline.
    exec_properties: A dict of execution properties.
      - input_base: input dir that contains Avro data.
    split_pattern: Split.pattern in Input config, glob relative file pattern
      that maps to input files with root directory given by input_base.

  Returns:
    PCollection of TF examples.
  """
    # directory to extract zip file
    input_base_uri = exec_properties[standard_component_specs.INPUT_BASE_KEY]

    # remote zip file uri to download zip file
    zip_file_uri = exec_properties[REMOTE_ZIP_FILE_URI]

    # downloading zip file from zip file uri into input_base_uri location
    zip_file_path = download_dataset(zip_file_uri, input_base_uri)

    # extracting zip file and deleteing zip file from directory
    with ZipFile(zip_file_path, FILE_READ_FORMAT) as zip_file:
        zip_file.extractall(path=os.path.join(input_base_uri))
    os.remove(zip_file_path)

    # obtain csv file path
    csv_file_path = os.path.join(input_base_uri, os.listdir(input_base_uri)[0])
    # comment the cdode
    # import pandas as pd
    # df = pd.read_csv(csv_file_path)
    # os.remove(csv_file_path)
    # df.iloc[:5000,:].to_csv(csv_file_path, index=None, header=True, mode="w")

    # uncomment the code
    return (pipeline
            | 'ReadCsvFile' >> beam.io.ReadFromText(csv_file_path)
            | 'ParseFile' >> beam.Map(parse_file)
            | "ToTFExample" >> beam.Map(utils.dict_to_example)
            )

    # utils.dict_to_example()
    # return (pipeline
    #         | 'ReadFromAvro' >> beam.io.ReadFromAvro(avro_pattern)
    #         | 'ToTFExample' >> beam.Map(utils.dict_to_example))


class Executor(BaseExampleGenExecutor):
    """TFX example gen executor for processing avro format.

  Data type conversion:
    integer types will be converted to tf.train.Feature with tf.train.Int64List.
    float types will be converted to tf.train.Feature with tf.train.FloatList.
    string types will be converted to tf.train.Feature with tf.train.BytesList
      and utf-8 encoding.

    Note that,
      Single value will be converted to a list of that single value.
      Missing value will be converted to empty tf.train.Feature().

    For details, check the dict_to_example function in example_gen.utils.


  Example usage:

    from tfx.components.base import executor_spec
    from tfx.components.example_gen.component import
    FileBasedExampleGen
    from tfx.components.example_gen.custom_executors import
    avro_executor

    example_gen = FileBasedExampleGen(
        input_base=avro_dir_path,
        custom_executor_spec=executor_spec.ExecutorClassSpec(
            avro_executor.Executor))
  """

    def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
        """Returns PTransform for avro to TF examples."""
        return _ZipToExample
