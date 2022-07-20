import csv
import datetime
import sys, os
import tensorflow as tf
import shutil
from census_consumer_complaint.exception.exception import CensusConsumerException
from urllib import request
import random
import numpy as np
from collections import namedtuple
from collections import OrderedDict
from zipfile import ZipFile

CSV_FILE_EXTENSION = ".csv"
TF_RECORD_FILE_EXTENSION = ".tfrecord"
ROW = 0


# defining function to convert value to appropriate data type which tf.Example accepts
def _bytes_feature(value):
    """
    created by: Deepranjan
    created on: 13/07/2022
    version: 1.0
    """
    try:
        value = bytes(value, encoding="utf-8")
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    except Exception as e:
        raise Exception(CensusConsumerException(e, sys)) from e


def _float_feature(value):
    try:
        """
        created by: Deepranjan
        created on: 13/07/2022
        version: 1.0
        """
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    except Exception as e:
        raise Exception(CensusConsumerException(e, sys)) from e


def _int64_feature(value):
    """
    created by: Avnish Yadav
    created on: 27/02/2022
    version: 1.0
    """
    try:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    except Exception as e:
        raise Exception(CensusConsumerException(e, sys)) from e


def _convert_csv_file_to_tf_record_file(csv_file_path, tf_record_file_dir: str):
    try:
        n_row = 0
        tf_record_file_name = os.path.basename(csv_file_path).replace(CSV_FILE_EXTENSION, TF_RECORD_FILE_EXTENSION)
        tf_record_file_path=os.path.join(tf_record_file_dir,tf_record_file_name)
        tf_record_file_writer = tf.io.TFRecordWriter(tf_record_file_path)
        with open(csv_file_path) as csv_file:
            reader = csv.DictReader(csv_file, delimiter=",", quotechar='"')
            for row in reader:
                n_row += 1
                if n_row%100000==0:
                    print(n_row,"\n",datetime.datetime.now())
                    break
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        "product": _bytes_feature(row["Product"]),
                        "sub_product": _bytes_feature(row["Sub-product"]),
                        "issue": _bytes_feature(row["Issue"]),
                        "sub_issue": _bytes_feature(row["Sub-issue"]),
                        "state": _bytes_feature(row["State"]),
                        "zip_code": _bytes_feature((row["ZIP code"])),
                        "company": _bytes_feature(row["Company"]),
                        "company_response": _bytes_feature(row["Company response to consumer"]),
                        "consumer_complaint_narrative": _bytes_feature(row["Consumer complaint narrative"]),
                        "timely_response": _bytes_feature(row["Timely response?"]),
                        "consumer_disputed": _bytes_feature(row["Consumer disputed?"]),
                    }
                )
                )
            
                tf_record_file_writer.write(example.SerializeToString())  
        tf_record_file_writer.close()
    except Exception as e:
        raise Exception(CensusConsumerException(e, sys)) from e


def transform_csv_to_tf_record_file(csv_file_dir, tf_record_file_dir: str):
    """
    Description: This function accept csv file directory and converts all csv file into
    tfrecord file at tf_record_file_dir
    =============================================================================
    :param csv_file_dir: Dir path containing csv files
    :param tf_record_file_dir: Dir path to generated tf record files from csv files
    :return: dict with file_path and number of row
    {file_path:n_row}

    """
    try:

        csv_files = filter(lambda x: x.endswith(CSV_FILE_EXTENSION),
                           os.listdir(csv_file_dir))
        os.makedirs(tf_record_file_dir,exist_ok=True)
        print(csv_files,tf_record_file_dir)
        for file in csv_files:
            csv_file_path = os.path.join(csv_file_dir, file)
            _convert_csv_file_to_tf_record_file(csv_file_path, tf_record_file_dir)
            os.remove(csv_file_path)
    except Exception as e:
        raise (CensusConsumerException(e, sys)) from e


def extract_zip_file(zip_file_path: str, extract_dir: str, zip_file_read_mode: str = "r") -> None:
    """Extracts a zip file to a directory.

  Args:
  zip_file_path: The path to the zip file.
  output_dir: The directory to extract the zip file to.
  """
    try:
        print("Extracting file in ",extract_dir)
        os.makedirs(extract_dir, exist_ok=True)
        with ZipFile(zip_file_path, zip_file_read_mode) as zip_file:
            zip_file.extractall(extract_dir)
    except Exception as e:
        raise e


def parse_file(element, columns):
    global ROW
    ROW += 1

    """Read a line of CSV file and transform it into ordered dict"""
    for line in csv.reader([element], quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):

        if len(line) != len(columns):
            line = ["" for i in range(len(columns))]
        if ROW % 100000 == 0:
            print(ROW, datetime.datetime.now())
            print(line)
        return OrderedDict(zip(columns, line))


## code added by Avnish327030
def download_dataset(zip_file_uri: str, download_dir: str) -> str:
    """Downloads a dataset from a given uri and saves it to a Download directory.

  Args:
    zip_file_uri: The uri of the dataset to download.
    download_dir: The directory to save the dataset to.

  Returns:
    The path to the downloaded dataset.

  Raises:
    ValueError: If the dataset cannot be downloaded.
  """

    try:
        print("Downloading file in ",download_dir)
        # Creating download_dir if not exists
        if os.path.exists(download_dir):
            shutil.rmtree(download_dir)
        os.makedirs(download_dir, exist_ok=True)
        # Obtaining zip file path to download zip file
        zip_file_path = os.path.join(download_dir, os.path.basename(zip_file_uri))
        request.urlretrieve(zip_file_uri, zip_file_path)
        return zip_file_path
    except Exception as e:
        raise ValueError('Failed to download dataset from {}. {}'.format(zip_file_uri, e))
