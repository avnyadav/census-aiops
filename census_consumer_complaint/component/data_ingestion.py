import sys
import os
from tfx.components.base import executor_spec
from exception.exception import CensusConsumerException
from collections import namedtuple

ZIP_INPUT_DATASET_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
ZIP_CSV_EXTRACTOR_GEN_NAME = "ZIP_CSV_EXTRACTOR_GEN"
DataIngestion = namedtuple("DataIngestion", ["zip_example_gen"])
INPUT_BASE = os.path.join(os.getcwd(),"zip_to_csv")

from census_consumer_complaint.custom_component.example_gen import RemoteZipCsvExampleGen


def get_data_ingestion_components(url: str = ZIP_INPUT_DATASET_URL, input_base: str = INPUT_BASE) -> DataIngestion:
    """
    :param input_base:
    :param url:
    :param self:
    :return: List of tfx component
    """
    try:

        input_config = {

        }
        output_config = {

        }
        zip_example_gen = RemoteZipCsvExampleGen(
            zip_file_uri=url,
            input_base=input_base,
        )

        return DataIngestion(zip_example_gen=zip_example_gen)

    except Exception as e:
        raise CensusConsumerException(e, sys) from e
