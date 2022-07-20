import sys
import os
from census_consumer_complaint.exception.exception import CensusConsumerException
from census_consumer_complaint.config.configuration import CensusConsumerConfiguration
from tfx.components import SchemaGen, StatisticsGen, Transform
from census_consumer_complaint.component.data_ingestion import DataIngestion
from census_consumer_complaint.component.data_validation import DataValidation

from collections import namedtuple

DataPreprocessing = namedtuple("DataPreprocessing", ["transformer"])

census_consumer_config = CensusConsumerConfiguration()


def get_data_preprocessing_components(data_ingestion: DataIngestion,
                                      data_validation: DataValidation
                                      ) -> DataPreprocessing:
    try:
        data_preprocessing_components = []
        transform = Transform(
            examples=data_ingestion.zip_example_gen.outputs['examples'],
            schema=data_validation.schema_gen.outputs['schema'],
            module_file=census_consumer_config.transform_module_file_path
        )
        return DataPreprocessing(transformer=transform)

    except Exception as e:
        raise CensusConsumerException(e, sys) from e
