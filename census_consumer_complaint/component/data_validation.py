import sys
import os
from census_consumer_complaint.exception.exception import CensusConsumerException

from tfx.components import SchemaGen, StatisticsGen, ExampleValidator

from census_consumer_complaint.custom_component.component import RemoteZipFileBasedExampleGen
from census_consumer_complaint.component.data_ingestion import DataIngestion
from collections import namedtuple

DataValidation = namedtuple("DataValidation", ["statistic_gen", "schema_gen", "example_val"])


def get_data_validation_components(data_ingestion: DataIngestion) -> DataValidation:
    """
    :param zip_example_gen:
    :param self:
    :return: List of tfx component
    """
    try:

        data_validation_components = []
        statistic_gen = StatisticsGen(
            examples=data_ingestion.zip_example_gen.outputs['examples']
        )

        schema_gen = SchemaGen(
            statistics=statistic_gen.outputs['statistics']
        )

        example_val = ExampleValidator(schema=schema_gen.outputs['schema'],
                                       statistics=statistic_gen.outputs['statistics'])

        return DataValidation(statistic_gen=statistic_gen,
                              schema_gen=schema_gen,
                              example_val=example_val
                              )

    except Exception as e:
        raise CensusConsumerException(e, sys) from e
