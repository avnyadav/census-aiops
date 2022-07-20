import logging
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.orchestration.metadata import sqlite_metadata_connection_config
from census_consumer_complaint.config.configuration import CensusConsumerConfiguration
from census_consumer_complaint.component.component import get_census_consumer_complaint_pipeline_component

census_consumer_config = CensusConsumerConfiguration()


PIPELINE_ENABLE_CACHE_STATUS = True


# Airflow-specific configs; these will be passed directly to airflow
def _create_pipeline():
    return pipeline.Pipeline(
        pipeline_name=census_consumer_config.pipeline_name,
        pipeline_root=census_consumer_config.pipeline_root,
        components=get_census_consumer_complaint_pipeline_component(),
        enable_cache=PIPELINE_ENABLE_CACHE_STATUS,
        metadata_connection_config=sqlite_metadata_connection_config(census_consumer_config.metadata_path),

    )


def run_apache_dag_pipeline():
    beam_pipeline = BeamDagRunner().run(_create_pipeline())

