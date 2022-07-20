from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from tfx.orchestration import pipeline
from tfx.orchestration.airflow.airflow_runner import AirflowDAGRunner
from tfx.orchestration.metadata import sqlite_metadata_connection_config
from census_consumer_complaint.config.configuration import CensusConsumerConfiguration

from census_consumer_complaint.component.component import get_census_consumer_complaint_pipeline_component

AIRFLOW_CONFIG_SCHEDULE_INTERVAL_KEY = "schedule_interval"
AIRFLOW_CONFIG_START_DATE_KEY = "start_date"
LOG_OVERRIDES_LOG_ROOT_KEY = "log_root"
LOG_OVERRIDES_LOG_LEVEL_KEY = "log_level"
PIPELINE_ADDITIONAL_PIPELINE_ARGS_KEY = "logger_args"
PIPELINE_ENABLE_CACHE_STATUS = False


census_consumer_config = CensusConsumerConfiguration()

# Airflow-specific configs; these will be passed directly to airflow

_airflow_config = {
    AIRFLOW_CONFIG_SCHEDULE_INTERVAL_KEY: census_consumer_config.scheduled_interval,
    AIRFLOW_CONFIG_START_DATE_KEY: census_consumer_config.start_date,

}

# Logging overrides
logger_overrides = {LOG_OVERRIDES_LOG_ROOT_KEY: census_consumer_config.log_dir,
                    LOG_OVERRIDES_LOG_LEVEL_KEY: logging.INFO}


def _create_pipeline():
    return pipeline.Pipeline(
        pipeline_name=census_consumer_config.pipeline_name,
        pipeline_root=census_consumer_config.pipeline_root,
        components=get_census_consumer_complaint_pipeline_component(),
        enable_cache=PIPELINE_ENABLE_CACHE_STATUS,
        metadata_connection_config=sqlite_metadata_connection_config(census_consumer_config.metadata_path),
        additional_pipeline_args={PIPELINE_ADDITIONAL_PIPELINE_ARGS_KEY: logger_overrides},
    )


def get_airflow_dag_pipeline():
    airflow_pipeline = AirflowDAGRunner(_airflow_config).run(_create_pipeline())
    return airflow_pipeline
