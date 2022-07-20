from census_consumer_complaint.config.configuration import CensusConsumerConfiguration
from census_consumer_complaint.exception.exception import CensusConsumerException
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.orchestration.metadata import sqlite_metadata_connection_config
import sys,os


class CensusConsumerInteractiveContext(CensusConsumerConfiguration):

    def __init__(self, *args, **kwargs):
        try:
            super(CensusConsumerInteractiveContext, self).__init__()
            self.interactive_context = None
        except Exception as e:
            raise (CensusConsumerException(e, sys)) from e

    def get_interactive_context(self):
        try:

            if self.interactive_context is None:
                self.interactive_context = InteractiveContext(
                    pipeline_name=self.pipeline_name,
                    pipeline_root=self.pipeline_root,
                    metadata_connection_config=sqlite_metadata_connection_config(self.metadata_path)
                )
            return self.interactive_context
        except Exception as e:
            raise (CensusConsumerException(e, sys)) from e
