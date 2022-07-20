from tfx.components import Trainer
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from census_consumer_complaint.exception.exception import CensusConsumerException
from census_consumer_complaint.component.data_ingestion import DataIngestion
from census_consumer_complaint.component.data_validation import DataValidation
from census_consumer_complaint.component.data_preprocessing import DataPreprocessing
from census_consumer_complaint.config.configuration import CensusConsumerConfiguration
from tfx.proto import trainer_pb2
from collections import namedtuple
import sys

TRAINING_STEPS = 1000
EVALUATION_STEPS = 100

ModelTrainer = namedtuple("ModelTrainer", ["trainer"])
census_consumer_config = CensusConsumerConfiguration()


def get_model_trainer_component(data_validation: DataValidation,
                                data_preprocessing: DataPreprocessing):
    try:

        training_kwargs = {
            "module_file": census_consumer_config.trainer_module_file_path,
            "examples": data_preprocessing.transformer.outputs['transformed_examples'],
            "schema": data_validation.schema_gen.outputs['schema'],
            "transform_graph": data_preprocessing.transformer.outputs['transform_graph'],
            "train_args": trainer_pb2.TrainArgs(num_steps=TRAINING_STEPS),
            "eval_args": trainer_pb2.EvalArgs(num_steps=EVALUATION_STEPS),
        }
        trainer = Trainer(**training_kwargs)
        model_trainer = ModelTrainer(trainer=trainer,
                                     )
        return model_trainer
    except Exception as e:
        raise CensusConsumerException(e, sys) from e
