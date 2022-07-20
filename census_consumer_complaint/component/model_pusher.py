from census_consumer_complaint.component.model_trainer import ModelTrainer
from census_consumer_complaint.component.model_evaluation import ModelEvaluation
from census_consumer_complaint.exception.exception import CensusConsumerException
from census_consumer_complaint.config.configuration import CensusConsumerConfiguration
import sys
from tfx.components import Pusher
from collections import namedtuple
from tfx.proto import pusher_pb2

ModelPusher = namedtuple("ModelPusher", ["pusher"])
config = CensusConsumerConfiguration()


def get_model_pusher_component(trainer: ModelTrainer, evaluator: ModelEvaluation) -> ModelPusher:
    try:
        pusher = Pusher(
            model=trainer.trainer.outputs['model'],
            model_blessing=evaluator.evaluator.outputs['blessing'],
            push_destination=pusher_pb2.PushDestination(
                filesystem=pusher_pb2.PushDestination.Filesystem(
                    base_directory=config.serving_model_dir
                )
            ),
        )
        print(f"save model dir:{config.serving_model_dir}")

        model_pusher = ModelPusher(pusher=pusher)
        return model_pusher

    except Exception as e:
        raise CensusConsumerException(e, sys) from e
