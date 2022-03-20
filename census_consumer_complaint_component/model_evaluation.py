from census_consumer_complaint_exception.exception import CensusConsumerException
from census_consumer_complaint_component.data_ingestion import DataIngestion
from census_consumer_complaint_component.model_trainer import ModelTrainer
import sys, os
from tfx.components import Evaluator
import tensorflow_model_analysis as tfma
from census_consumer_complaint_component.feature_engineering.feature_engineering import LABEL_KEY
from collections import namedtuple

ModelEvaluation = namedtuple("ModelEvaluation", "evaluator")


def get_model_evaluation_component(data_ingestion: DataIngestion, trainer: ModelTrainer):
    try:
        eval_config = tfma.EvalConfig(

            model_specs=[tfma.ModelSpec(label_key=LABEL_KEY)],
            slicing_specs=[tfma.SlicingSpec(),
                           tfma.SlicingSpec(feature_keys=['Product'])],
            metrics_specs=[tfma.MetricsSpec(metrics=[tfma.MetricConfig(class_name='BinaryAccuracy'),
                                                     tfma.MetricConfig(class_name='ExampleCount'),
                                                     tfma.MetricConfig(class_name="AUC")
                                                     ]
                                            )])
        evaluator = Evaluator(
            examples=data_ingestion.zip_example_gen.outputs['examples'],
            model=trainer.trainer.outputs['model'],
            eval_config=eval_config

        )
        return ModelEvaluation(evaluator=evaluator)

    except Exception as e:
        raise CensusConsumerException(e, sys) from e
