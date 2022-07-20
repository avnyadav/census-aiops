from census_consumer_complaint.exception.exception import CensusConsumerException
from census_consumer_complaint.component.data_ingestion import DataIngestion
from census_consumer_complaint.component.model_trainer import ModelTrainer
import sys, os

from tfx.components import Evaluator
from tfx.v1.dsl import Resolver, Channel
from tfx.v1.types.standard_artifacts import Model, ModelBlessing

from tfx.v1.dsl import experimental

import tensorflow_model_analysis as tfma
from census_consumer_complaint.component.feature_engineering.feature_engineering import LABEL_KEY
from collections import namedtuple

ModelEvaluation = namedtuple("ModelEvaluation", ["evaluator", "resolver"])


def get_model_evaluation_component(data_ingestion: DataIngestion, trainer: ModelTrainer):
    try:
        eval_config = tfma.EvalConfig(
            model_specs=[
                tfma.ModelSpec(
                    signature_name="serving_default",
                    label_key=LABEL_KEY,

                )
            ],
            slicing_specs=[tfma.SlicingSpec(), tfma.SlicingSpec(feature_keys=["product"])],
            metrics_specs=[
                tfma.MetricsSpec(
                    metrics=[
                        tfma.MetricConfig(
                            class_name="BinaryAccuracy",
                            threshold=tfma.MetricThreshold(
                                value_threshold=tfma.GenericValueThreshold(
                                    lower_bound={"value": 0.5}
                                ),
                                change_threshold=tfma.GenericChangeThreshold(
                                    direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                    absolute={"value": -1e-10},
                                ),
                            ),
                        ),
                        tfma.MetricConfig(class_name="Precision"),
                        tfma.MetricConfig(class_name="Recall"),
                        tfma.MetricConfig(class_name="ExampleCount"),
                        tfma.MetricConfig(class_name="AUC"),
                    ],
                )
            ],
        )
        model_resolver = Resolver(
            strategy_class=experimental.LatestBlessedModelStrategy,
            model=Channel(type=Model),
            model_blessing=Channel(type=ModelBlessing),
        )
        evaluator = Evaluator(
            examples=data_ingestion.zip_example_gen.outputs['examples'],
            model=trainer.trainer.outputs['model'],
            baseline_model=model_resolver.outputs["model"],
            eval_config=eval_config,
        )
        evaluator = Evaluator(
            examples=data_ingestion.zip_example_gen.outputs['examples'],
            model=trainer.trainer.outputs['model'],
            eval_config=eval_config,
            baseline_model=model_resolver.outputs['model'],

        )
        return ModelEvaluation(resolver=model_resolver, evaluator=evaluator)

    except Exception as e:
        raise CensusConsumerException(e, sys) from e
