from textSummarization.pipeline.Stage01_data_ingestion import DataIngestionTrainingPipeline
from textSummarization.pipeline.Stage02_data_validation import DataValidationTrainingPipeline
from textSummarization.pipeline.Stage_03_data_transformation import DataTransformationTrainingPipeline
from textSummarization.pipeline.Stage_04_model_trainer import ModelTrainerTrainingPipeline
from textSummarization.pipeline.Stage_05_model_evaluation import ModelEvaluationPipeline
from textSummarization.logging import logger


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Validation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_validation  = DataValidationTrainingPipeline()
   data_validation .main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Transformation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_transformation  = DataTransformationTrainingPipeline()
   data_transformation .main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Model Trainer stage"
try:
   logger.info("***********************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_trainer  =ModelTrainerTrainingPipeline()
   model_trainer .main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Model Evaluation stage"
try:
   logger.info("***********************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_evaluation  =ModelEvaluationPipeline()
   model_evaluation .main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

# By- Jai Dixit