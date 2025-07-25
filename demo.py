# below code is to check the logging config
# from src.logger import logging

# logging.debug("This is a debug message.")
# logging.info("This is an info message.")
# logging.warning("This is a warning message.")
# logging.error("This is an error message.")
# logging.critical("This is a critical message.")

# --------------------------------------------------------------------------------

# # below code is to check the exception config
# from src.logger import logging
# from src.exception import MyException
# import sys

# try:
#     a = 1+'Z'
# except Exception as e:
#     logging.info(e)
#     raise MyException(e, sys) from e

# --------------------------------------------------------------------------------

# from src.pipline.training_pipeline import TrainPipeline

# pipline = TrainPipeline()
# pipline.run_pipeline()

# from src.pipline.popular_companies_trainer_pipeline import TrainPipeline_Popular

# pipeline = TrainPipeline_Popular()
# pipeline.run_pipeline()

from src.pipline.prediction_pipeline import PredictionPipeline

portfolio_value = 500
stock_options = ["AAPL", "MSFT", "GOOGL"]
weights = [0,4,0.1,0.5]
pipeline = PredictionPipeline(portfolio_value,stock_options,weights)
forecast_df = pipeline.run_pipeline()
print(forecast_df)
forecast_df.to_csv('forecast.csv')