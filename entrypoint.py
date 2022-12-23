#code
from model import predict
from conf import settings
from conf import logging


#user is asked to choose which model he wants to use for predictions
m_num = int(input())
prediction = get_predictions(settings.PREDICTION.VALUES, m_num)
logging.info(f"prediction:{prediction}")
