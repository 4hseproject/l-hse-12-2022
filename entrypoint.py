#code
from model.RF import get_predictions
from conf import settings, logging


#user is asked to choose which model he wants to use for predictions
m_num = int(input())
prediction = get_predictions(settings.PREDICTION.VALUES, m_num)
logging.info(f"prediction:{prediction}")
