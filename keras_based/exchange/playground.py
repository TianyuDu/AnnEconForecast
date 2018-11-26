quit()
python3.6
import sys
sys.path.append("./core/containers/")
sys.path.append("./core/models/")
import config
import methods
from methods import *
from models import *
from multi_config import *

from multivariate_container import MultivariateContainer
from multivariate_lstm import MultivariateLSTM

c = MultivariateContainer(
    file_dir,
    target,
    load_multi_ex,
    CON_config)

model = MultivariateLSTM(c, config=NN_config)