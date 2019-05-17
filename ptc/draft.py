from tqdm import tqdm
from time import sleep
import tqdm

with tqdm.trange(10) as t:
    for i in t:
        sleep(0.2)
        t.set_description('GENdasda %i' % i)
