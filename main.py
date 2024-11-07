from utils import *
from strategy import *
from .searching import *

import optuna
import os
from typing import List
import warnings
from concurrent.futures import ThreadPoolExecutor
import logging

warnings.filterwarnings('ignore')

while True:
    option = int(input("""
          1. Searching
          2. Optimize
          3. Testing
          """))
    
    if option == 1:
        searching()
    elif option == 2:
        optimize()
    elif option == 3:
        testing()