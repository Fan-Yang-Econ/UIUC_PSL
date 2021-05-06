import os

import pandas as pd
import pkg_resources

file_path = pkg_resources.resource_filename('MovieAppBackendServer.data', 'df_nn.csv')
df_nn = pd.read_csv(file_path)

file_path = pkg_resources.resource_filename('MovieAppBackendServer.data', 'df_rating_avg.csv')
df_rating_avg = pd.read_csv(file_path)
