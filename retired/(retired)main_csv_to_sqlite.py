"""
Script to transfer csv feature set into sqlite db for data streaming
Be sure to customize root_path to match actual csv file directory
"""
import numpy as np
import pandas as pd
import time
from sqlalchemy import create_engine

print(">>> define file paths")
root_path = '../data'
x_train_file = '/'.join([root_path, 'scaled-network-data/x_train.csv'])
x_test_file = '/'.join([root_path, 'scaled-network-data/x_test.csv'])
y_train_file = '/'.join([root_path, 'scaled-network-data/y_train.csv'])
y_test_file = '/'.join([root_path, 'scaled-network-data/y_test.csv'])

print(">>> glimpse files")
print(pd.read_csv(x_train_file, nrows=2))
print(pd.read_csv(x_test_file, nrows=2))
print(pd.read_csv(y_train_file, nrows=2))
print(pd.read_csv(y_test_file, nrows=2))


print(">>> link to sqlite database mfcc_phonemes.db")
mpdb = create_engine('sqlite:///../data/mfcc_phonemes.db')


def load_csv_to_sqlite(file_path, chunksize, table_name, db_conn):
    i = 0
    j = 1
    train_start = time.time()
    print(">>> itering through csv file")
    for df in pd.read_csv(file_path, chunksize=chunksize, iterator=True):
        df = df.rename(columns={c: c.replace(" ", '_') for c in df.columns})
        df.index += j
        i += 1
        df.to_sql(table_name, db_conn, if_exists='append', index=False)
        j = df.index[-1] + 1
        print(">>> iter %i: successful" % i)
    train_end = time.time()
    print(">>> loading csv took %i seconds" % (train_end - train_start))

print(">>> loading x_test.csv into sqlite db")
load_csv_to_sqlite(x_test_file, 10000, 'x_test', mpdb)

print(">>> loading x_train.csv into sqlite db")
load_csv_to_sqlite(x_train_file, 10000, 'x_train', mpdb)

print(">>> loading y_test.csv into sqlite db")
load_csv_to_sqlite(y_test_file, 10000, 'y_test', mpdb)

print(">>> loading y_train.csv into sqlite db")
load_csv_to_sqlite(y_train_file, 10000, 'y_train', mpdb)
