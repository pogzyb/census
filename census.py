# UCI census dataset construction
import pandas as pd
import numpy as np
import re
import os
import sys

class UCIdataset(object):
    """
    Class to read data into Dataframes from files
    """
    def __init__(self, basepath):
        self.basepath = basepath

    def readinColumns(self, filename):
        with open(self.basepath+filename) as data:
            data = data.read()
            cols = re.findall(r'\(.+\)', data)
            cols = [x.strip('(').strip(')') for x in cols]
            cols.append('target')
        return cols

    def readinData(self, filename, columns):
        with open(self.basepath+filename) as data:
            data = data.readlines()
            dataset = []
            for row in data:
                dataset.append(row.lstrip(' ').split(','))
        return pd.DataFrame(dataset, columns=columns)


def main():

    reader = UCIdataset('census/')
    columns = reader.readinColumns('columns.txt')
    # grab training
    training = reader.readinData('census-income.data', columns)
    testing = reader.readinData('census-income.test', columns)
    # create csv's
    print("Writing Training to CSV. . .")
    training.to_csv('training.csv', index=False)
    print("Writing Testing to CSV. . .")
    testing.to_csv('testing.csv', index=False)

    print("Sample .head() from training.csv:\n {}\n".format(training.head()))

    print("Done.")


    return

# - - - - - -
if __name__ == "__main__":
    main()

    # columns = ['AAGE',
    #     'ACLSWKR',
    #     'ADTIND',
    #     'ADTOCC',
    #     'AGI',
    #     'AHGA',
    #     'AHRSPAY',
    #     'AHSCOL',
    #     'AMARITL',
    #     'AMJIND',
    #     'AMJOCC',
    #     'ARACE',
    #     'AREORGN',
    #     'ASEX',
    #     'AUNMEM',
    #     'AUNTYPE',
    #     'AWKSTAT',
    #     'CAPGAIN',
    #     'CAPLOSS',
    #     'DIVVAL',
    #     'FEDTAX',
    #     'FILESTAT',
    #     'GRINREG',
    #     'GRINST',
    #     'HHDFMX',
    #     'HHDREL',
    #     'MARSUPWT',
    #     'MIGMTR1',
    #     'MIGMTR3',
    #     'MIGMTR4',
    #     'MIGSAME',
    #     'MIGSUN',
    #     'NOEMP',
    #     'PARENT',
    #     'PEARNVAL',
    #     'PEFNTVTY',
    #     'PEMNTVTY',
    #     'PENATVTY',
    #     'PRCITSHP',
    #     'PTOTVAL',
    #     'SEOTR',
    #     'TAXINC',
    #     'VETQVA',
    #     'VETYN',
    #     'WKSWORK']
