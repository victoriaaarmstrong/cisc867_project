import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_data_file(file_name):
    """
    unpickle and scale the data
    :param file_name:
    :return: dataframe
    """

    unpickled = pd.read_pickle(file_name)

    ## Get the BVP data from the whole set
    df = pd.DataFrame.from_dict(unpickled['signal']['wrist']['BVP'])
    df.columns = ['BVP']

    return df


def create_data():
    participants = ["S2.pkl", "S3.pkl", "S4.pkl"]

    window = 100
    increment = 50
    drop_amount = 20

    for participant in participants:
        ## Make sure the data is labeled
        p_id = participant.replace(".pkl", "_")
        data = read_data_file("participants/"+participant)

        ## Drop negative values and Nan values
        id_names = data[data['BVP'] <= 0].index
        data.drop(id_names, inplace=True)
        data.dropna()

        ## Initialize counters
        lengthDf = len(data)
        start = 0
        end = window
        counter = 1

        while end < lengthDf:
            ## Get real and sparse dataframe copies
            real = data.iloc[start:end].copy()
            real.columns = ['BVP']
            sparse = data.iloc[start:end].copy()
            sparse.columns = ['BVP']

            ## Remove percentage of the data that you don't want
            drop_id = np.random.choice(sparse.index, drop_amount, replace=False)
            sparse.loc[drop_id, 'BVP'] = float("nan")

            ## Name the data
            real_name = p_id + str(counter)
            sparse_name = p_id + str(counter)

            ## Plot real
            fig = plt.figure(figsize=(2.50, 2.50))
            plt.axis("off")
            fig1 = real['BVP'].plot().get_figure()
            fig1.savefig('data/real/' + real_name)
            plt.close()

            ## Plot fake
            fig = plt.figure(figsize=(2.50, 2.50))
            plt.axis("off")
            fig2 = sparse['BVP'].plot().get_figure()
            fig2.savefig('data/sparse/' + sparse_name)
            plt.close()

            start = start + increment
            end = end + increment
            counter += 1

        print("done " + participant)

    print("all generated")

    return

