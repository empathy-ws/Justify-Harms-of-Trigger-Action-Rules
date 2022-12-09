import pandas as pd
import numpy as np
import io, csv

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def sub(l1, l2):
    l3 = [x for x in l1 if x not in l2]
    return l3

###############################################################################
# Remove duplicate
###############################################################################

def unique(list1):
    x = np.array(list1)
    return list(np.unique(x))

###############################################################################
# Retrieval of services in the dataset
###############################################################################

def getServices(train_df):
    triggerChannelTitle = list(train_df['triggerChannelTitle'])
    triggerChannelTitle = unique(triggerChannelTitle)

    actionChannelTitle = list(train_df['actionChannelTitle'])
    actionChannelTitle = unique(actionChannelTitle)

    combination = triggerChannelTitle + actionChannelTitle
    return unique(combination)

###############################################################################
# Load data
###############################################################################

col_names = ['triggerTitle','triggerChannelTitle','actionTitle','actionChannelTitle','title','desc','target']

full_train_df = pd.read_csv('./FullDataset_116k.csv',skiprows=1,sep=';',names=col_names,encoding = "ISO-8859-1")

full_services = getServices(full_train_df)

col_names = ['triggerTitle','triggerChannelTitle','actionTitle','actionChannelTitle','title','desc','target', 'justification']

subset_train_df = pd.read_csv('./training_dataset.csv',skiprows=1,sep=';',names=col_names,encoding = "ISO-8859-1")

subset_services = getServices(subset_train_df)

###############################################################################
# Save services not present in the training set
###############################################################################

unknown_service = sub(full_services, subset_services)

col_names = ['service']

with io.open('./unknowm_services.csv', mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=col_names, delimiter=";")
    writer.writeheader()
    for i in range(len(unknown_service)):
        writer = csv.DictWriter(csv_file, fieldnames=col_names, delimiter=";")
        writer.writerow({'service': unknown_service[i]})




