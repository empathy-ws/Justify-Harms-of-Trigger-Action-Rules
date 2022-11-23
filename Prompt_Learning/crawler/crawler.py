from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import pandas as pd
import numpy as np
import io, csv

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
# Retrieval of a service description from the Google Play Store
###############################################################################

def getDescription(service):
    options = Options()
    options.headless = True
    options.add_argument("--window-size=1920,1200")
    driver = webdriver.Chrome(options=options, service=Service(ChromeDriverManager().install()))

    url = 'https://play.google.com/store/search?q={}&c=apps&hl=en_US&gl=US'.format(service)
    driver.get(url)

    description = ""

    try:
        items = driver.find_elements(by=By.TAG_NAME, value='a')
        for item in items:
            if item.get_attribute("href").__contains__("details"):
                print(str(item.get_attribute("href"))+"&hl=en_US&gl=US")
                driver.get(str(item.get_attribute("href"))+"&hl=en_US&gl=US")
                local_description = driver.find_element(by=By.XPATH, value='//div[@data-g-id="description"]')
                description += local_description.text
                break

        driver.quit()

        return description
    except:
        print("Error:", service)
        return ""

###############################################################################
# Load data
###############################################################################

col_names = ['triggerTitle','triggerChannelTitle','actionTitle','actionChannelTitle','title','desc','target']

train_df = pd.read_csv('./FullDataset_116k.csv',skiprows=1,sep=';',names=col_names,encoding = "ISO-8859-1")

services = getServices(train_df)

###############################################################################
# Save descriptions
###############################################################################

col_names = ['service','description']

with io.open('./services_description.csv', mode='a', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=col_names, delimiter=";")
    writer.writeheader()

for i in range(0, len(services)):
    description = getDescription(services[i])
    print('-' * 40 + services[i] + " " + str(i) + '-' * 40)
    print(services[i], ":", description)
    with io.open('./services_description.csv', mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=col_names, delimiter=";")
        writer.writerow({'service': services[i], 'description': description})
