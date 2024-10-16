import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import os
import re 
import requests

# ECR URL:
#  https://www.fantasypros.com/nfl/rankings/ros-ppr-overall.php
## Fitz URL:
# # https://www.fantasypros.com/nfl/rankings/pat-fitzmaurice.php?type=ros&scoring=PPR&position=ALL

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

today_date = date.today().strftime("%Y-%m-%d").replace("-", "_")
today_date
def scrape_fantasypros_ros_ppr():
    url = "https://www.fantasypros.com/nfl/rankings/ros-ppr-overall.php"
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
    try:
        driver.get(url)
        print("Page loaded successfully")
        
        wait = WebDriverWait(driver, 10)
        table = wait.until(EC.presence_of_element_located((By.ID, "ranking-table")))
        print("Table found")
        
        # Scroll to load more rows
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Wait for new content to load
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            
            # Check if we have loaded enough rows
            rows = table.find_elements(By.CSS_SELECTOR, "tr.player-row")
            if len(rows) >= 200:
                break
        
        print(f"Loaded {len(rows)} rows")
        
        headers = [th.text for th in table.find_elements(By.TAG_NAME, "th") if th.text != "WSIT"]
        
        rows_data = []
        for tr in rows[:200]:  # Limit to 200 rows
            row = [td.text for td in tr.find_elements(By.TAG_NAME, "td") if "wsis-cell" not in td.get_attribute("class")]
            if row:
                rows_data.append(row)
        
        df = pd.DataFrame(rows_data, columns=headers)
        return df
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    finally:
        driver.quit()



# Use the function to get the rankings
ros_ppr_rankings = scrape_fantasypros_ros_ppr()


# Check if rankings were successfully scraped
if ros_ppr_rankings is not None:
    print(ros_ppr_rankings.head())
    print(f"Total rows: {len(ros_ppr_rankings)}")
    ros_ppr_rankings.columns = ros_ppr_rankings.columns.str.lower()
    ros_ppr_rankings = ros_ppr_rankings[['rk', 'player name', 'pos']]
    ros_ppr_rankings.to_csv(f"{base_path}basic_ecr_{today_date}.csv", index=False, mode='w+' )
    print(f"Rankings saved to {base_path}basic_ecr_{today_date}.csv")
else:
    print("Failed to scrape rankings. Please check the error messages above.")



