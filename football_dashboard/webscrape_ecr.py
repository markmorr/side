import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import os
import re 
import requests
# from bs4 import BeautifulSoup

# # Step 1: Fetch the web page content
# base_path = 'C:/Users/16028/OneDrive/Documents/football_analytics/'
# import datetime
# from datetime import datetime
# today_date = datetime.now().strftime("%Y-%m-%d").replace("-", "_")
# print(today_date)
# from googlesearch import search
# WEEK_NUM = 5


# # I want help scraping the following URL:
#  https://www.fantasypros.com/nfl/rankings/ros-ppr-overall.php
## Let's try this URL instead first: 
# # https://www.fantasypros.com/nfl/rankings/pat-fitzmaurice.php?type=ros&scoring=PPR&position=ALL
# # This one is tricky becaus I want you to also somehow select the Experts that I want to generate the right table
# # (On the webpage there is a special button that says "Experts" that lets you 
# pick which experts rankings you want to generate the rankinss from)

# # First things first you an use the default Experts without worryin about that yet. Help me webscrape the table from the url above
# # The table I want you to webscrape is the one under this html I think:
#  <table id="ranking-table" class="table table-striped player-table table-hover js-table-caption"><thead><!----> <tr class="primary-header__tr"><th class="rank-cell sticky-cell sticky-cell-one sorted__th sorted__th--desc mpb-player__border">RK <!----></th><th class="wsis-cell hide-print hide-export sticky-cell sticky-cell-two sticky-cell-two--left-extra wsis-cell hide-print hide-export">WSIT <!----></th><th class="player-cell">Player Name <!----></th><th class="">POS <!----></th><th class="matchup-star-cell">SOS SEASON <!----></th><th class="matchup-star-cell">SOS PLAYOFFS <!----></th><th class="">ECR VS. ADP <!----></th></tr></thead> <tbody><!----> <tr class="player-row mpb-player__tr mpb-player__tr--taken"><td class="sticky-cell sticky-cell-one mpb-player__border--taken">1</td><td class="sticky-cell sticky-cell-two sticky-cell-two--left-extra wsis-cell hide-print hide-export"><input type="checkbox" class="table-checkbox" aria-label="Toggle Saquon Barkley to compare with other players." value="[object Object]"></td><td class=""><div class="player-cell player-cell__td"

# # ... existing imports ...

# # Add these imports if not already present
# from bs4 import BeautifulSoup
# import requests
# import pandas as pd

# # ... existing code ...
# # ... existing imports ...
# def scrape_fantasypros_ros_ppr():
#     url = "https://www.fantasypros.com/nfl/rankings/ros-ppr-overall.php"
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         print("HTTP request successful")
#         soup = BeautifulSoup(response.content, 'html.parser')
        
#         # Print out all table ids found on the page
#         table_ids = [table.get('id') for table in soup.find_all('table') if table.get('id')]
#         print(f"Table IDs found on the page: {table_ids}")
        
#         # Find the table
#         table = soup.find('table', id='ranking-table')
#         if not table:
#             print("Table with id 'ranking-table' not found.")
#             # Try to find the table by class instead
#             table = soup.find('table', class_='table table-striped player-table table-hover js-table-caption')
#             if table:
#                 print("Found table using class names instead of id.")
#             else:
#                 print("Table not found using class names either.")
#                 return None
        
#         # Rest of the function remains the same...
        
#     except requests.RequestException as e:
#         print(f"An error occurred while fetching the URL: {e}")
#         return None
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         return None

# # Use the function to get the rankings
# ros_ppr_rankings = scrape_fantasypros_ros_ppr()

# # Rest of the code remains the same...
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

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
    ros_ppr_rankings.to_csv(f"{base_path}ros_ppr_rankings_{today_date}.csv", index=False)
    print(f"Rankings saved to {base_path}ros_ppr_rankings_{today_date}.csv")
else:
    print("Failed to scrape rankings. Please check the error messages above.")







from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import date

def scrape_pat_fitzmaurice_rankings():
    # URL of the page to scrape
    url = "https://www.fantasypros.com/nfl/rankings/pat-fitzmaurice.php?type=ros&scoring=PPR&position=ALL"

    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode (uncomment if needed)

    # Set up the Chrome driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        # Navigate to the URL
        driver.get(url)
        print("Page loaded successfully")

        # Wait for the table to load
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "ranking-table")))
        print("Table found")

        # Get the page source and create a BeautifulSoup object
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Find the table
        table = soup.find('table', {'id': 'ranking-table'})

        if table:
            # Extract table headers
            headers = [th.text.strip() for th in table.find_all('th') if th.text.strip() != "WSID"]

            # Extract table rows
            rows = []
            for tr in table.find_all('tr', class_='player-row'):
                row = [td.text.strip() for td in tr.find_all('td') if 'wsid' not in td.get('class', [])]
                if row:
                    rows.append(row)

            # Create a DataFrame
            df = pd.DataFrame(rows, columns=headers)

            return df

        else:
            print("Table not found. Please check the page structure.")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    finally:
        # Close the browser
        driver.quit()

# Use the function to get the rankings
pat_fitzmaurice_rankings = scrape_pat_fitzmaurice_rankings()

# Check if rankings were successfully scraped
if pat_fitzmaurice_rankings is not None:
    print(pat_fitzmaurice_rankings.head())
    print(f"Total rows: {len(pat_fitzmaurice_rankings)}")

    # Save to CSV
    today_date = date.today().strftime("%Y-%m-%d")
    filename = f"pat_fitzmaurice_rankings_{today_date}.csv"
    pat_fitzmaurice_rankings.to_csv(filename, index=False)
    print(f"Rankings saved to {filename}")
else:
    print("Failed to scrape rankings. Please check the error messages above.")













from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import date

def scrape_pat_fitzmaurice_rankings():
    url = "https://www.fantasypros.com/nfl/rankings/pat-fitzmaurice.php?type=ros&scoring=PPR&position=ALL"
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--ignore-ssl-errors')
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
    try:
        driver.get(url)
        print("Page loaded successfully")
        
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, "data")))
        print("Table found")
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        table = soup.find('table', {'id': 'data'})
        
        if table:
            headers = [th.text.strip() for th in table.find('thead').find_all('th')]
            
            rows = []
            for tr in table.find('tbody').find_all('tr'):
                row = [td.text.strip() for td in tr.find_all('td')]
                if row:
                    rows.append(row)
            
            df = pd.DataFrame(rows, columns=headers)
            return df
        else:
            print("Table not found. Please check the page structure.")
            return None
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    finally:
        driver.quit()

# Use the function to get the rankings
pat_fitzmaurice_rankings = scrape_pat_fitzmaurice_rankings()

# Check if rankings were successfully scraped
if pat_fitzmaurice_rankings is not None:
    print(pat_fitzmaurice_rankings.head())
    print(f"Total rows: {len(pat_fitzmaurice_rankings)}")
    
    today_date = date.today().strftime("%Y-%m-%d")
    filename = f"pat_fitzmaurice_rankings_{today_date}.csv"
    pat_fitzmaurice_rankings.to_csv(filename, index=False)
    print(f"Rankings saved to {filename}")
else:
    print("Failed to scrape rankings. Please check the error messages above.")