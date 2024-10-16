WEEK_NUM = 7


# pip install googlesearch-python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import os
import re 
import requests
from bs4 import BeautifulSoup

# Step 1: Fetch the web page content

rb_url = "https://www.thescore.com/news/3066601"
wr_url = "https://www.thescore.com/news/3066602"
te_url = "https://www.thescore.com/news/3066603"
qb_url = "https://www.thescore.com/news/3088438"
# for url in [rb_url, wr_url, te_url, qb_url]:

print('hi')
base_path = 'C:/Users/16028/OneDrive/Documents/football_analytics/'
import datetime
from datetime import datetime
today_date = datetime.now().strftime("%Y-%m-%d").replace("-", "_")
print(today_date)
from googlesearch import search
positions = {"rb": "Running Backs", "wr": "Wide Recivers", "te": "Tight Ends", "qb": "Quarterbacks"}
for position, position_name in positions.items():
    # Step 1: Search for the correct URL using the title
    search_query = f"Fantasy: Trade Value Chart - {position_name} (Week {WEEK_NUM}) site:thescore.com"
    urls = list(search(search_query, num_results=1))

    # Step 2: Check if URLs were found
    if urls:
        url = urls[0]  # First search result URL
        print(f"Found URL: {url}")
        
        # Step 3: Fetch the web page content
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Step 4: Locate the table using the 'table-responsive' class
        table = soup.find('div', class_='table-responsive').find('table')

        # Step 5: Extract the table header (thead)
        thead = table.find('thead')
        headers = [header.text.strip() for header in thead.find_all('th')]

        # Step 6: Extract the table body (tbody) and rows (tr)
        tbody = table.find('tbody')
        rows = tbody.find_all('tr')

        # Step 7: Extract data from each row
        data = []
        for row in rows:
            cols = [col.text.strip() for col in row.find_all('td')]
            data.append(cols)

        # Step 8: Create a pandas DataFrame
        df = pd.DataFrame(data, columns=headers)

        # Step 9: Save to CSV or print the DataFrame
        df.to_csv(f'{base_path}boone_{position}_{today_date}.csv', index=False)

    # Display the DataFrame
        print(df)
    else:
        print("No URLs found!")


