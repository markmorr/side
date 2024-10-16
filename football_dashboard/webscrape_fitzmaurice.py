import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import date

def scrape_fantasy_rankings(expert_name, url):
    """
    Scrapes the fantasy football rankings from the provided URL and saves the data to a CSV file.
    
    Parameters:
    expert_name (str): The name of the expert whose rankings are being scraped (used in the filename).
    url (str): The URL of the page to scrape.
    
    Returns:
    pd.DataFrame: The scraped rankings as a DataFrame.
    """
    
    # Send a GET request to the webpage
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the table element by its id ('data')
        table = soup.find('table', {'id': 'data'})

        # Create lists to store the data
        headers = []
        rows = []

        # Extract headers from the table (th elements)
        for th in table.find_all('th'):
            headers.append(th.text.strip())

        # Extract data from rows (tr and td elements)
        for tr in table.find_all('tr')[1:]:  # Skip the header row
            row = [td.text.strip() for td in tr.find_all('td')]
            if row:
                rows.append(row)

        # Create a DataFrame from the extracted data
        rankings_df = pd.DataFrame(rows, columns=headers)

        # Prepare the filename with today's date
        today_date = date.today().strftime("%Y-%m-%d").replace("-", "_")
        base_path = 'C:/Users/16028/OneDrive/Documents/football_analytics/'
        filename = f"{base_path}{expert_name}_{today_date}.csv"

        # Save the DataFrame to a CSV file
        rankings_df.to_csv(filename, index=False, mode='w')
        print(f"Rankings saved to {filename}")

        # Return the DataFrame
        return rankings_df

    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return None

expert = "fitzmaurice"
url = "https://www.fantasypros.com/nfl/rankings/pat-fitzmaurice.php?type=ros&scoring=PPR&position=ALL"

df = scrape_fantasy_rankings(expert, url)
print(df.head())  






df.columns


