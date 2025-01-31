import os
import requests
import csv
import re
import json
import time
from datetime import datetime
from typing import List, Dict, TypedDict
from concurrent.futures import ThreadPoolExecutor
from langgraph.graph import StateGraph, END
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

# *****************************************************
#                   LangGraph State
# *****************************************************

class ProjectState(TypedDict):
    project_id: str
    title: str
    base_data: Dict
    news_urls: List[str]
    scraped_content: str
    financial_data: Dict
    final_data: Dict

class NewsGatherer:
    def __init__(self):
        self.news_api_key = os.getenv("NEWSAPI_KEY")
        self.serp_api_key = os.getenv("SERPAPI_KEY")  

    def fetch_news_from_newsapi(self, query):
        """Fetch news from NewsAPI with retry and exponential backoff"""
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={self.news_api_key}"
        retries = 1
        delay = 5  # Start with 5s delay
        logging.debug(f"Fetching news from NewsAPI for query: {query}")
        
        start_time = time.time()
        for _ in range(retries):
            response = requests.get(url)
            
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                elapsed_time = time.time() - start_time
                logging.debug(f"NewsAPI response received in {elapsed_time:.2f} seconds")
                return [article['url'] for article in articles]

            if response.status_code == 429:  # Too many requests
                logging.warning(f"NewsAPI rate limit hit! Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                logging.error(f"NewsAPI Error {response.status_code}: {response.text}")
                break  # No point retrying on other errors
        
        return []  # Return empty if all retries fail

    def fetch_news_from_serpapi(self, query):
        """Fetch news from SerpApi"""
        url = f"https://serpapi.com/search?q={query}&api_key={self.serp_api_key}&tbm=nws"  # The "tbm=nws" parameter specifies news results
        logging.debug(f"Fetching news from SerpApi for query: {query}")
        
        start_time = time.time()
        response = requests.get(url)

        elapsed_time = time.time() - start_time
        logging.debug(f"SerpApi response received in {elapsed_time:.2f} seconds")
        
        if response.status_code == 200:
            articles = response.json().get('news_results', [])
            return [article['link'] for article in articles]
        
        logging.error(f"SerpApi Error {response.status_code}: {response.text}")
        return []

    def gather(self, state):
        """Main function to fetch news from multiple sources"""
        keywords = ["cost overrun", "project delay", "budget overrun", "construction delay"]
        news_urls = []

        logging.debug("Starting news gathering process...")
        start_time = time.time()
        for keyword in keywords:
            query = f"{state['title']} {keyword}" # {keyword}"

            # Try SerpApi first
            urls = self.fetch_news_from_serpapi(query)
            if not urls:
                logging.debug(f"No results from SerpApi for {keyword}, switching to NewsAPI...")
                urls = self.fetch_news_from_newsapi(query)
            
            news_urls.extend(urls)

        state["news_urls"] = list(set(news_urls))  # Remove duplicates
        elapsed_time = time.time() - start_time
        logging.debug(f"News gathering completed in {elapsed_time:.2f} seconds")
        return state


class ContentScraper:
    def scrape(self, state: ProjectState) -> ProjectState:
        content = []
        logging.debug(f"Starting content scraping for {len(state['news_urls'])} URLs...")
        start_time = time.time()
        for url in state["news_urls"]:
            try:
                response = requests.get(url, timeout=10)
                content.append(re.sub('<[^<]+?>', '', response.text)[:2000])
            except Exception as e:
                logging.error(f"Error scraping URL {url}: {e}")
                continue
        state["scraped_content"] = "\n".join(content)
        elapsed_time = time.time() - start_time
        logging.debug(f"Content scraping completed in {elapsed_time:.2f} seconds")
        return state


class FinancialAnalyzer:
    def analyze(self, state: ProjectState) -> ProjectState:
        prompt = f"""
        Analyze text to extract:
        - Final project cost (numeric with currency)
        - Actual completion date (YYYY-MM-DD)
        - Currency (ISO code)
        Return JSON with keys: final_total, actual_date, currency
        
        Text: {state['scraped_content'][:500]}
        """
        
        logging.debug(f"Starting financial analysis for project: {state['title']}")
        start_time = time.time()
        try:
            response = model.generate_content(
                prompt,
                generation_config = genai.GenerationConfig(
                    max_output_tokens=300,
                    temperature=0.1,
                )
            )

            # print(f" llm gen : {response.text}")
            return json.loads(response.text)
            state["financial_data"] = json.loads(response['completion'])
        except Exception as e:
            logging.error(f"Error during financial analysis: {e}")
            state["financial_data"] = {}
        
        elapsed_time = time.time() - start_time
        logging.debug(f"Financial analysis completed in {elapsed_time:.2f} seconds")
        return state


class DataValidator:
    def validate(self, state: ProjectState) -> ProjectState:
        logging.debug(f"Starting data validation for project: {state['title']}")
        start_time = time.time()
        
        financial = state["financial_data"]
        base = state["base_data"]
        
        final_cost = float(re.sub(r"[^\d.]", "", financial.get("final_total", ""))) if financial.get("final_total", "").strip() else None  
        actual_date = self._parse_date(financial.get('actual_date', ''))

        state["final_data"] = {
            'title': base['project_name'],
            'country_name': ', '.join(base.get('countryname', ['N/A'])),
            'state_name': '',
            'city_name': '',
            'status': base.get('status', 'N/A'),
            'url': base.get('url', 'N/A'),
            'estimated_total_project_cost': float(base.get('totalamt', '0').replace(',', '')),
            'final_total_project_cost': final_cost,
            'currency': financial.get('currency', 'USD'),
            'estimated_completion_date': self._parse_date(base.get('closingdate', 'N/A')),
            'actual_completion_date': actual_date,
            "cost_overrun" : final_cost - float(base.get("totalamt", "0").replace(",", "")) if final_cost is not None else None,
            'time_delay': self._calculate_delay(base.get('closingdate', ''), actual_date),
            'citations': state["news_urls"] + [base.get('url', '')],
            'project_id' : base.get('id')
        }

        elapsed_time = time.time() - start_time
        logging.debug(f"Data validation completed in {elapsed_time:.2f} seconds")
        return state

    def _parse_date(self, date_str: str) -> str:
        """Try multiple date formats."""
        if not date_str or date_str.strip().upper() == "N/A":
            return "N/A"
        
        date_formats = [
            "%Y-%m-%d",              # 2024-06-30
            "%m/%d/%Y %I:%M:%S %p",  # 6/30/2024 12:00:00 AM
            "%m/%d/%Y",              # 6/30/2024
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt).date().isoformat()
            except ValueError:
                continue
        
        logging.error(f"Error parsing date: {date_str}")
        return "N/A"

    def _calculate_delay(self, est_date: str, act_date: str) -> int:
        """Calculate time delay between estimated and actual dates."""
        try:
            est = self._parse_date(est_date)
            act = self._parse_date(act_date)
            if est == "N/A" or act == "N/A":
                return 0

            est_dt = datetime.strptime(est, "%Y-%m-%d")
            act_dt = datetime.strptime(act, "%Y-%m-%d")
            return max(0, (act_dt - est_dt).days)
        except Exception as e:
            logging.error(f"Error calculating delay for dates {est_date}, {act_date}: {e}")
            return 0

# *****************************************************
#                   Agent Workflow
# *****************************************************

def create_workflow():
    logging.debug("Creating agent workflow...")
    workflow = StateGraph(ProjectState)
    
    # Add nodes
    workflow.add_node("news_gatherer", NewsGatherer().gather)
    workflow.add_node("content_scraper", ContentScraper().scrape)
    workflow.add_node("financial_analyzer", FinancialAnalyzer().analyze)
    workflow.add_node("data_validator", DataValidator().validate)
    
    # Set edges
    workflow.set_entry_point("news_gatherer")
    workflow.add_edge("news_gatherer", "content_scraper")
    workflow.add_edge("content_scraper", "financial_analyzer")
    workflow.add_edge("financial_analyzer", "data_validator")
    workflow.add_edge("data_validator", END)
    
    return workflow.compile()

# *****************************************************
#                   Main Execution
# *****************************************************

def fetch_world_bank_projects():
    """
    Fetches World Bank project data from the API.
    """
    url = "https://search.worldbank.org/api/v2/projects"
    params = {
        "format": "json",
        "status": "Closed",
        # "fl": "id,project_name,countryname,boardapprovaldate,closingdate,totalamt,url,status",
        "rows": "2"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("projects", {})
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error: {e}")
        return {}

def process_projects(projects: Dict[str, Dict]):
    logging.debug(f"Starting project processing for {len(projects)} projects...")
    workflow = create_workflow()
    results = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for project_id, project in projects.items():
            state = ProjectState(
                project_id=project_id,
                title=project.get('project_name', ''),
                base_data=project,
                news_urls=[],
                scraped_content='',
                financial_data={},
                final_data={})
            futures.append(executor.submit(workflow.invoke, state))
        
        for future in futures:
            results.append(future.result()["final_data"])

    logging.debug("Project processing completed")
    return results

if __name__ == "__main__":
    # Fetch and process projects
    projects = fetch_world_bank_projects()
    final_data = process_projects(projects)
    
    # Save results
    with open("project_metrics.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=final_data[0].keys())
        writer.writeheader()
        writer.writerows(final_data)
