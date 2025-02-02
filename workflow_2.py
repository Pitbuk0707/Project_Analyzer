import os
import requests
import csv
import re
import json
import time
from datetime import datetime
from typing import List, Dict, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from langgraph.graph import StateGraph, END
import logging
from googleapiclient.discovery import build
from firecrawl import FirecrawlApp
import google.generativeai as genai
from dotenv import load_dotenv
import os
from groq import Groq


# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Configure Gemini
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.search_service = build("customsearch", "v1", developerKey=self.google_api_key)
        
    def _fetch_google_results(self, query: str) -> List[str]:
        """Enhanced Google search with news focus and ranking"""
        try:
            print(f"Fetching Google results for query: {query}")
            result = self.search_service.cse().list(
                q=query,
                cx=self.google_cse_id,
                num=10
            ).execute()
            
            items = result.get('items', [])
            ranked_items = sorted(items, 
                key=lambda x: self._calculate_relevance(x, query), 
                reverse=True
            )
            print(f"Ranked items: {ranked_items}")
            return [item['link'] for item in ranked_items]
        
        except Exception as e:
            logging.error(f"Google Search API error: {str(e)[:100]}")
            return []

    def _calculate_relevance(self, item: Dict, query: str) -> float:
        """Calculate relevance score based on title/snippet match"""
        text = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
        query_terms = query.lower().split()
        return sum(1 for term in query_terms if term in text)

    def gather(self, state: ProjectState) -> ProjectState:
        """Google-powered news gathering with smart query formulation"""
        print(f"Gathering news for project: {state['title']}")
        keywords = ["Actual Completion Cost", "Actual Completion Date"]
        news_urls = []
        
        base_query = f"{state['title']} {state['base_data'].get('countryshortname', '')}"
        
        for keyword in keywords:
            full_query = f"{base_query} {keyword}"
            print(f"Searching for keyword: {keyword}")
            news_urls += self._fetch_google_results(full_query)
            time.sleep(1)  # Rate limit protection

        state["news_urls"] = list(dict.fromkeys(news_urls))
        state["news_urls"] = [url for url in state["news_urls"] if not url.endswith(".pdf")]

        print(f"News URLs gathered: {state['news_urls']}")
        return state

class ContentScraper:
    def __init__(self):
        self.app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
        
    def scrape(self, state: ProjectState) -> ProjectState:
        """Firecrawl-powered content extraction with markdown cleaning"""
        print(f"Scraping content for project: {state['title']}")
        logging.debug(f"Scraping {len(state['news_urls'])} URLs with Firecrawl...")
        try:
            result = self.app.batch_scrape_urls(state["news_urls"][:5], params={
                'formats': ['markdown']
            })
            markdown_list = [item['markdown'] for item in result['data']]
            cleaned_contents = [re.sub(r'\n{3,}', '\n\n', item) for item in markdown_list]
            state["scraped_content"] = cleaned_contents
            print(f"Scraped content: {state['scraped_content']}")
        except Exception as e:
            logging.warning(f"Firecrawl error: {str(e)[:100]}")
            state["scraped_content"] = []
        return state


class FinancialAnalyzer:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
    def analyze(self, state: ProjectState) -> ProjectState:
        """Analyze content and accumulate partial data from all URLs"""
        print(f"Analyzing financial data for project: {state['title']}")
        financial_data = {
            "final_total": None,
            "actual_date": None,
            "currency": None
        }

        for idx, content in enumerate(state["scraped_content"][:5]):
            try:
                print(f"Analyzing content from URL: {state['news_urls'][idx]}")
                prompt = f"""
                Given the following content, analyze it to extract the following details for the project titled: {state['title']}:
                - Actual Completion Cost (numeric value)
                - Actual Completion Date (YYYY-MM-DD format)
                - Currency Code (3-letter ISO code)

                Return a JSON with keys: final_total, actual_date, and currency. If not found, return 'None'.

                Content: {content}
                """
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "you are expert in extracting key information from scraped content."
                        },
                        {  "role": "user",
                            "content": prompt,
                        }
                    ],
                    model="deepseek-r1-distill-llama-70b",
                    temperature=0.01,
                )

                response = chat_completion.choices[0].message.content
                print(f"LLM response: {response}")

                # Extract JSON from response
                match = re.search(r'```json\s*\n(.*?)\n```', response, re.DOTALL)
                if match:
                    extracted_json = match.group(1)
                    actual_json = json.loads(extracted_json)
                    print(f"Extracted JSON: {actual_json}")

                    # Accumulate partial data
                    for key in ["final_total", "actual_date", "currency"]:
                        if actual_json.get(key) not in [None, "None", "Objective not met"]:
                            financial_data[key] = actual_json[key]

            except Exception as e:
                logging.debug(f"Analysis attempt failed: {str(e)[:100]}")
                continue

        print(f"Accumulated financial data: {financial_data}")
        state["financial_data"] = financial_data
        return state


class DataValidator:
    def validate(self, state: ProjectState) -> ProjectState:
        print(f"Validating data for project: {state['title']}")
        logging.debug(f"Starting data validation for project: {state['title']}")
        start_time = time.time()
        
        financial = state["financial_data"]
        base = state["base_data"]

        # Handle case where LLM returns "Objective not met"
        if financial.get("final_total") == "Objective not met":
            print("LLM could not extract financial data. Setting all financial fields to None.")
            financial = {
                "final_total": None,
                "actual_date": None,
                "currency": None
            }

        actual_currency = financial.get("currency")
        final_total = financial.get("final_total", None)

        # Convert final_total to float if it's a valid numeric string
        if final_total is not None and isinstance(final_total, str):
            try:
                final_cost = float(re.sub(r"[^\d.]", "", final_total)) if final_total.strip() else None
            except ValueError:
                print(f"Could not convert final_total to float: {final_total}. Setting to None.")
                final_cost = None
        elif isinstance(final_total, (int, float)):
            final_cost = float(final_total)
        else:
            final_cost = None

        # Convert currency if necessary
        if final_cost is not None and actual_currency and actual_currency.lower() != "usd":
            final_cost *= 0.0012  # Example conversion rate (adjust as needed)

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
            "cost_overrun": final_cost - float(base.get("totalamt", "0").replace(",", "")) if final_cost is not None else None,
            'time_delay': self._calculate_delay(base.get('closingdate', ''), actual_date),
            'citations': state["news_urls"] + [base.get('url', '')],
            'project_id': base.get('id')
        }

        elapsed_time = time.time() - start_time
        logging.debug(f"Data validation completed in {elapsed_time:.2f} seconds")
        print(f"Final data: {state['final_data']}")
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
    
    workflow.add_node("news_gatherer", NewsGatherer().gather)
    workflow.add_node("content_scraper", ContentScraper().scrape)
    workflow.add_node("financial_analyzer", FinancialAnalyzer().analyze)
    workflow.add_node("data_validator", DataValidator().validate)
    
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
    url = "https://search.worldbank.org/api/v2/projects"
    params = {
        "format": "json",
        "countrycode": "IN",
        "sectorcode": "TI",  # Sector code for Transportation
        "status": "Closed",
        "rows": "50"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        all_data = data.get("projects", {})

        # target_ids = ['P121185','P096021','P130339','P124639','P090585','P077856','P050668','P074018']
        # filtered_json = {target_id: all_data[target_id] for target_id in target_ids}

        return all_data

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return {}

def process_projects(projects: Dict[str, Dict]):
    logging.debug(f"Starting project processing for {len(projects)} projects...")
    workflow = create_workflow()
    
    # Check if file exists to determine if header is needed
    file_exists = os.path.isfile("project_metrics_llama2.csv")
    
    with open("project_metrics_llama2.csv", "a", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'title', 'country_name', 'state_name', 'city_name', 'status', 'url',
            'estimated_total_project_cost', 'final_total_project_cost', 'currency',
            'estimated_completion_date', 'actual_completion_date', 'cost_overrun',
            'time_delay', 'citations', 'project_id'
        ])
        
        if not file_exists:
            writer.writeheader()
        
        for project_id, project in projects.items():
            try:
                print(f"Processing project: {project_id}")
                state = ProjectState(
                    project_id=project_id,
                    title=project.get('project_name', ''),
                    base_data=project,
                    news_urls=[],
                    scraped_content='',
                    financial_data={},
                    final_data={}
                )
                result = workflow.invoke(state)
                writer.writerow(result["final_data"])
                f.flush()  # Ensure data is written to file immediately
                print(f"Successfully processed and saved project: {project_id}")
            except Exception as e:
                logging.error(f"Error processing project {project_id}: {str(e)[:100]}")
                continue

    logging.debug("Project processing completed")

if __name__ == "__main__":
    projects = fetch_world_bank_projects()
    process_projects(projects)