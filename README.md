# Project_Analyzer

# Project Tracker AI Agent

Automated system for tracking project cost overruns and delays using AI agents.

## Features
- World Bank project data integration
- News/article scraping
- Gemini AI analysis
- Data validation pipeline

Automated data collection from official and news sources

AI-powered financial analysis

Data validation and standardization

Parallel processing for scalability

2. Architecture & Components
   
a. LangGraph Workflow
   graph LR
      A[News Gatherer] --> B[Content Scraper]
      B --> C[Financial Analyzer]
      C --> D[Data Validator]
      D --> E[Final Dataset]

   State Management: Uses ProjectState TypedDict to maintain context

   Parallel Execution: Processes multiple projects simultaneously

b. Core Components
- News Gatherer:

   Dual-source strategy (SerpAPI + NewsAPI)
   
   Intelligent query formulation with project-specific keywords
   
   Exponential backoff for rate limiting

- Content Scraper:

   Robust HTML cleaning with regex
   
   Content length optimization for LLM processing
   
   Error handling for failed requests

- Financial Analyzer:

   Gemini Pro LLM integration
   
   Structured JSON extraction prompt engineering
   
   Temperature control for consistent outputs

- Data Validator:

   Multi-format date parsing
   
   Financial calculations (cost overrun/time delay)
   
   Data type normalization

## Setup
1. Clone repo
2. Install requirements:
   ```bash
   pip install -r requirements.txt


### Scalability Improvements

1. **Enhanced Sources**:
   - Add government procurement portals
   - Integrate RSS feeds
   - Use Scrapy spider + Common Crawl  (for open-source)

2. **Monitoring**:
   - Prometheus/Grafana for system metrics
   - Sentry for error tracking

3. **Advanced AI**:
   - Fine-tuned LLM models
   - Multi-source verification system
   - Confidence scoring for metrics
   - Add response caching
4. **Data Pipeline**:
   - Use Apache Airflow for workflow orchestration

