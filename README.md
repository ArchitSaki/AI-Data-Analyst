# AI Data Analyst

An AI-powered data analysis application that enables users to upload CSV datasets, ask questions in natural language, generate SQL-based insights, visualize results, and explore business intelligence reports through an interactive API-driven workflow.

The application uses **Groq-powered LLM inference, FastAPI, Pandas, and DuckDB** to simplify data analysis without requiring users to write SQL manually.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Environment Configuration](#environment-configuration)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Example Use Cases](#example-use-cases)
- [Testing](#testing)
- [Error Handling and Security](#error-handling-and-security)
- [Current Scope and Limitations](#current-scope-and-limitations)
- [Future Improvements](#future-improvements)
- [Author](#author)

## Overview

Traditional data analysis often requires knowledge of SQL, data manipulation libraries, and visualization tools. AI Data Analyst provides a simpler interface where users can upload structured data and ask questions using natural language.

The application processes uploaded CSV files, uses an LLM to assist with analysis, executes SQL queries against the available datasets, and returns structured results that can be used for further analysis and visualization.

It also includes business insights, data quality checks, anomaly detection, chart generation, and conversation-related functionality.

## Key Features

### 1. Natural-Language Data Analysis

- Ask questions about uploaded CSV datasets using plain English.
- Generate SQL queries from natural-language questions.
- Execute analytical queries against the uploaded data.
- Return query results and human-readable explanations.
- Support single-dataset and multi-dataset analysis.

### 2. Multi-CSV Analysis

- Upload multiple CSV files in one request.
- Reference multiple datasets during analysis.
- Support analytical questions that require combining related datasets.
- Validate file types and reject duplicate filenames.
- Preserve compatibility with single-file uploads.

### 3. Business Insights

- Generate business insight reports from structured datasets.
- Produce calculated findings and supporting evidence.
- Generate an AI-assisted business summary.
- Provide a fallback summary when AI-generated summarization is unavailable.

### 4. Data Quality Analysis

- Inspect datasets for potential data quality problems.
- Identify issues relevant to reliable analysis.
- Integrate data quality functionality into the analytical workflow.

### 5. Anomaly Detection

- Identify potential anomalies in datasets.
- Integrate anomaly analysis with the broader data analysis workflow.

### 6. Data Visualization

- Generate charts from analytical results.
- Support category comparisons and appropriate chart types.
- Make analytical results easier to interpret visually.

### 7. Conversation Context

- Support conversation IDs in API requests.
- Associate analysis requests with conversation context.
- Support conversation-related workflows in the application.

### 8. API-Based Design

- Expose functionality through FastAPI endpoints.
- Accept CSV uploads using multipart form requests.
- Return structured JSON responses.
- Provide automatic API documentation through FastAPI.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application and analysis logic |
| FastAPI | REST API development |
| Groq API | LLM inference |
| Pandas | CSV processing and tabular data manipulation |
| DuckDB | SQL-based analytical query execution |
| LangGraph | Graph-based analysis workflow and conversation orchestration |
| Pydantic | Data validation where used |
| Pytest | Automated testing |
| Git and GitHub | Version control and source code hosting |

Additional libraries are used for charting, data quality analysis, and anomaly detection.


## 🚀 Project Demo

Explore the live application and try AI-powered data analysis.

- **Live Application:** [AI Data Analyst](https://ai-data-analyst-ruddapu5uz5r9kwuou8zgw.streamlit.app/)
- **Source Code:** [GitHub Repository](https://github.com/ArchitSaki/AI-Data-Analyst)
- **Video Demonstration:** [Demo Video](https://drive.google.com/file/d/1lRpLwpYXTGQXJlrrWu9xn5zXup9x0A-i/view?usp=sharing)

### Try It Out

1. Open the live application.
2. Upload a CSV dataset.
3. Ask questions about your data in natural language.
4. Explore generated SQL queries, analytical results, visualizations, and business insights.



## System Architecture

The application follows a modular architecture that separates API handling, data processing, AI-assisted analysis, SQL execution, and analytical services.

```text
                  User / API Client
                         |
                         v
                    FastAPI API
                         |
                         v
                 CSV Upload & Validation
                         |
                         v
                 Data Loading (Pandas)
                         |
                         v
              AI-Assisted Analysis Workflow
                  (Groq / LangGraph)
                         |
                         v
                SQL Query Processing
                         |
                         v
                  DuckDB Execution
                         |
                         v
              Structured Analysis Results
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
          Answers      Charts     Insights
                         |
                         v
                 JSON API Response
```

The business insights, data quality, and anomaly detection services provide additional analytical capabilities.

### Design Principles

- **Separation of concerns:** API routes and analytical services have separate responsibilities.
- **Modularity:** Analytical functionality is organized into individual services.
- **Input validation:** Uploaded files are validated before processing.
- **Testability:** Core functionality and API behavior are covered by automated tests.
- **Error handling:** Expected validation and analysis errors are converted into appropriate HTTP responses.

## Project Structure

The project is organized into API routes, service modules, and automated tests.

```text
ai-data-analyst/
│
├── app/
│   ├── api/
│   │   └── chat.py
│   │
│   ├── services/
│   │   ├── ai_analyst.py
│   │   ├── data_loader.py
│   │   ├── business_insights.py
│   │   ├── dataset_store.py
│   │   ├── charting.py
│   │   └── ...
│   │
│   └── ...
│
├── tests/
│   ├── test_ai_analyst_contracts.py
│   ├── test_analysis_graph.py
│   ├── test_analytics.py
│   ├── test_anomaly_integration.py
│   ├── test_business_insights.py
│   ├── test_business_insights_integration.py
│   ├── test_charting.py
│   ├── test_chat_api.py
│   ├── test_data_quality.py
│   ├── test_data_quality_integration.py
│   └── test_sql_engine.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

*Note: The structure above highlights the main known modules. Keep the actual filenames and directories in this README aligned with the repository before submission.*

## Getting Started

### Prerequisites

Install the following before running the project:

- Python 3.12 or another version supported by the project dependencies.
- Git.
- A Groq API key.
- A terminal such as PowerShell, Command Prompt, or a VS Code terminal.

### 1. Clone the Repository

```bash
git clone https://github.com/ArchitSaki/AI-Data-Analyst.git
cd AI-Data-Analyst
```

### 2. Create a Virtual Environment

For Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is restricted by your PowerShell policy, use the virtual environment's Python executable directly.

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root and configure the environment variables required by your application's configuration.

Example:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
```

If the application uses an environment variable to select the Groq model, configure that variable according to the actual settings in the codebase.

**Important:** Never commit your real API key or other secrets to GitHub. Keep `.env` excluded through `.gitignore`. Commit a `.env.example` file containing placeholders instead.

## Running the Application

Start the FastAPI server using the application's actual entry-point module. For example, if the FastAPI instance is defined in `app.main`:

```powershell
uvicorn app.main:app --reload
```

If your entry point differs, replace `app.main:app` with the correct module and application variable.

Once the server is running, open:

- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

These interfaces allow you to inspect endpoints and send test requests interactively.

## API Documentation

### 1. Ask a Question — `POST /api/ask`

Accepts a natural-language question and one or more CSV files.

**Content type:** `multipart/form-data`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `question` | String | Yes | Natural-language analytical question |
| `file` | CSV file | No* | Single-file upload |
| `files` | CSV files | No* | Multi-file upload |
| `conversation_id` | String | No | Conversation identifier |

*At least one CSV file must be supplied through `file` or `files`.*

Example using cURL:

```bash
curl -X POST "http://127.0.0.1:8000/api/ask" \
  -F "question=Show total revenue by region" \
  -F "file=@sales.csv"
```

For multiple files:

```bash
curl -X POST "http://127.0.0.1:8000/api/ask" \
  -F "question=Join customers and orders" \
  -F "files=@customers.csv" \
  -F "files=@orders.csv"
```

The response contains the analysis output. Depending on the request and analysis workflow, the response can include fields such as the question, generated SQL, answer, columns, results, and conversation ID.

Example response structure:

```json
{
  "conversation_id": "example-conversation",
  "question": "Show total revenue by region",
  "answer": "Revenue summarized by region.",
  "sql": "SELECT region, SUM(revenue) FROM sales GROUP BY region",
  "columns": ["region", "total_revenue"],
  "results": []
}
```

The response above is illustrative; actual results and fields depend on the query and implementation.

### 2. Generate Business Insights — `POST /api/insights`

Accepts a CSV file and generates a business insights report.

**Content type:** `multipart/form-data`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file` | CSV file | Yes | Dataset to analyze |
| `conversation_id` | String | No | Optional identifier for the report and associated dataset |

Example:

```bash
curl -X POST "http://127.0.0.1:8000/api/insights" \
  -F "file=@sales.csv"
```

The response includes a conversation identifier, calculated insights, a business summary, and an analysis type indicating `business_insights`, subject to the generated report.

### HTTP Error Responses

The API uses HTTP errors to communicate invalid requests and analysis failures.

| Status | Meaning |
|---|---|
| `400 Bad Request` | Missing files, invalid file types, duplicate filenames, or input validation errors |
| `422 Unprocessable Entity` | Certain analysis or execution failures |
| `200 OK` | Successful request |

For exact request and response schemas, use the running application's Swagger UI.

## Example Use Cases

### Sales Analysis

Upload a sales dataset and ask:

- What is the total revenue?
- Which region generates the highest revenue?
- What are the monthly sales trends?
- Which products contribute the most revenue?

### Customer and Order Analysis

Upload customer and order CSV files and ask:

- How many orders belong to each customer?
- Which customers have the highest order value?
- What is the total revenue by customer segment?

### Data Quality

Use the data quality functionality to investigate potential missing values, inconsistencies, and other issues that could affect analytical results.

### Anomaly Detection

Use anomaly analysis to investigate unusual values or patterns that may deserve further inspection.

### Business Reporting

Generate a business insights report from a dataset and review the calculated findings alongside the AI-generated summary.

## Testing

The project includes an automated test suite built with Pytest.

The tests cover multiple areas, including:

- AI analysis contracts.
- Analysis graph behavior.
- Analytics functionality.
- Anomaly detection integration.
- Business insights and API integration.
- Chart generation.
- Chat API validation and file uploads.
- Data quality checks and integration.
- SQL engine behavior and safety.

### Run All Tests

From the project root:

```powershell
python -m pytest -q
```

Or, using the project's Windows virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Run an individual test module:

```powershell
python -m pytest tests/test_chat_api.py -q
```

**Current verification:** The full automated test suite passed during local development on October 20, 2026.

This records the observed test result; it does not imply that every possible input, production environment, or deployment scenario has been tested.

## Error Handling and Security

The application includes input validation and error handling for common API scenarios.

- Validates CSV file extensions before processing uploads.
- Rejects requests without an uploaded dataset.
- Rejects duplicate filenames in multi-file analysis requests.
- Handles CSV loading and validation errors.
- Converts selected analysis errors into HTTP responses.
- Closes uploaded files after request processing.
- Uses environment-based configuration for API credentials.

### Security Considerations

Generated SQL and uploaded data should be treated as untrusted inputs. Before production deployment, ensure that SQL validation prevents unauthorized statements, restricts access to permitted tables, and handles resource-intensive queries safely.

Never expose API keys in source code, logs, responses, or public repositories. Uploaded data should also be handled according to appropriate privacy and retention requirements.

## Current Scope and Limitations

- Input data is expected to be in CSV format.
- AI-generated answers and SQL can occasionally be incorrect and should be validated for important decisions.
- Analysis quality depends on dataset structure, column names, data quality, and question clarity.
- Multi-file analysis requires compatible schemas or meaningful relationships between datasets.
- The current test suite verifies defined application behavior, not every possible dataset or production workload.
- Deployment, authentication, persistent storage, and other production capabilities should be evaluated separately before public release.

## Future Improvements

Potential future enhancements include:

- A polished interactive web frontend.
- More advanced multi-dataset joins and schema discovery.
- Additional visualization types and exportable reports.
- Stronger SQL validation and query resource limits.
- More comprehensive end-to-end testing.
- User authentication and access control.
- Persistent conversation and dataset storage.
- Deployment with monitoring, logging, and performance optimization.

These are potential improvements, not claims about existing functionality.

## Author

**Archit Saki**

Computer Engineering Student | AI/ML and Generative AI Developer

GitHub: [ArchitSaki](https://github.com/ArchitSaki)

---

*Built as an AI-powered data analysis project to explore natural-language analytics, LLM integration, SQL execution, and modular backend engineering.*
