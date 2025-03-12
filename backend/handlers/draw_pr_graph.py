import calendar
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np
import os
import pandas as pd
import datetime
from fastapi.responses import FileResponse, JSONResponse
import uuid
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

from config.definitions import ROOT_DIR
from .azure_client import get_langchain_azure_model

# Configure image storage
API_HOST = os.getenv("API_HOST", "http://localhost:8000")
IMAGES_DIR = Path("backend/static/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

class PRChartGenerator:
    def __init__(self, csv_path):
        DATA_DIR = ROOT_DIR / "data"
        fpath = DATA_DIR /csv_path
        df=pd.read_csv(fpath)
        df=df[df['status'] != 'DRAFT']
        df = df.drop(columns=["id","step", "reports_to_position_id", "parent_job_profile_id","orgchart_json",
                              "user_id","title","position_number",
                              "submission_id","classification_id","submitted_at","updated_at","crm_assigned_to_account_id","crm_id",
                              "crm_json","shareUUID","additional_info","profile_json","crm_lookup_name","max_step_completed","parent_job_profile_version",
                              "excluded_manager_position","reports_to_position","time_to_approve","resubmitted_at","unknownStateSince","unknownStateMetadata",
                              "user_name","user_email","department_name","classification_name","location_name","job_profile_number",
                              "job_profile_overview","user_username","department_id","classification_peoplesoft_id"])
        # Parse dates
        df['approved_at'] = pd.to_datetime(df['approved_at'], errors='coerce')

        # rename columns to make them easier to parse
        df = df.rename(columns={
            'department_code': 'department',
            'organization_name': 'ministry'
        })
        
        sampleRow=df.iloc[19].copy()

        self.df = df

        self.llm = get_langchain_azure_model(
            model_name="Mistral-small",
            api_version="2024-05-01-preview",
            temperature=0.5,
            top_p=0.4,
            model_kwargs={
                "max_tokens": 4000,
                # "stop": ["\n", "###"]  # Add natural stopping points
            }
        )
        
        self.parser = JsonOutputParser()

        # 2. Use pd.to_datetime() for temporal columns
        # 4. Prefer horizontal bar charts for role comparisons
        # 5. Include proper date formatting for time-based plots
        # 6. Handle NaN values in optional_requirements with .dropna()
        self.prompt = ChatPromptTemplate.from_template(
            """Generate matplotlib visualization code for: {query}

DataFrame Context:
Variable name: df

Key Columns:
Categorical: status, classification_employee_group_id, approval_type, classification_code, classification_grade, employee_group_name, department, ministry, job_profile_title
Temporal: approved_at

Sample Data (first row):
{sample_row}

Requirements:
1. Use count() as primary metric for aggregation
2. Include detailed comments about the thought process
3. Do not include any adjustable parameters
4. Do not set xticks/yticks range
5. Keep code simple
6. Follow examples closely

Example 1. Response for "Show approval status distribution":

```
# Group data by status and count occurrences
status_counts = df['status'].value_counts()

# Create pie chart
plt.figure(figsize=(10, 6))
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Approval Status')
plt.tight_layout()
```

Example 2. Response for "Show top 5 job profiles":

```
# Count occurrences of each job profile
job_counts = df['job_profile_title'].value_counts().head(5)

# Create horizontal bar chart
plt.figure(figsize=(10, 6))
job_counts.plot(kind='barh')
plt.title('Top 5 Job Profiles')
plt.xlabel('Number of Records')
plt.ylabel('Job Profile')
plt.tight_layout()
```

Example 3. Response for "Show approvals over time":

```
#Group by date and count approvals
daily_approvals = df.groupby(df['approved_at'].dt.date).size()

#Create line plot
plt.figure(figsize=(12, 6))
plt.plot(daily_approvals.index, daily_approvals.values)
plt.title('Approval Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Approvals')
plt.xticks(rotation=45)
plt.tight_layout()
```

Example 4. Response for "Show distribution by employee group":

```
# Group by employee group and count
group_counts = df['employee_group_name'].value_counts()

# Create vertical bar chart with color gradient
colors = plt.cm.viridis(np.linspace(0, 1, len(group_counts)))
plt.figure(figsize=(10, 6))
group_counts.plot(kind='bar', color=colors)
plt.title('Distribution by Employee Group')
plt.xlabel('Employee Group')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
```

Example 5. Response for "Show classification grade distribution":

```
# Count records by classification grade
grade_counts = df['classification_grade'].value_counts()

# Create donut chart
plt.figure(figsize=(10, 6))
plt.pie(grade_counts, labels=grade_counts.index, autopct='%1.1f%%', pctdistance=0.85)
plt.title('Classification Grade Distribution')

# Create donut hole
center_circle = plt.Circle((0,0), 0.70, fc='white')
plt.gca().add_artist(center_circle)
plt.tight_layout()
```

Example 6. Response for "Show department code vs organization name":

```
# Create cross-tabulation
dept_org_counts = pd.crosstab(df['department_code'], df['organization_name'])

# Create stacked bar chart
plt.figure(figsize=(12, 6))
dept_org_counts.plot(kind='bar', stacked=True)
plt.title('Department Code Distribution by Organization')
plt.xlabel('Department Code')
plt.ylabel('Count')
plt.legend(title='Organization', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45)
plt.tight_layout()
```

Example 7. Response for "Show position request with completed status approval trends by month with a line chart":

```
# Convert approved_at to datetime and extract month
df['approved_at'] = pd.to_datetime(df['approved_at'], unit='ms')
df['month'] = df['approved_at'].dt.to_period('M')

# Filter for completed status
completed_requests = df[df['status'] == 'COMPLETED']

# Group by month and count completed requests
monthly_completed = completed_requests.groupby('month').size()

# Convert PeriodIndex to timestamps for plotting
plot_index = monthly_completed.index.to_timestamp()

# Create line chart
plt.figure(figsize=(12, 6))
plt.plot(plot_index, monthly_completed.values)
plt.title('Completed Position Request Approval Trends by Month')
plt.xlabel('Month')
plt.ylabel('Number of Completed Requests')
plt.xticks(rotation=45)
plt.tight_layout()
```

Notes about data:
- approval_type is one of these: ['VERIFIED', 'AUTOMATIC', 'REVIEWED', nan]. Only use it if asked about "approval type"
- status is one of these: ['VERIFICATION', 'COMPLETED', 'CANCELLED', 'ACTION_REQUIRED']
- classification_employee_group_id (Classification employee group) is one of these: ['GEU', 'MGT', 'PEA', nan]
- ONLY USE FROM AVAILABLE COLUMNS: ['approved_at', 'status', 'classification_employee_group_id', 'approval_type', 'classification_code', 'classification_grade', 'employee_group_name', 'department', 'ministry', 'job_profile_title']

Generate ONLY the Python code with these requirements.
Current query: {query}"""
        )
 
        sample_row_str=str(sampleRow.to_frame().T.to_json(
            orient='records',
            indent=2
        ))
        self.chain = (
            RunnablePassthrough.assign(
                columns=lambda _: list(self.df.columns),
                # sample_row=lambda _: sampleRow.to_frame().T.to_csv(index=False)
                sample_row=lambda _: sample_row_str
            )
            | self.prompt
            | self.llm
        )
        print('initialized draw_profiles_graph')
        # self.chain.invoke({"query":"make a graph of position requests"})

    def execute(self, code: str):
        globals_dict = {
            'plt': plt,
            'pd': pd,
            'np': np,
            'sns': sns,
            'datetime': datetime,
            'df': self.df,
            'calendar': calendar,
            '__builtins__': __builtins__  # This gives access to all built-ins
        }
        
        exec(code, globals_dict)


# Initialize with your DataFrame

_CHART_GENERATOR = PRChartGenerator("job profiles/2025-02-07_position_requests.csv")

async def handle_draw_pr_graph(
    query: str,
    model: str = "Mistral-small",
    temperature: float = 0.7,
    max_tokens: int = 300
) -> dict:
    # The temperature parameter is already set in the PRChartGenerator class initialization
    # and doesn't need to be passed to the chain.invoke method as it's part of the model configuration
    generated_code = await _CHART_GENERATOR.chain.ainvoke({"query": query})
    
    if isinstance(generated_code, JSONResponse):
        return generated_code
    
    # Extract code block
    if "```python" in generated_code.content:
        code_parts = generated_code.content.split("```python")
        if len(code_parts) > 1:
            code_block = code_parts[1].split("```")[0].strip()
    elif "```" in generated_code.content:
        code_parts = generated_code.content.split("```")
        if len(code_parts) > 1:
            code_block = code_parts[1].strip()
    else:
        code_block=generated_code.content
    
    # Create fresh figure and execute code
    plt.figure()
    print('executing code block: ')
    print(code_block)

    _CHART_GENERATOR.execute(code_block)

    # Generate filename and save
    filename = f"plot_{uuid.uuid4().hex[:8]}.png"
    filepath = IMAGES_DIR / filename
    plt.savefig(filepath, format='png', bbox_inches='tight')
    plt.close()
    
    # Cleanup old files
    files = sorted(IMAGES_DIR.glob("*.png"), key=os.path.getctime)
    if len(files) > 20:
        for file in files[:-20]:
            file.unlink()
    
    image_url = f"{API_HOST}/static/images/{filename}"
    
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": f"Generated plot:\n\n![Generated plot]({image_url})"
            },
            "finish_reason": "stop"
        }]
    }