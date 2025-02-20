import calendar
import json
import time
import matplotlib.pyplot as plt
import io
import numpy as np
import os
import pandas as pd
from fastapi.responses import FileResponse
import uuid
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

from config.definitions import ROOT_DIR

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
                              "job_profile_overview","user_username","department_id"])
        # Parse dates
        df['approved_at'] = pd.to_datetime(df['approved_at'], errors='coerce')

        sampleRow=df.iloc[19].copy()

        self.df = df

        self.llm = AzureAIChatCompletionsModel(
            endpoint=os.getenv('AZURE_ENDPOINT'),
            credential=os.getenv('AZURE_API_KEY'),
            model_name="Mistral-small",
            api_version="2024-05-01-preview",
            model_kwargs={"max_tokens": 4000},
            
            temperature=0.5,
            top_p=0.4
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
Categorical: status, classification_employee_group_id, classification_peoplesoft_id, approval_type, classification_code, classification_grade, employee_group_name, department_code, organization_name, job_profile_title
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

Group data by status and count occurrences
status_counts = df['status'].value_counts()

Create pie chart
plt.figure(figsize=(10, 6))
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Approval Status')
plt.tight_layout()

Example 2. Response for "Show top 5 job profiles":

Count occurrences of each job profile
job_counts = df['job_profile_title'].value_counts().head(5)

Create horizontal bar chart
plt.figure(figsize=(10, 6))
job_counts.plot(kind='barh')
plt.title('Top 5 Job Profiles')
plt.xlabel('Number of Records')
plt.ylabel('Job Profile')
plt.tight_layout()

Example 3. Response for "Show approvals over time":

Convert approved_at to datetime if needed
df['approved_at'] = pd.to_datetime(df['approved_at'])

Group by date and count approvals
daily_approvals = df.groupby(df['approved_at'].dt.date).size()

Create line plot
plt.figure(figsize=(12, 6))
plt.plot(daily_approvals.index, daily_approvals.values)
plt.title('Approval Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Approvals')
plt.xticks(rotation=45)
plt.tight_layout()

Example 4. Response for "Show distribution by employee group":

Group by employee group and count
group_counts = df['employee_group_name'].value_counts()

Create vertical bar chart with color gradient
colors = plt.cm.viridis(np.linspace(0, 1, len(group_counts)))
plt.figure(figsize=(10, 6))
group_counts.plot(kind='bar', color=colors)
plt.title('Distribution by Employee Group')
plt.xlabel('Employee Group')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()

Example 5. Response for "Show classification grade distribution":

Count records by classification grade
grade_counts = df['classification_grade'].value_counts()

Create donut chart
plt.figure(figsize=(10, 6))
plt.pie(grade_counts, labels=grade_counts.index, autopct='%1.1f%%', pctdistance=0.85)
plt.title('Classification Grade Distribution')

Create donut hole
center_circle = plt.Circle((0,0), 0.70, fc='white')
plt.gca().add_artist(center_circle)
plt.tight_layout()

Example 6. Response for "Show department code vs organization name":

Create cross-tabulation
dept_org_counts = pd.crosstab(df['department_code'], df['organization_name'])

Create stacked bar chart
plt.figure(figsize=(12, 6))
dept_org_counts.plot(kind='bar', stacked=True)
plt.title('Department Code Distribution by Organization')
plt.xlabel('Department Code')
plt.ylabel('Count')
plt.legend(title='Organization', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45)
plt.tight_layout()

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
        print('initialized draw_graph')
        # self.chain.invoke({"query":"make a graph of all profiles by classification"})

    def execute(self, code: str):
        globals_dict = {
            'plt': plt,
            'pd': pd,
            'np': np,
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
    # # Generate plot code through LangChain
    generated_code = await _CHART_GENERATOR.chain.ainvoke({"query": query})
    
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

    # code_block="# Split and stack organizations, then reset the index\norgs = df['organizations'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True)\norgs = orgs.to_frame('org')\n\n# Now join will work because both DataFrames share the same index\norg_views = df.join(orgs).groupby('org')['views'].sum()\n\n# Plot the results\norg_views.nlargest(5).plot(kind='barh')\nplt.title('Top 5 Organizations by Total Views')\nplt.xlabel('Total Views')\nplt.tight_layout()"
    # code_block="plt.figure(figsize=(10,6))\ndf.groupby('role_type')['views'].sum().sort_values().plot(kind='barh')\nplt.title('Total Views by Role Type')\nplt.xlabel('Total Views')\nplt.ylabel('Role Type')\nplt.tight_layout()"
    
    # Create fresh figure and execute code
    plt.figure()
    print('executing code block: ')
    print(code_block)



    # # Convert created_at to datetime and extract month
    # df['created_at'] = pd.to_datetime(df['created_at'], unit='ms')
    # df['month'] = df['created_at'].dt.month

    # # Group data by month and sum views
    # monthly_views = df.groupby('month')['views'].sum()

    # # Create horizontal bar plot
    # monthly_views.plot(kind='barh')

    # # Add title and labels
    # plt.title('Views by Month')
    # plt.xlabel('Total Views')
    # plt.ylabel('Month')

    # # Set xticks to show month names
    # plt.xticks(range(1, 13), calendar.month_name[1:])

    # # Ensure layout is properly spaced
    # plt.tight_layout()



    _CHART_GENERATOR.execute(code_block)

    
    # TRY
    # df=_CHART_GENERATOR.df
    # # Expand role_type column for better readability if required
    # df['role_type'] = df['role_type'].str.split(', ')

    # # Group by role_type and sum the views
    # views_by_role_type = df.groupby('role_type')['views'].sum()

    # # Plot the results as a horizontal bar chart
    # plt.figure(figsize=(10,6))
    # views_by_role_type.plot(kind='barh')

    # # Set titles and labels
    # plt.title('Total Views by Role Type')
    # plt.xlabel('Total Views')
    # plt.ylabel('Role Type')

    # # Ensure proper spacing
    # plt.tight_layout()
    # END TRY

    # if not success:
    #     return {
    #         "error": f"Code execution failed: {error}",
    #         "generated_code": code_block
    #     }
    
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
        }],
        # "usage": {
        #     "prompt_tokens": len(generated_code.split()),
        #     "completion_tokens": 0,
        #     "total_tokens": 0
        # }
    }