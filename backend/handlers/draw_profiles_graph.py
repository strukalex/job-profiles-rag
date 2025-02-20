import calendar
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
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

class ChartGenerator:
    def __init__(self, csv_path):
        DATA_DIR = ROOT_DIR / "data"
        fpath = DATA_DIR /csv_path
        df=pd.read_csv(fpath)
        df = df.drop(columns=["accountabilities", "knowledge_skills_abilities", "job_experience","education",
                              "professional_registration_requirements","preferences","willingness_statements",
                              "security_screenings","behavioural_competencies","is_archived","context","overview","optional_requirements",
                              "version","program_overview","state"])

        # Identify columns containing JSON data
        json_columns = ['role', 'role_type', 'classifications', 'organizations', 
                        'scopes', 'job_families', 'streams', 'reports_to']

        # Track which columns contain lists
        list_columns = set()

        def format_json_string(x,column_name):
            if not isinstance(x, str):
                return x
            try:
                data = json.loads(x)
                if isinstance(data, list):
                    list_columns.add(column_name)  # Track columns that contain lists
                    return ', '.join([f"{item['name']}" for item in data])
                elif isinstance(data, dict):
                    return f"{data['name']}"
                return x
            except json.JSONDecodeError:
                return x

        # Apply the transformation in one step
        df[json_columns] = df[json_columns].apply(lambda col: col.apply(lambda x: format_json_string(x, col.name)))

        # Convert only the columns we know contained lists
        for column in list_columns:
            df[column] = df[column].str.split(', ')

        # Parse dates
        df['valid_from'] = pd.to_datetime(df['valid_from'], errors='coerce')
        df['valid_to'] = pd.to_datetime(df['valid_to'], errors='coerce')
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')

        sampleRow=df.iloc[0].copy()
        # Modify the sample row's comma-separated fields
        sampleRow['classifications'] = ["classification1", "classification2", "classification3"]
        sampleRow['organizations'] = ["organization1", "organization2", "organization3"]
        sampleRow['scopes'] = ["scope1", "scope2", "scope3"]
        sampleRow['job_families'] = ["jobfamily1", "jobfamily2", "jobfamily3"]
        sampleRow['streams'] = ["stream1", "stream2", "stream3"]
        sampleRow['reports_to'] = ["manager1", "manager2", "manager3"]


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
            - Variable name: df
            - Key Columns:
            * Categorical: state, type, role, role_type
            * Temporal: created_at, updated_at, published_at
            * Numerical: views
            * List-like: classifications, organizations, scopes, job_families, streams
            * Metadata: id, number
            
            Sample Data (first row):
            {sample_row}
            
            Requirements:
            1. Use views as primary metric for aggregation
            2. Include detailed comments about the thought process
            3. Do not include any adjustable parameters
            4. Do not set xticks/yticks range
            5. Keep code simple
            6. Follow examples closely
            
            Example 1. Response for "Show views by role type":
            ```
            # Group data by role_type and sum views
            role_type_views = df.groupby('role_type')['views'].sum()

            # Create horizontal bar plot
            role_type_views.plot(kind='barh')

            # Add title and labels
            plt.title('Views by Role Type')
            plt.xlabel('Total Views')
            plt.ylabel('Role Type')

            # Ensure layout is properly spaced
            plt.tight_layout()
            ```

            Example 2. Response for "Show top 5 organizations by total views":
            ```
            # Expand organizations list into separate rows
            orgs = df.explode('organizations')

            # Sum views for each organization
            org_views = orgs.groupby('organizations')['views'].sum()

            # Get the top 5 organizations by view count
            top_orgs = org_views.nlargest(5)

            # Create horizontal bar chart
            top_orgs.plot(kind='barh')
            plt.title('Top 5 Organizations by Total Views')
            plt.xlabel('Total Views')
            plt.tight_layout()
            ```

            Other documentation:
            - Create pie chart:
            ```
            plt.pie(org_counts, labels=org_counts.index)
            ```
            

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
            'sns': sns,
            'df': self.df,
            'calendar': calendar,
            '__builtins__': __builtins__  # This gives access to all built-ins
        }
        
        exec(code, globals_dict)


# Initialize with your DataFrame

_CHART_GENERATOR = ChartGenerator("job profiles/2025-02-07_profiles.csv")

async def handle_draw_profile_graph(
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