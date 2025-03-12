import calendar
import json
import time
import matplotlib.pyplot as plt
import io
import numpy as np
import os
import pandas as pd
from fastapi.responses import FileResponse, JSONResponse
import uuid
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

from config.definitions import ROOT_DIR
from .azure_client import get_langchain_azure_model


class HelpProvider:
    def __init__(self):
        
        self.llm = get_langchain_azure_model(
            model_name="Mistral-small",
            api_version="2024-05-01-preview",
            model_kwargs={
                "max_tokens": 8000,
                # "stop": ["\n", "###"]  # Add natural stopping points
            }
        )
        
        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant answering user's questions from context:
             Accessing Job Store
 
How do I access Job Store?
If you're a people leader, there is a new tile on your Manager Self-Service dashboard that will launch Job Store. Updated links to the new Job Store are also available on Careers & MyHR.
 
What access do I have?
Excluded people leaders can create new positions in their department ID. Included people leaders can request access to create positions with approval from their ministry HR lead. Follow your internal ministry process for this. All employees have access to explore the job profiles on Job Store.

Creating new positions in Job Store
 
Have Classification and Exclusion Services changed?
No. The BC Public Service Agency's Classification and Exclusion Services continue to provide classification, exclusion and organizational design expertise and services. The job classification process has not changed. The process for creating a new position using pre-approved and pre-classified job profiles has been significantly streamlined through digital self-service using the new Job Store.
 
Have the classification frameworks changed?
No. The classification frameworks such as Public Service Job Evaluation Plan (PSJEP) and Management Classification and Compensation Framework (PDF, 287 KB) are the same.

 
Do I still require ministry approval to hire?
Yes. The current process for seeking ministry approvals is not changing. You must continue following your ministry processes and/or contact your ministry HR.

 
Can I get an immediate position number for all classifications of positions?
The process for creating a position number for included positions is the same regardless of classification. If the selected job profile and any edits made to the profile do not require verification (see below for the definition of 'verification required') by Classification and Exclusion Services, the position number will be provided immediately.
If the profile requires verification or changes are made to a job profile that may affect the classification, the position is submitted for verification and follow the classification process, which may take a few days to a few weeks if a full classification review is required.

 
What job profiles can I use in the application?
Job Store is pre-populated with pre-approved generic job profiles that are common across the BC Public Service or a specific ministry. The application then further filters the job profiles by the classification levels appropriate for the reporting relationship. The positions shown when you're creating a new position are only the ones pre-approved, generic and meeting the reporting requirements.
Job Store populates the relevant job profiles reporting to the people leader selected. You can't use job profiles for positions or job types that don't currently report to the people leader selected.

 
Are all job profiles in Job Store available for automatic position number creation?
Many job profiles lead to the automatic creation of a position number if the criteria is met in the context section. The context section of the job profile ensures the organizational structure and any qualifying conditions are understood and matched. You must review the context section of the job profile to ensure the organizational structure and any qualifying conditions are met.
Some job profiles need verification before a position number is created. Verification may be required because of changes to significant job accountabilities.
Some profiles always require verification due to the nature of the role. For example, some positions have limitations required by legislation and require a review by Classification Services to ensure legislation is upheld. Excluded positions also require union approval and require a process outside of Job Store.

 
Creating Schedule A positions from Job Store
Job Store profiles can be used for union and Schedule A (statutorily excluded) positions. There are some exceptions for Schedule A positions requiring verification by Classification and Exclusion Services. Classification and Exclusion Services verify it is appropriate to create the requested position using a Schedule A job code if the work unit is a statutorily excluded work unit per the Public Service Labour Relations Act (PSLRA). This does not apply to excluded management positions (band level positions), which must follow the exclusion process prior to being created.
Position requests can be submitted to AskMyHR through Job Store with available profiles for excluded roles. However, these still require an exclusion viability review and potentially subsequent classification review (if any amendments are made to the profile or the contextual requirements are not met). For more information, please read the Exclusion Review Process (PDF, 137KB).

 
Can I get a position number for excluded management positions right away?
No. Any new excluded positions must go through an initial exclusion viability review and subsequent classification approval first (if required). Classification and Exclusion Services will follow up with you for the required information needed to complete an exclusion review.

Verification
 
What does 'verification required' mean?
There are two scenarios where verification by the Classification and Exclusion Services team may be required before creating a new position using a Job Store profile:

Scenario 1: You have made edits to accountabilities that may impact classification. Alert pop-ups in Job Store will let you know when you are making significant edits. In this scenario, Classification and Exclusion Services verify whether edits impact classification and if a classification review is required
Scenario 2: Select profiles in Job Store always require verification to ensure the predetermined context (a set of mandatory criteria to be able to use a profile) is met
 
Do I need to create a separate service request through AskMyHR if my request requires verification by Classification and Exclusion Services?
No. When you submit your request for verification, the system automatically creates an AskMyHR service request, and you will receive a confirmation email. For excluded management positions only, your AskMyHR submission will be automatically split into two requests: One for exclusion and one for classification. You will be provided with separate request numbers for both.
 
Does Job Store automatically send a service request to Classification and Exclusion Services?
Yes. When you submit your position request and verification is required, it automatically creates an AskMyHR service request. You will also receive a copy of your request via email.

 
Does Job Store create position numbers?
Yes. Job Store is fully integrated with PeopleSoft and can create position numbers when you click 'generate'.

 
Can I create a net new position using a job profile that isn't in Job Store?
No. The application is for pre-approved and pre-classified job profiles only. If you want to duplicate a position you already have, please submit an AskMyHR service request using the category Job Classification > Create a New Position.

Job profiles
 
Have the job profiles been updated?
Yes. All job profiles on Job Store have been updated to reflect more inclusive language and a more concise approach to accountabilities. Equivalencies for minimum job requirements and Indigenous Relations Behavioural Competencies have been added to all job profiles.

 
What does the context mean?
The context indicates specific criteria required to meet the classification and use the job profile, such as reporting relationships, scope of work and location of the position. You should always review the context prior to using the job profile to ensure the position meets these qualifying criteria. If you are not sure, please contact Classification and Exclusion Services through AskMyHR.

 
How is the classification framework being used in Job Store?
All profiles in Job Store have been classified based on existing classification frameworks such as Public Service Job Evaluation Plan and Management Classification and Compensation Framework (PDF, 287KB).

 
What if I don't see the job profile I want on Job Store?
Please contact Classification and Exclusion Services via AskMyHR using the category Job Classification > Create a New Position). They may:

Know of other job profiles under development but not yet published that may help you
Point you to an existing job profile that is similar and can be used
Be able to help with edits to make a profile suitable
Work with you to have a custom job profile written and classified
 
What if I want to change an accountability after I create a position number?
Please submit a service request to Classification and Exclusion Services via AskMyHR, providing details of the changes required. To expedite processing, please include an up-to-date organization chart of the work unit.

Position management
 
My organization chart generated through Job Store system is wrong, what do I do?
Your organization chart visualizes the real-time position data in PeopleSoft. If you need to update your organization chart, please complete the Position Management Form (IDIR restricted).
If you receive an error message indicating you aren't authorized to access these forms, please contact your ministry representative (XLS, 54KB).

 
It says Job Store is for new positions only. What do I do if I want to reclassify a position?
Please submit a service request for a classification review to Classification and Exclusion Services via AskMyHR (Job Classification > Classification Review). The reclassification process remains the same. Visit Classification Review Process on Careers & MyHR for more information.

 
What does 'positions may be audited' mean?
Positions automatically created through Job Store may be selected at random for audit. The purpose of auditing is to ensure the specified context for each job profile is met. During the audit, you may be asked for work examples for the topic position.

 
What may happen if an audit determines the position does not meet the criteria? What risks are there?
If a position does not meet the criteria outlined, this could result in:

The requirement for a full classification review of the position
Potential downward classification for incumbents of the topic position or other related positions, that may reclassification when the position becomes vacant
Incorrect classification has impacts on internal pay equity and can impact team morale
Miscellaneous
 
How do I proceed with hiring/recruitment using a position created through Job Store?
The hiring request process remains the same. Submit the Hiring Request Form with the position number and with a final copy of the job profile from Job Store. To download your final copy:

In Job Store, open a 'Completed' position request
Navigate to the 'Actions' tab
Click on the 'Download' link in the 'Approved job profile' row to download the Word document
 
Can I download the organization chart?
Yes. To download the organization chart, click on the 'Download' button on the right side of the chart to download a PNG or SVG file.

 
Can I download job profiles?
Currently, Job Store only supports downloading job profiles after the position number has been generated in Job Store. The steps to do that are given below:

In Job Store, open a 'Completed' position request
Navigate to the 'Actions' tab
Click on the 'Download' link in the 'Approved job profile' row to download the Word document
 
Can I create multiple positions at once or duplicate an existing position in Job Store?
No. Currently, you can only create one position at a time.

Can I start a new organizational chart (for example, for a new branch) by creating positions?
No. If you need to start a new organization chart, please contact Classification and Exclusion Services via Classification and Exclusion Services via AskMyHR.

            Current query: {query}"""
        )
 
        
        self.chain = (
             self.prompt
            | self.llm
        )


_HELP_PROVIDER = HelpProvider()

async def handle_provide_help(
    query: str,
    model: str = "Mistral-small",
    temperature: float = 0.7,
    max_tokens: int = 300
) -> dict:
    # # Generate plot code through LangChain
    response = await _HELP_PROVIDER.chain.ainvoke({"query": query})
    
    if isinstance(response, JSONResponse):
        return response

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": f"{response.content}"
            },
            "finish_reason": "stop"
        }],
    }