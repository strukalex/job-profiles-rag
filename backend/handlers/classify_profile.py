import asyncio
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

# Define the factors
FACTORS = [
    "factor1","factor2","factor3","factor4","factor5","factor6","factor7","factor8","factor9","factor10","factor11","factor12","factor13"
]

class ClassificationProvider:
    def __init__(self):
        self.llm = get_langchain_azure_model(
            model_name="Mistral-small",
            api_version="2024-05-01-preview",
            model_kwargs={
                "max_tokens": 8000,
                # "stop": ["\n", "###"]  # Add natural stopping points
            }
        )
        
        self.parser = JsonOutputParser()
        
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are an expert job classification analyst specializing in evaluating position profiles for public sector organizations.
Your task is to classify the following job profile for the factor into one of the levels (A, B, C, etc.). 
You must explicitly compare the profile against the level definitions, explaining why it fits the chosen level and why it doesn't fit the adjacent levels.

Factor description:
{factor_description}

Analyze the profile and provide your classification in json format only. Your justification must:

1. Cite specific language from the chosen level definition that matches the profile using quotations
2. Explain why the profile exceeds the requirements of the level below, using quotations from the level definition
3. Explain why the profile does not meet the requirements of the level above, using quotations from the level definition
4. Reference specific examples from the profile to support your reasoning and quotations from the level definition
5. Use quotations from the level definition above as much as possible to provide justifications
6. Provide quotations from level definitions above for primary designation BUT ALSO FOR LEVELS ABOVE AND BELOW
7. DO NOT include new line characters in output, output justification as a single string. Do not format json in any way. Generate standard json only

Output format:
{{"level": "level letter","justification": "detailed comparison with level definitions, including why it exceeds lower level and doesn't meet higher level"}}

Example output:
{{"level" : "D","justification" : "The Full Stack Developer profile fits Level D as it requires..."}}

Profile to evaluate:

{profile}
"""
        )

        self.factor_descriptions=self.load_factor_descriptions()
        
        self.chains = {factor: self.create_chain(factor) for factor in FACTORS}

    def create_chain(self, factor):
        def handle_llm_response(response):
            if isinstance(response, JSONResponse):
                return response
            return self.parser.invoke(response)
        
        return (
            RunnablePassthrough.assign(factor=lambda _: factor)
            | self.prompt_template
            | self.llm
            | handle_llm_response
        )

    async def classify_factor(self, factor, profile, factor_description):
        result = await self.chains[factor].ainvoke({
            "profile": profile,
            "factor_description": factor_description
        })
        return {factor: result}

    async def classify_all_factors(self, profile):
        
        tasks = [
            self.classify_factor(factor, profile, self.factor_descriptions[factor])
            for factor in FACTORS
        ]
        results = await asyncio.gather(*tasks)

        # Return first JSONResponse found, if any
        for result in results:
            for factor_result in result.values():
                if isinstance(factor_result, JSONResponse):
                    return factor_result
                
        return {k: v for d in results for k, v in d.items()}
    
    def load_factor_descriptions(self):
        factor_descriptions = {}
        path = Path("data/classification")
        for factor in FACTORS:
            file_path = os.path.join(path, f'{factor}.txt')
            with open(file_path, 'r') as file:
                factor_descriptions[factor] = file.read().strip()
        return factor_descriptions

# Initialize with your DataFrame

_FACTOR_CLASSIFIER = ClassificationProvider()

factor_weights = {
        'factor1': {'A': 20, 'B': 40, 'C': 60, 'D': 100, 'E': 145, 'F': 190, 'G': 250, 'H': 280, 'I': 305, 'J': 330},
        'factor2': {'A': 20, 'B': 40, 'C': 60, 'D': 100, 'E': 150, 'F': 175, 'G': 200, 'H': 250, 'I': 300},
        'factor3': {'A': 10, 'B': 20, 'C': 30, 'D': 45, 'E': 60, 'F': 75},
        'factor4': {'A': 5, 'B': 10, 'C': 15, 'D': 22.5, 'E': 33, 'F': 43},
        'factor5': {'A': 15, 'B': 30, 'C': 50, 'D': 75, 'E': 120, 'F': 160, 'G': 190, 'H': 220},
        'factor6': {'A': 5, 'B': 10, 'C': 15, 'D': 22.5, 'E': 33, 'F': 43, 'G': 58, 'H': 73},
        'factor7': {'A': 5, 'B': 10, 'C': 15, 'D': 22.5, 'E': 33, 'F': 43},
        'factor8': {'A': 5, 'B': 9, 'C': 13, 'CD':14, 'CE':15, 'CF':16.5, 'CG':19, 'CH':21, 'D': 19, 'DE': 20, 'DF': 21, 'DG': 23, 'DH': 25, 'DI':27},
        'factor9': {'A': 5, 'B': 10, 'C': 15, 'D': 25, 'E': 40, 'F': 50},
        'factor10': {'A': 3, 'B': 6, 'C': 12, 'D': 18, 'E': 24, 'F': 30},
        'factor11': {'A': 3, 'B': 6, 'C': 12, 'D': 18, 'E': 24, 'F': 30},
        'factor12': {'A': 2, 'B': 4, 'C': 6, 'D': 9, 'E': 12},
        'factor13': {'A': 2, 'B': 4, 'C': 6, 'D': 9, 'E': 12}
    }

def calculate_total_points(results):
    try:
        total = 0
        for factor, result in results.items():
            if factor in factor_weights:
                total += factor_weights[factor][result['level']]
        return total
    except Exception as e:
        raise e

def determine_grid_level(points):
    point_bands = {
        (100, 189): 'Grid Level 06',
        (190, 279): 'Grid Level 07',
        (280, 369): 'Grid Level 09',
        (370, 459): 'Grid Level 11',
        (460, 544): 'Grid Level 13',
        (545, 624): 'Grid Level 14',
        (625, 714): 'Grid Level 18',
        (715, 804): 'Grid Level 21',
        (805, 864): 'Grid Level 24',
        (865, 924): 'Grid Level 27',
        (925, 1044): 'Grid Level 30'
    }
    
    for (min_points, max_points), level in point_bands.items():
        if min_points <= points <= max_points:
            return level
    
    return 'Grid Level 32' if points > 1044 else 'Grid Level 06'

def format_factor_results(results):
    factor_names = {
        'factor1': 'Knowledge',
        'factor2': 'Mental Demands',
        'factor3': 'Interpersonal Communication',
        'factor4': 'Physical Coordination/Dexterity',
        'factor5': 'Work Assignments',
        'factor6': 'Financial Responsibility',
        'factor7': 'Asset/Information Responsibility',
        'factor8': 'Human Resources Responsibility',
        'factor9': 'Safety Responsibility',
        'factor10': 'Sensory/Multiple Demands',
        'factor11': 'Physical Effort',
        'factor12': 'Surroundings',
        'factor13': 'Hazards'
    }
    
    formatted_results = []
    i=0
    for factor, data in results.items():
        i+=1
        factor_name = factor_names.get(factor, factor)
        points = factor_weights[factor][data['level']]
        formatted_results.append(
            f"### {str(i)}. {factor_name}\n"
            f"- **Level:** {data['level']} ({points} points)\n"
            f"- **Justification:** {data['justification']}\n"
        )
    
    return "\n".join(formatted_results)

async def handle_classify_profile(
    query: str,
    model: str = "Mistral-small",
    temperature: float = 0.7,
    max_tokens: int = 300
) -> dict:
    # todo - extract the profile
    # profile = extract_profile(query)  # Implement this function to extract the profile from the query
    
    if("sonnet" not in query):
        profile = query
        # factor_descriptions = load_factor_descriptions()  # Implement this function to load factor descriptions
        # benchmarks = load_benchmarks()  # Implement this function to load benchmarks
        
        results = await _FACTOR_CLASSIFIER.classify_all_factors(profile)

        if isinstance(results, JSONResponse):
            return results
    else:

        # results = {
        #     'factor1': {'level': 'A', 'justification': 'implementation details'},
        #     'factor2': {'level': 'A', 'justification': 'implementation details'},
        #     'factor3': {'level': 'A', 'justification': 'implementation details'}
        # }

        results = {
            'factor1': {"level": "H", "justification": 'The Full Stack Developer profile aligns with Level H as it requires understanding "the theory of a trade, craft, operational, professional or technical area(s) with the requirement to plan, research and review complex issues with many factors to consider." The role exceeds Level G requirements which only focus on "understanding principles" and "analyzing, diagnosing, and interpreting standards" - this position goes beyond by leading development teams, advising executives, and managing complex multi-year projects. The profile demonstrates Level H through activities like "plans and executes multiple simultaneous systems development projects" and "advises executive on business or organizational issues." The position requires advanced technical knowledge to "develop front-end and back-end enterprise solutions" and "designs and implements data warehouse architecture." It doesn\'t reach Level I as it doesn\'t involve "understanding all related issues of a significant program which requires considerable resources and provides services to a large part of a ministry" - instead focusing on specific technical implementations and solutions rather than ministry-wide program management. While the role advises executives, it doesn\'t have the scope of Level I\'s requirement to "plan or develop policy or provide authoritative advice for the program" at a ministry-wide level.'},
            'factor2': {"level": "G", "justification": "The Full Stack Developer position aligns with Degree G as it requires 'judgement required to modify methods, techniques or approaches so they will work with new or changed circumstances or objectives to plan a course of action.' This is evidenced by the role's responsibility to develop and enhance computer systems, similar to the benchmark example 'develop computer systems enhancements (BM021).' The position exceeds Degree F requirements which focus on 'performing precise review and manipulation of sophisticated data' as this role goes beyond just developing programs and integrating platforms - it requires modifying and adapting approaches to meet changing business needs. The profile shows this through responsibilities like 'developing new concepts,' 'designing and implementing data warehouse architecture,' and 'developing front-end and back-end enterprise solutions.' The position does not meet Degree H requirements which focus on 'evaluating effectiveness of policies, programs or services and develop proposals for improvements' or 'managing policies, programs or services.' While the role advises executives and leads development projects, it does not have the broad program evaluation and management responsibilities described in Degree H. The technical nature of the work, requirement to modify approaches for new circumstances, and system enhancement responsibilities clearly align with Degree G's focus on modifying methods and techniques."},
            'factor3': {"level": "D", "justification": "The Full Stack Developer position aligns with Level D which requires 'persuasion required to use basic counselling skills to induce cooperation, interview for information or provide advice, or use basic negotiating skills to reach a settlement where the parties are generally cooperative and have common interests.' This is evidenced by the profile requirements to 'advise executive and senior management on alternatives and solutions,' and 'communicate technical concepts to a non-technical audience to gain consensus.' The position exceeds Level C which only requires 'discretion required to exchange information needing explanation' as the role involves leading teams, advising executives, and negotiating with stakeholders. The position does not meet Level E requirements for 'influence required to use formal counselling or formal negotiating skills in dealing with sensitive issues or agreements where the parties are not cooperative or do not have common interests' as the profile indicates collaborative work environments and generally cooperative stakeholders, shown in requirements like 'collaborates with other teams' and 'ensures client requirements and priorities are understood.'"},
            'factor4': {"level": "A", "justification": "The Full Stack Developer position requires only basic coordination and dexterity as it primarily involves computer-based work. The role focuses on software development, system design, and project management activities that require minimal physical coordination beyond basic keyboard and mouse usage. The position's core duties - coding, reviewing others' work, advising executives, and managing projects - all require minimal physical demands. While technical skills are extensive, the physical coordination requirements do not exceed Degree A's 'basic coordination and dexterity.' The role does not meet Degree B's requirements for 'some coordination and dexterity' as it does not involve any specialized physical skills or manual techniques beyond standard office computer use. The job duties are primarily cognitive and analytical in nature, with physical demands limited to standard computer interface operations."},
            'factor5': {"level": "G", "justification": "The Full Stack Developer position aligns with Comparative Effects Level V and Freedom to Act Level 6, resulting in Level G classification. The role 'affects the strategic direction, or the decision-making of senior ministry executives, for a significant ministry program, project, or system' as evidenced by advising executive on business issues, establishing strategic plans, and managing mission-critical database projects. The Freedom to Act matches Level 6 where 'work is guided by general policies, plans, guidelines or standards and requires planning, organizing or evaluating projects/cases' as shown by planning multiple systems development projects and leading professional teams. This exceeds Level F requirements which only involves 'considerable' impact and managing technical programs, while this role has significant ministry-wide impact through strategic direction and executive advisory. The position does not reach Level H as it does not demonstrate Level VI 'major' impacts affecting 'decisions made by senior ministry or government executives on major issues' or affecting 'the strategic direction of a major ministry program' - while influential, the role's scope remains within significant rather than major ministry programs."},
            'factor6': {"level": "E", "justification": "The Full Stack Developer position demonstrates significant financial responsibility through several key aspects of the role. The profile shows the position 'advises executive on business or organizational issues and collaboratively establishes strategic plans and budgets' and 'determines need for contract resources, develops contract specifications, estimates costs.' These responsibilities exceed Degree D (moderate responsibility) as they involve direct input into budgetary decisions and contract specifications. The role also 'advises executive and senior management on alternatives and solutions, product evaluation, risk assessment, and cost benefit analysis of existing and future applications' which demonstrates significant financial accountability through recommendations that impact organizational resources. While the position has significant financial responsibility through budgeting and contract activities, it does not reach Degree F (considerable responsibility) as it does not have primary accountability for major departmental budgets or final authority over large-scale financial commitments. The position makes recommendations and provides input but works collaboratively rather than having sole financial authority. The role's financial impact comes through its technical leadership and advisory capacity rather than direct control of major financial resources."},
            'factor7': {"level": "E", "justification": "The Full Stack Developer position demonstrates 'considerable responsibility' for information assets that exceeds Level D's 'significant responsibility' and aligns with Level E. The role manages mission-critical database development projects, designs and implements data warehouse architecture, and handles complex data models. They lead multiple simultaneous systems development projects and are responsible for enterprise-wide solutions that affect multiple ministries. The position exceeds Level D as they not only handle information systems but have strategic influence through advising executives on alternatives, solutions, and risk assessments for existing and future applications. They also determine development tools and database configurations, showing deeper control over information assets. However, it does not reach Level F's 'major responsibility' as the role focuses on specific technical domains rather than having ultimate authority over all information assets. The position maintains considerable but not complete control over information systems, working within established frameworks rather than setting organization-wide information management policies. Key evidence includes their responsibility for 'managing multi-year mission-critical database development projects,' 'determining appropriate development tools and database configurations,' and 'designing and implementing data warehouse architecture.' These duties show considerable but not ultimate responsibility for information assets."},
            'factor8': {"level": "C", "justification": "The Full Stack Developer profile best aligns with Level C as it requires the position to 'assign, monitor and examine work of assigned workers for accuracy and quality, usually as a group leader, project team leader.' This is evidenced by the profile stating 'Leads a team of professionals, defining work assignments, and verifying and reviewing code produced by others.' The profile exceeds Level B which only requires 'provide formal instruction or training to other workers' as this role has broader team leadership responsibilities beyond just training. However, it does not fully meet Level D requirements which specify 'supervise assigned employees directly or through subordinate supervisors and appraise employee performance or take disciplinary action.' While the role leads and reviews work, there is no mention of formal performance appraisals or disciplinary responsibilities. The profile specifically mentions 'verifying and reviewing code' and 'defining work assignments' which directly matches Level C's requirement to 'assign, monitor and examine work.' Based on the profile's description of leading a team of professionals, this would likely fall into the CD category (greater than 1 FTE and up to 5 FTEs)."},
            'factor9': {"level": "B", "justification": "The Full Stack Developer role demonstrates 'limited care and attention for the well-being or safety of others' which aligns with Level B. While the position leads a team and collaborates with others, the primary focus is on technical development rather than direct responsibility for others' well-being or safety. The role exceeds Level A's 'minimal care' requirement through team leadership responsibilities and collaboration with diverse stakeholders. However, it does not reach Level C's 'moderate care and attention' as the position's impact on others' well-being is indirect and limited to standard professional interactions. Key evidence includes: leading a team of professionals, collaborating with other teams, and communicating with non-technical audiences. The technical nature of the work and focus on system development rather than human safety or well-being supports this classification."},
            'factor10': {"level": "D", "justification": "The Full Stack Developer profile demonstrates an intense requirement (Level D) for sensory effort and multiple demands. The profile shows frequent high-intensity activities that align with Column D ratings, particularly in multiple areas: 1) Visual focus requirements are intense and sustained, matching item #28 'On page or screen to scrutinize documents, reports, databases, etc.' at the D level, as the role requires constant development and review of code. 2) Multiple demands are at the highest level, with the profile showing the need to 'manage concurrent projects' and 'prepare response by a critical deadline with little advance notice' (items #35 and #38), which are rated at Column D when frequent. The role exceeds Level C (Focused requirement) as it involves leading multiple simultaneous development projects while managing team members and dealing with executive-level stakeholders, requiring intense concentration and multiple demand management beyond just focused attention. It does not fit Level B (Close requirement) as the responsibilities far exceed occasional or regular monitoring, requiring almost constant high-level sensory engagement. The profile specifically requires managing 'multiple, simultaneous systems development projects,' 'leading a team of professionals,' and 'executing repeatable automated processes' while maintaining communication with various stakeholders, demonstrating the intense level of multiple demands characteristic of Level D."},
            'factor11': {"level": "B", "justification": "The Full Stack Developer position best aligns with Degree B (Light physical effort) based on the physical demands described in the profile. The role primarily involves sitting at a computer with frequent visual focus on screens, keyboarding, and occasional standing or walking. The position requires 'focusing visual attention to computer screens' and 'keyboarding without speed requirement' which falls into Column B when performed frequently. This exceeds Degree A (Relatively light physical effort) which only accommodates occasional to regular keyboarding and screen focus, but does not meet the requirements for Degree C (Moderate physical effort) as it lacks any significant physical demands like regular lifting, climbing, or working in awkward positions. The primary physical activities are limited to desk work, with the main physical demands being prolonged sitting, keyboarding, and screen viewing - activities that characterize light physical effort without substantial muscular exertion."},
            'factor12': {"level": "A", "justification": "The Full Stack Developer position best aligns with Level A as it primarily involves working in a normal office setting with typical office conditions. The profile describes development work that takes place in a standard office environment, with no mention of significant exposure to disagreeable elements or challenging social interactions. While the role involves team leadership and collaboration, these interactions are professional in nature and do not rise to the level of 'unpleasant dealings with upset, angry, demanding or unpredictable people' that would warrant a higher rating. The position does not require exposure to elements like dust, noise, chemicals, or outdoor conditions that would qualify for Level B or above. The role exceeds the minimum requirements for Level A through its professional office setting but does not meet the criteria for Level B which would require regular exposure to crowded conditions, public work-sites, or disagreeable social interactions. The profile shows no evidence of physical discomfort, exposure to environmental elements, or challenging client interactions that would justify a higher classification."},
            'factor13': {"level": "B", "justification": "The Full Stack Developer position aligns with Level B (Limited exposure) based on the hazards present in the role. The primary hazards involve frequent keyboarding/repetitive motion (Column B: F) and occasional office hazards (Column A). The role involves primarily computer-based work developing web applications and databases, with extensive keyboard use for coding and development tasks. This exceeds Level A (Minimal exposure) which only covers occasional to regular keyboarding, but does not reach Level C (Moderate exposure) as it lacks significant physical hazards or exposure to dangerous conditions. The position does not involve working with hazardous materials, at heights, or in dangerous environments that would warrant a higher rating. The main physical risks are ergonomic in nature from prolonged computer use and standard office environment hazards."},
        }
   
    # results['factor1']['justification']
    # print('result: ', results)

    total_points = calculate_total_points(results)  # Implement this function to calculate total points
    grid_level = determine_grid_level(total_points)  # Implement this function to determine the final grid level
    
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": f"""# Job Profile Classification Summary

## Overview
- **Total Points:** {total_points}
- **Final Classification:** {grid_level}

## Factor Evaluations
{format_factor_results(results)}

## Determination Process
1. Evaluated each of the 13 factors individually
2. Converted factor levels to points and calculated total: {total_points} points
3. Matched total points to appropriate grid level range

**Final Result:** The profile falls into {grid_level} based on the total point calculation."""
        },
        "finish_reason": "stop"
    }],
}