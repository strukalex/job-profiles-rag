import asyncio
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

# Define the factors
FACTORS = [
    "factor1","factor2"
]

class FactorClassifier:
    def __init__(self):
        self.llm = AzureAIChatCompletionsModel(
            endpoint=os.getenv('AZURE_ENDPOINT'),
            credential=os.getenv('AZURE_API_KEY'),
            model_name="Mistral-small",
            api_version="2024-05-01-preview",
            # model_kwargs={"max_tokens": 1000},
            # temperature=0.2
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

            Output format:
            {{
                "level": "[level letter]",
                "justification": "[detailed comparison with level definitions, including why it exceeds lower level and doesn't meet higher level]"
            }}

            Profile to evaluate:

            {profile}
            """
        )

        self.factor_descriptions=self.load_factor_descriptions()
        
        self.chains = {factor: self.create_chain(factor) for factor in FACTORS}

    def create_chain(self, factor):
        return (
            RunnablePassthrough.assign(factor=lambda _: factor)
            | self.prompt_template
            | self.llm
            | self.parser
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

_FACTOR_CLASSIFIER = FactorClassifier()

def calculate_total_points(results):
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
    
    total = 0
    for factor, result in results.items():
        if factor in factor_weights:
            total += factor_weights[factor][result['level']]
    return total

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
    for factor, data in results.items():
        factor_name = factor_names.get(factor, factor)
        formatted_results.append(
            f"• {factor_name}:\n"
            f"  Level: {data['level']}\n"
            f"  Justification: {data['justification']}\n"
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
    
    profile = query
    # factor_descriptions = load_factor_descriptions()  # Implement this function to load factor descriptions
    # benchmarks = load_benchmarks()  # Implement this function to load benchmarks
    
    results = await _FACTOR_CLASSIFIER.classify_all_factors(profile)
    
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
                "content": f"""Job Profile Classification Summary

                            📊 Factor Evaluations:
                            {format_factor_results(results)}

                            📈 Total Points: {total_points}
                            🎯 Final Classification: {grid_level}

                            This classification was determined by:
                            1. Evaluating each of the 13 factors individually
                            2. Converting factor levels to points and calculating the total ({total_points} points)
                            3. Matching the total points to the appropriate grid level range

                            The profile falls into {grid_level} based on the total point calculation."""
            },
            "finish_reason": "stop"
        }],
    }