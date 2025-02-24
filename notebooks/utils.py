import pandas as pd
import json
from typing import List
from langchain_core.documents import Document

import json

# PROFILE PER DOCUMENT
def get_job_profile_documents(csv_path: str) -> List[Document]:
    """Process job profiles CSV and create one document per job profile"""
    df = pd.read_csv(csv_path)
    documents = []

    for idx, row in df.iterrows():
        # Parse JSON fields
        # json_fields = ['role', 'role_type', 'scopes', 'classifications', 'organizations']
        parsed_data = {}
        
        # for field in json_fields:
        #     if pd.notna(row.get(field)):
        #         try:
        #             data = json.loads(row[field])
        #             if field in ['classifications', 'organizations']:
        #                 parsed_data[field] = ", ".join([f"{item['name']}" + (f" ({item['code']})" if field == 'organizations' else "") for item in data])
        #             elif field == 'scopes':
        #                 parsed_data[field] = ", ".join([item["name"] for item in data])
        #             else:
        #                 parsed_data[field] = data['name']
        #         except json.JSONDecodeError:
        #             parsed_data[field] = ""

        # Process classifications
        if pd.notna(row.get('classifications')):
            try:
                classifications_data = json.loads(row['classifications'])
                # Create a set to remove duplicates
                classification_names = set(item['name'] for item in classifications_data)
                parsed_data['classifications'] = ", ".join(sorted(classification_names))
            except json.JSONDecodeError:
                parsed_data['classifications'] = ""

        # Process organizations
        if pd.notna(row.get('organizations')):
            try:
                organizations_data = json.loads(row['organizations'])
                # Create a set of tuples (name, code) to remove duplicates
                org_items = set((item['name'], item['code']) for item in organizations_data)
                # Join with formatting
                parsed_data['organizations'] = ", ".join(
                    f"{name} ({code})" for name, code in sorted(org_items)
                )
            except json.JSONDecodeError:
                parsed_data['organizations'] = ""

        # Process other JSON fields (role, role_type, scopes)
        for field in ['role', 'role_type', 'scopes']:
            if pd.notna(row.get(field)):
                try:
                    data = json.loads(row[field])
                    if field == 'scopes':
                        scope_names = set(item["name"] for item in data)  # Remove duplicates
                        parsed_data[field] = ", ".join(sorted(scope_names))
                    else:
                        parsed_data[field] = data['name']
                except json.JSONDecodeError:
                    parsed_data[field] = ""

        # Create metadata
        metadata = {
            "title": row.get("title", ""),
            "number": row.get("number", ""),
            "type": row.get("type", ""),
            "context": row.get("context", ""),
            "views": row.get("views", ""),
            "role": parsed_data.get('role', ""),
            "role_type": parsed_data.get('role_type', ""),
            "scopes": parsed_data.get('scopes', ""),
            "classifications": parsed_data.get('classifications', ""),
            "organizations": parsed_data.get('organizations', ""),
            "created_at": row.get("created_at", ""),
            "updated_at": row.get("updated_at", ""),
            "row_index": idx,
        }

        # Build content sections
        content_sections = [
            f"Job Profile Title: {metadata['title']}",
            f"Classifications: {metadata['classifications']}",
            f"Organizations: {metadata['organizations']}"
        ]

        # Array fields to process
        array_fields = {
            "behavioural_competencies": "Behavioural Competencies",
            "education": "Education",
            "job_experience": "Job Experience",
            "professional_registration_requirements": "Professional Registration Requirements",
            "preferences": "Preferences",
            "knowledge_skills_abilities": "Knowledge, Skills, and Abilities",
            "willingness_statements": "Willingness Statements",
            "security_screenings": "Security Screenings",
            "accountabilities": "Accountabilities",
        }

        # Process each section
        for field, section_title in array_fields.items():
            if pd.notna(row.get(field)):
                try:
                    items = json.loads(row[field])
                    if(len(items)==0):
                        continue
                    content_sections.append(f'\n{row.get("title", "")} {section_title}:')
                    
                    if field == "behavioural_competencies":
                        
                        section_items = [f"• {item['name']}: {item['description']}" for item in items]
                    else:
                        section_items = [f"• {item['text']}" for item in items]
                    
                    content_sections.extend(section_items)
                except json.JSONDecodeError:
                    continue

        # Create one document with all content
        doc = Document(
            page_content="\n".join(content_sections),
            metadata=metadata
        )
        documents.append(doc)

    return documents

# THIS CREATES VECTORSTORE WITH DOCUMENTS PER SECTION
# Outputs like this:
# ('page_content', "Job Profile Title: Intake Support Clerk\nClassifications: Clerk R9, Clerk R9, ...\nOrganizations:
#  Min of Environment & Parks (ENV), Ministry of Infrastructure (INF), Mining and Critical Minerals (MCM), 
# Agriculture and Food (AGR), Attorney General (AG), ...\n\nSection: Professional Registration Requirements\n")
def get_job_profile_documents_per_section(csv_path: str) -> List[Document]:
    """Process job profiles CSV with pandas and JSON handling"""
    df = pd.read_csv(csv_path)
    documents = []

    for idx, row in df.iterrows():
        # Parse JSON fields if necessary
        role = json.loads(row['role']) if pd.notna(row.get('role')) else None
        role_type = json.loads(row['role_type']) if pd.notna(row.get('role_type')) else None
        scopes = json.loads(row['scopes']) if pd.notna(row.get('scopes')) else None

        # Base metadata for all documents
        base_metadata = {
            "title": row.get("title", ""),
            "number": row.get("number", ""),
            "type": row.get("type", ""),
            "context": row.get("context", ""),
            "views": row.get("views", ""),
            "role": f"{role['name']}" if role else "",
            "role_type": f"{role_type['name']}" if role_type else "",
            "scopes": ", ".join([item["name"] for item in scopes]) if scopes else "",
            "created_at": row.get("created_at", ""),
            "updated_at": row.get("updated_at", ""),
            "row_index": idx,
        }

        # Process classifications and organizations
        classifications = ""
        organizations = ""
        for field in ["classifications", "organizations"]:
            if pd.notna(row.get(field)):
                try:
                    items = json.loads(row[field])
                    if field=="classifications":
                        tags = [f"{item['name']}" for item in items]
                    else:
                        tags = [f"{item['name']} ({item['code']})" for item in items]
                    
                    base_metadata[field] = ", ".join(tags)
                    if field == "classifications":
                        classifications = ", ".join(tags)
                    else:
                        organizations = ", ".join(tags)
                except json.JSONDecodeError:
                    pass

        # Add a prefix to provide context for each document
        prefix_content = f"""Job Profile Title: {base_metadata['title']}
Classifications: {classifications}
Organizations: {organizations}
"""

        # Define array fields with semantic section titles
        array_fields = {
            "behavioural_competencies": ("Behavioural Competencies", ("name", "description")),
            "education": ("Education", ("text",)),
            "job_experience": ("Job Experience", ("text",)),
            "professional_registration_requirements": ("Professional Registration Requirements", ("text",)),
            "preferences": ("Preferences", ("text",)),
            "knowledge_skills_abilities": ("Knowledge, Skills, and Abilities", ("text",)),
            "willingness_statements": ("Willingness Statements", ("text",)),
            "security_screenings": ("Security Screenings", ("text",)),
            "accountabilities": ("Accountabilities", ("text",)),
        }

        # Process each array field
        for field, (section_title, attributes) in array_fields.items():
            if pd.notna(row.get(field)):
                try:
                    items = json.loads(row[field])
                    section_content = prefix_content + f"\nSection: {section_title}\n"
                    
                    # Collect all items for this section
                    if field == "behavioural_competencies":
                        section_items = [f"• {item['name']}: {item['description']}" for item in items]
                    else:
                        section_items = [f"• {item[attributes[0]]}" for item in items]
                    
                    # Join all items with newlines
                    section_content += "\n".join(section_items)
                    
                    # Create one document for the entire section
                    doc = Document(
                        page_content=section_content,
                        metadata={
                            **base_metadata,
                            "section": section_title
                        }
                    )
                    documents.append(doc)
                except json.JSONDecodeError as e:
                    print(f"Row {idx}: Invalid JSON in {field} - {str(e)}")

    return documents