import json
import pandas as pd
import os

def flatten_work_experiences(experiences):
    """Extract key information from work experiences list"""
    if not experiences:
        return "", ""
    
    companies = []
    roles = []
    
    for exp in experiences:
        companies.append(exp.get('company', ''))
        roles.append(exp.get('roleName', ''))
    
    return ",".join(companies), ",".join(roles)

def flatten_education(education):
    """Extract key information from education object"""
    if not education or not isinstance(education, dict):
        return "", "", "", ""
    
    highest_level = education.get('highest_level', '')
    
    degrees = education.get('degrees', [])
    schools = []
    subjects = []
    gpas = []
    
    for degree in degrees:
        schools.append(degree.get('originalSchool', ''))
        subjects.append(degree.get('subject', ''))
        # Only add GPA if it exists
        if degree.get('gpa'):
            gpas.append(f"{degree.get('originalSchool', 'Unknown')}: {degree.get('gpa', '')}")
    
    return highest_level, ",".join(schools), ",".join(subjects), ",".join(gpas)

def flatten_salary(salary_exp):
    """Extract salary information"""
    if not salary_exp:
        return ""
    
    return salary_exp.get('full-time', '')

def convert_json_to_csv(json_file, csv_file):
    """Convert JSON file to CSV"""
    # Read JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Create flattened data structure
    flattened_data = []
    
    for candidate in data:
        # Extract and flatten nested structures
        companies, roles = flatten_work_experiences(candidate.get('work_experiences', []))
        highest_edu, schools, subjects, gpas = flatten_education(candidate.get('education', {}))
        salary = flatten_salary(candidate.get('annual_salary_expectation', {}))
        
        # Create flattened record
        record = {
            'name': candidate.get('name', ''),
            'email': candidate.get('email', ''),
            'phone': candidate.get('phone', ''),
            'location': candidate.get('location', ''),
            'submitted_at': candidate.get('submitted_at', ''),
            'work_availability': ','.join(candidate.get('work_availability', [])),
            'annual_salary_expectation': salary,
            'companies': companies,
            'roles': roles,
            'highest_education': highest_edu,
            'schools': schools,
            'subjects': subjects,
            'gpas': gpas,
            'skills': ','.join(candidate.get('skills', []))
        }
        
        flattened_data.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(flattened_data)
    
    # Save to CSV
    df.to_csv(csv_file, index=False)
    print(f"Converted {json_file} to {csv_file}")
    print(f"Total candidates: {len(data)}")

if __name__ == "__main__":
    json_file = 'main_data.json'
    csv_file = 'candidates.csv'
    
    convert_json_to_csv(json_file, csv_file)
    
    # Display preview of CSV
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        print("\nPreview of CSV data:")
        print(df.head())
