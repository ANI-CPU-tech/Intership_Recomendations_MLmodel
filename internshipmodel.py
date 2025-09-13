import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

url='interships/internships.csv'

data=pd.read_csv(url)


data["Job Skills"]=data[["Skill 1","Skill 2","Skill 3","Skill 4","Skill 5","Skill 6"]].fillna("").agg(" ".join, axis=1) 
data["Job Location"]=data[["Location"]].fillna("")
data["Job Perks"]=data[["Perk 1","Perk 2","Perk 3"]].fillna("").agg(" ".join, axis=1)
data["Job Data"]=data["Job Skills"]+ " " + data["Job Location"]+ " " + data["Job Perks"]

features=data["Job Data"]
target=[["Company","Job Title"]]
print(features)

vectorizer=TfidfVectorizer()
TfMatrix=vectorizer.fit_transform(features)
print(TfMatrix)

def job_recommendations(user_skills, user_location="",user_perks="", top=5):
    # Combine user skills and optional location
    user_profile = user_skills + " " + user_location + " " + user_perks
    user_vec = vectorizer.transform([user_profile])
    
    # Compute cosine similarity
    similarities = cosine_similarity(user_vec, TfMatrix).flatten()
    
    # Get top matches
    top_indices = similarities.argsort()[-top:][::-1]
    
    # Return recommended internships
    return data.iloc[top_indices][["Company", "Job Title", "Location"]]

# User input
skill_no = int(input("Enter the number of skills you have: "))
skills = [input(f"Enter skill {i+1}: ") for i in range(skill_no)]
skills_str = " ".join(skills)

perk_no = int(input("Enter the number of perks you have: "))
perks = [input(f"Enter skill {i+1}: ") for i in range(perk_no)]
perks_str = " ".join(perks)

location = input("Enter preferred location : ")

# Get recommendations
best_recommendations = job_recommendations(skills_str, location,perks_str)
print(best_recommendations)