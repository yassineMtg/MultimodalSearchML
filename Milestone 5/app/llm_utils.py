# app/llm_utils.py

import google.generativeai as genai

genai.configure(api_key="AIzaSyC-iX2JqmUkXOKqnIb4k1EDsUMx3AFqGb8")

def rewrite_query_gemini(prompt: str) -> str:
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content([
            f"""You are a search assistant. Convert the user query below into a short, concise list of search keywords only. 
DO NOT explain. DO NOT format in markdown. Just return a single short line of keywords separated by commas.

User query: "{prompt}"
Search keywords:"""
        ])
        return response.text.strip().split("\n")[0]  # only take first line
    except Exception as e:
        print("❌ Gemini API Error:", e)
        return prompt
