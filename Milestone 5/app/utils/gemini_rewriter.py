# app/utils/gemini_rewriter.py

import google.generativeai as genai

def rewrite_query_with_gemini(prompt: str, api_key: str) -> str:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return prompt  # fallback to original
