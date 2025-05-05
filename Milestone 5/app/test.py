import google.generativeai as genai
genai.configure(api_key="AIzaSyC-iX2JqmUkXOKqnIb4k1EDsUMx3AFqGb8")
models = genai.list_models()
for m in models:
    print(m.name, "=>", m.supported_generation_methods)
