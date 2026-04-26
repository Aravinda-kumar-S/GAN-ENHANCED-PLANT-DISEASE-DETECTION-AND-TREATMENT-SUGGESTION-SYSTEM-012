from google import genai
import os
import streamlit as st

def troubleshooting():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)
        
        with open("model_list.txt", "w") as out:
            try:
                for m in client.models.list():
                    out.write(f"{m.name}\n")
            except Exception as e:
                out.write(f"Error: {e}")
    except Exception as e:
        print(f"Failed to load secrets: {e}")

if __name__ == "__main__":
    troubleshooting()
