import os
import pandas as pd
import google.generativeai as genai
import time
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Using Flash-Lite for the 4x larger daily free quota (1,000 requests/day)
model = genai.GenerativeModel('gemini-3.1-flash-lite-preview')

def restore_english_words(thai_text):
    prompt = f"""
    You are an expert linguist in Thai-English code-switching.
    Your task is to take a Thai sentence that contains transliterated English words (คำทับศัพท์)
    and convert ONLY the transliterated English words back to their original English spelling.
    
    Strict Rules:
    1. DO NOT translate native Thai words into English (this include โกวาจี).
    2. Leave the native Thai words exactly as they are.
    3. Preserve the original spacing.
    
    Example:
    เพิ่ม ใน เพลลิสต์ เพลง โปรด to เพิ่ม ใน playlist เพลง โปรด
    
    Input: {thai_text}
    """
    
    attempt = 1
    while True:
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "quota" in error_msg or "503" in error_msg or "500" in error_msg:
                wait_time = min(attempt * 5, 60) 
                print(f"\n[!] API busy/Rate limit. Pausing {wait_time}s before retrying... (Attempt {attempt})")
                time.sleep(wait_time)
                attempt += 1
            else:
                print(f"Non-retryable Error on: '{thai_text}'. Error: {e}")
                return thai_text 


# Load your Data
df = pd.read_csv("thai_foreign.csv")
print(f"Loaded {len(df)} rows. Starting text conversion...")

# Dictionary to remember previously translated sentences
translation_cache = {}
processed_texts = []

for index, row in df.iterrows():
    original_text = row['text']
    
    if original_text in translation_cache:
        cleaned_text = translation_cache[original_text]
        print(f"[{index}] SKIPPED API (Duplicate) -> Cleaned: {cleaned_text}")
        processed_texts.append(cleaned_text)
        
    else:
        cleaned_text = restore_english_words(original_text)
        
        translation_cache[original_text] = cleaned_text
        processed_texts.append(cleaned_text)
        
        print(f"[{index}] Original: {original_text}")
        print(f"[{index}] Cleaned:  {cleaned_text}")
        print("-" * 30)
        
        time.sleep(4.5) 

# Save the results
df['text_cleaned'] = processed_texts
df.to_csv("cleaned_dataset.csv", index=False, encoding='utf-8-sig')

print(f"Processing complete! Saved to 'cleaned_dataset.csv'.")
print(f"Unique sentences processed via API: {len(translation_cache)}")