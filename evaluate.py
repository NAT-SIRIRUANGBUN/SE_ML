import pandas as pd
import re
import jiwer
from pythainlp.tokenize import word_tokenize

def clean_whisper(text):
    text = str(text)
    if 'text:' in text:
        after_text = text.split('text:')[1].strip()
        match = re.search(r'^(.*?)(?:\n|\s+(?:\w+|\S+),\s*prob:|$)', after_text)
        if match:
            return match.group(1).strip()
    return text.strip()

def calculate_ewa(truth_list, pred_list):
    total_eng_words = 0
    correct_eng_words = 0
    for t, p in zip(truth_list, pred_list):
        t_words = re.findall(r'[a-zA-Z]+', str(t).lower())
        p_words = re.findall(r'[a-zA-Z]+', str(p).lower())
        total_eng_words += len(t_words)
        for w in t_words:
            if w in p_words:
                correct_eng_words += 1
                p_words.remove(w)
    return (correct_eng_words / total_eng_words) * 100 if total_eng_words > 0 else 0

def calculate_wer(truth_list, pred_list):
    wer_scores = []
    for t, p in zip(truth_list, pred_list):
        t_tokens = " ".join(word_tokenize(str(t), engine='newmm'))
        p_tokens = " ".join(word_tokenize(str(p), engine='newmm'))
        
        if not t_tokens.strip():
            continue
            
        try:
            score = jiwer.wer(t_tokens, p_tokens)
            wer_scores.append(score)
        except ValueError:
            pass
            
    return (sum(wer_scores) / len(wer_scores)) * 100 if wer_scores else 0

def main():
    print("Loading datasets...")
    try:
        df_truth = pd.read_csv("Categorized-Data/Gowajee-Corpus/thai_foreign/final_data.csv")
        df_small = pd.read_csv("Gowajee-Thai-Foreign-Whisper-Small.csv", header=0, names=['pred'])
        df_large = pd.read_csv("Gowajee-Thai-Foreign-Whisper-Large.csv", header=0, names=['pred'])
        
        df_truth['small_pred'] = df_small['pred'].apply(clean_whisper)
        df_truth['large_pred'] = df_large['pred'].apply(clean_whisper)
        
        print(f"Loaded: Truth={len(df_truth)}, Small={len(df_small)}, Large={len(df_large)}")
        
        ewa_small = calculate_ewa(df_truth['text_cleaned'], df_truth['small_pred'])
        ewa_large = calculate_ewa(df_truth['text_cleaned'], df_truth['large_pred'])
        
        wer_small = calculate_wer(df_truth['text_cleaned'], df_truth['small_pred'])
        wer_large = calculate_wer(df_truth['text_cleaned'], df_truth['large_pred'])
        
        print("\n📊 สรุปผลการประเมินประสิทธิภาพ (Metrics)")
        print("============================================================")
        print(f"1. Whisper Large -> EWA: {ewa_large:.2f}% | WER: {wer_large:.2f}%")
        print(f"2. Whisper Small -> EWA: {ewa_small:.2f}% | WER: {wer_small:.2f}%")
        print("============================================================")
        print("* หมายเหตุ: EWA ยิ่งมากยิ่งดี / WER ยิ่งน้อยยิ่งดี\n")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. {e}")
        exit(1)

if __name__ == "__main__":
    main()
