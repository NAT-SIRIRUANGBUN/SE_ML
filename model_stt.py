import whisper
import time
import re
from jiwer import cer

def calculate_eng_preservation(ground_truth, hypothesis):
    """ฟังก์ชันนับว่าคำศัพท์ภาษาอังกฤษในเฉลย โดนถอดออกมาถูกต้องกี่เปอร์เซ็นต์"""
    # ดึงคำภาษาอังกฤษทั้งหมดออกมา (ตัวเล็กทั้งหมดเพื่อเทียบง่าย)
    gt_eng_words = re.findall(r'[A-Za-z]+', ground_truth.lower())
    hyp_eng_words = re.findall(r'[A-Za-z]+', hypothesis.lower())
    
    if not gt_eng_words:
        return 100.0 # ถ้าประโยคนั้นไม่มี Eng เลย ถือว่า 100%
        
    # นับคำที่ตรงกัน
    preserved_count = sum(1 for word in gt_eng_words if word in hyp_eng_words)
    return (preserved_count / len(gt_eng_words)) * 100

# 1. โหลดโมเดล (ใช้ small ไปก่อนเพื่อความรวดเร็ว)
print("Loading Whisper model...")
model = whisper.load_model("small")

# 2. เตรียมชุดข้อมูลทดสอบ (แก้ชื่อไฟล์เสียงและเฉลยให้ตรงกับที่คุณอัดจริง)
test_data = [
    {
        "audio": "Chulalongkorn University.m4a",
        "ground_truth": ""
    }
]

# 3. กำหนด Prompt รวมคำศัพท์เทคนิค
my_prompt = "Model, Ensemble, Accuracy, Push, Code, Github, Pipeline, Data"

print("\n🚀 เริ่มการทดสอบ Whisper Prompting...")
for item in test_data:
    print(f"\n--- กำลังทดสอบไฟล์: {item['audio']} ---")
    
    start_time = time.time()
    
    # รันถอดเสียง
    result = model.transcribe(
        item["audio"],
        language="th",
        initial_prompt=my_prompt
    )
    
    # คำนวณเวลาและ Metrics
    process_time = time.time() - start_time
    hypothesis = result["text"]
    
    error_rate = cer(item["ground_truth"], hypothesis)
    eng_rate = calculate_eng_preservation(item["ground_truth"], hypothesis)
    
    print(f"✅ Ground Truth: {item['ground_truth']}")
    print(f"🤖 ผลลัพธ์โมเดล:  {hypothesis}")
    print(f"⏱️ เวลาประมวลผล:   {process_time:.2f} วินาที")
    print(f"📊 CER (ตัวอักษรผิด): {error_rate:.2f} (ยิ่งน้อยยิ่งดี)")
    print(f"🔤 อัตราคงคำ Eng:  {eng_rate:.2f}% (ยิ่งมากยิ่งดี)")