from faster_whisper import WhisperModel
from app.lib.OPENAI import OpenAIClient


class WhisperClient:
    """Wraps OpenAI Whisper for STT and Thai→English cleanup."""

    def __init__(self, model: str = 'large'):
        model_name = "large-v3" if model == "large" else model
        self.speech_model = WhisperModel(model_name, device="auto", compute_type="default")
        self.openai = OpenAIClient()

    def speech_to_text(self, file_path: str, initial_prompt: str = None, language: str = 'th', progress_callback=None) -> dict:
        """Run Whisper transcription and return the raw result dict."""
        segments_generator, info = self.speech_model.transcribe(
            file_path,
            word_timestamps=True,
            initial_prompt=initial_prompt,
            language=language
        )
        
        result_dict = {"text": "", "segments": []}
        
        for segment in segments_generator:
            if progress_callback and info.duration > 0:
                percent = min(99, int((segment.end / info.duration) * 100))
                progress_callback(percent)
                
            result_dict["text"] += segment.text + " "
            
            words_list = []
            if segment.words:
                for w in segment.words:
                    words_list.append({
                        "word": w.word,
                        "probability": w.probability
                    })
                    
            result_dict["segments"].append({
                "id": segment.id,
                "text": segment.text,
                "words": words_list
            })
            
        result_dict["text"] = result_dict["text"].strip()
        return result_dict

    def thai_to_english(self, data: dict) -> str:
        """Post-process Whisper output: convert Thai transliterations back to English words."""
        segment_str = ""
        for segment in data['segments']:
            segment_str += f"id: {segment['id']}, text:{segment['text']}\n"
            for word in segment['words']:
                segment_str += f"{word['word']}, prob: {round(word['probability'], 2)}\n"

        prompt = f"""
นี่คือผลลัพธ์การทำ speech to text จาก model (whisper model) ซึ่งในที่นี้อาจจะมีคำทับศัพท์ภาษาอังกฤษอยู่บ้าง ให้นำคำทับศัพท์เหล่านั้นเปลี่ยนกลับเป็นคำภาษาอังกฤษ โดยเป็นตัวพิมพ์เล็กทั้งหมด
ตอบกลับมาเพียงประโยคที่ให้ไปที่แก้ไขคำทับศัพท์เป็นภาษาอังกฤษแล้ว โดยไม่ต้องแปลคำภาษาไทยเป็นอังกฤษ แค่เปลี่ยนคำที่มีคำสะกดภาษาอังกฤษ เป็นภาษาอังกฤษ ถ้าไม่มีเลยก็คืนค่าคำเดิมกลับมาได้
ไม่ต้องคืนมาใน format อื่น และไม่มีการใช้ "\\n" หรือตัวอักษรพิเศษใด ๆ ในคำตอบนอกเหนือจากประโยคที่แก้ไข
ไม่มีพวก id: อะไรพวกนี้ แค่ประโยค

ข้อมูล:
{segment_str}
"""
        return self.openai.ask_plain_text(prompt)

    def run_full(self, file_path: str, initial_prompt: str = None, language: str = 'th') -> str:
        """Full pipeline: transcribe → Thai-to-English cleanup → return final text."""
        data = self.speech_to_text(file_path, initial_prompt, language)
        return self.thai_to_english(data)

Whisper = WhisperClient