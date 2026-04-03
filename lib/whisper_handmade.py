import whisper
import OPENAI

class Whisper:
    def __init__(self, model='medium'):
        self.speech_model = whisper.load_model(model)
        self.openai = OPENAI.OpenAI()

    def speech_to_text(self, file_path, initial_prompt = None, language='th'):
        result = self.speech_model.transcribe(
            audio= file_path,
            word_timestamps=True,
            initial_prompt = initial_prompt,
            language=language
        )
        
        return result

    def thai_to_english(self, data):

        
        segment_str = ""

        for segment in data['segments']:
            segment_str += f'id: {segment['id']}, text:{segment['text']}\n'
            for word in segment['words']:
                segment_str += f'{word['word']}, prob: {round(word['probability'], 2)}\n'

        prompt = \
f"""
นี่คือผลลัพธ์การทำ speech to text จาก model (whisper model) ซึ่งในที่นี้อาจจะมีคำทับศัพท์ภาษาอังกฤษอยู่บ้าง ให้นำคำทับศัพท์เหล่านั้นเปลี่ยนกลับเป็นคำภาษาอังกฤษ โดยเป็นตัวพิมพ์เล็กทั้งหมด
ตอบกลับมาเพียงประโยคที่ให้ไปที่แก้ไขคำทับศัพท์เป็นภาษาอังกฤษแล้ว โดยไม่ต้องแปลคำภาษาไทยเป็นอังกฤษ แค่เปลี่ยนคำที่มีคำสะกดภาษาอังกฤษ เป็นภาษาอังกฤษ ถ้าไม่มีเลยก็คืนค่าคำเดิมกลับมาได้
ไม่ต้องคืนมาใน format อื่น และไม่มีการใช้ "\ n" หรือตัวอักษรพิเศษใด ๆ ในคำตอบนอกเหนือจากประโยคที่แก้ไข 
บางครั้งการ transcribe มาจาก whisper model อาจเกิดความผิดพลาดทำให้สะกดบางคำผิดไป สามารถเดาคำที่ถูกต้องจาก context เองได้เลย
ไม่มีพวก id: อะไรพวกนี้ แค่ประโยค
ขอย้ำว่าอะไรเป็นคำทับศัพท์ภาษาอังกฤษ ให้แปลงเป็นคำภาษาอังกฤษให้หมด โดยไม่ต้องสนบริบทรอบข้าง เช่น ไอศครีม -> ice cream

ข้อมูล:
{segment_str}
"""
        
        res = self.openai.ask_plain_text(prompt)
        return res
    
    def run_full(self, file_path, initial_prompt = None, language='th'):
        data = self.speech_to_text(file_path, initial_prompt, language)
        return self.thai_to_english(data)