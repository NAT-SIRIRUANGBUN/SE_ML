import speech_recognition as sr
import pyttsx3

# initialize the recognizer
r = sr.Recognizer()

# ENHANCEMENT 1: Tune the recognizer settings
r.energy_threshold = 4000  # Increase if picking up too much background noise
r.dynamic_energy_threshold = True  # Adapts to changing noise levels
r.pause_threshold = 0.8  # Seconds of silence before phrase is considered complete
r.phrase_threshold = 0.3  # Minimum seconds of speaking audio before phrase
r.non_speaking_duration = 0.5  # Seconds of silence to mark end of phrase

def record():
    # loop in case of errors
    while(1):
        try:
            # use mic
            with sr.Microphone() as source:
                
                # prepare recognizer to receive input
                r.adjust_for_ambient_noise(source, duration=1.0)
                
                # listen for audio
                print("Listening...")
                audio = r.listen(source)
                
                # convert to text
                text = r.recognize_google(audio, language="en-US")
                print("You said: " + text)
                return text
                
                # try:
                #     # try English first
                #     text = r.recognize_google(audio, language="en-US")
                #     print("You said: " + text)
                #     return text
                
                # except:
                #     # then Thai if Eng fails
                #     text = r.recognize_google(audio, language='th-TH')
                #     print("คุณพูดว่า: " + text)
                #     return text
                
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
            
        except sr.UnknownValueError:
            print("Unknown Value Error. Please try again.")
    
    return 

def output(text):
    # open text
    # a = new texts will be appended to the file
    try:
        with open("output.txt", "a", encoding="utf-8") as f:
            f.write(text + "\n")
        return True
    except Exception as e:
        print(f"Error writing to file: {e}")
        return False

while(1):
    
    # record audio and convert to text
    text = record()
    
    # output text to file
    # output(text)
    # print("Text written to output.txt")