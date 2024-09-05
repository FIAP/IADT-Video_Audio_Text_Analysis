import speech_recognition as sr
import os

def transcribe_audio_to_text(audio_path, text_output_path):
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)  # lê todo o áudio do arquivo
        
        try:
            # Usa o serviço de reconhecimento de fala do Google com configuração para português do Brasil
            text = recognizer.recognize_google(audio, language="pt-BR")
            print("Transcrição: " + text)
            
            # Salva a transcrição em um arquivo de texto
            with open(text_output_path, 'w', encoding='utf-8') as file:
                file.write(text)
                
        except sr.UnknownValueError:
            print("Google Speech Recognition não conseguiu entender o áudio")
        except sr.RequestError as e:
            print("Erro ao solicitar resultados do serviço de reconhecimento de fala do Google; {0}".format(e))

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(script_dir, 'audio1.wav')  # Entrada do audio em .wav
    text_output_path = os.path.join(script_dir, 'transcricao_audio.txt')

    transcribe_audio_to_text(audio_path, text_output_path)

if __name__ == "__main__":
    main()