import cv2
import face_recognition
import os
import numpy as np
from tqdm import tqdm
from deepface import DeepFace

def load_images_from_folder(folder):
    known_face_encodings = []
    known_face_names = []

    # Percorrer todos os arquivos na pasta fornecida
    for filename in os.listdir(folder):
        # Verificar se o arquivo é uma imagem
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Carregar a imagem
            image_path = os.path.join(folder, filename)
            image = face_recognition.load_image_file(image_path)
            # Obter as codificações faciais (assumindo uma face por imagem)
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                face_encoding = face_encodings[0]
                # Extrair o nome do arquivo, removendo o sufixo numérico e a extensão
                name = os.path.splitext(filename)[0][:-1]
                # Adicionar a codificação e o nome às listas
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)

    return known_face_encodings, known_face_names

def detect_faces_and_emotions(video_path, output_path, known_face_encodings, known_face_names):
    # Capturar vídeo do arquivo especificado
    cap = cv2.VideoCapture(video_path)

    # Verificar se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obter propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Loop para processar cada frame do vídeo com barra de progresso
    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        # Ler um frame do vídeo
        ret, frame = cap.read()

        # Se não conseguiu ler o frame (final do vídeo), sair do loop
        if not ret:
            break

        # Analisar o frame para detectar faces e expressões
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Obter as localizações e codificações das faces conhecidas no frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Inicializar uma lista de nomes para as faces detectadas
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Desconhecido"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

        # Iterar sobre cada face detectada pelo DeepFace
        for face in result:
            # Obter a caixa delimitadora da face
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            
            # Obter a emoção dominante
            dominant_emotion = face['dominant_emotion']

            # Desenhar um retângulo ao redor da face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Escrever a emoção dominante acima da face
            cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Associar a face detectada pelo DeepFace com as faces conhecidas
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                if x <= left <= x + w and y <= top <= y + h:
                    # Escrever o nome abaixo da face
                    cv2.putText(frame, name, (x + 6, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    break
        
        # Escrever o frame processado no vídeo de saída
        out.write(frame)

    # Liberar a captura de vídeo e fechar todas as janelas
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Caminho para a pasta de imagens com rostos conhecidos
image_folder = 'images'

# Carregar imagens e codificações
known_face_encodings, known_face_names = load_images_from_folder(image_folder)

# Caminho para o arquivo de vídeo na mesma pasta do script
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'video.mp4')  # Substitua 'meu_video.mp4' pelo nome do seu vídeo
output_video_path = os.path.join(script_dir, 'output_video_recognize.mp4')  # Nome do vídeo de saída

# Chamar a função para detectar emoções e reconhecer faces no vídeo, salvando o vídeo processado
detect_faces_and_emotions(input_video_path, output_video_path, known_face_encodings, known_face_names)