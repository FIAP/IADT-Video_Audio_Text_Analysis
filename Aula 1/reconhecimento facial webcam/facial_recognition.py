import cv2
import face_recognition
import os
import numpy as np

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

def main():
    image_folder = 'images'  # Caminho para a pasta de imagens
    known_face_encodings, known_face_names = load_images_from_folder(image_folder)  # Carregar imagens e codificações

    video_capture = cv2.VideoCapture(0)  # Iniciar captura de vídeo da webcam

    while True:
        ret, frame = video_capture.read()  # Capturar um único frame de vídeo
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Redimensionar o frame para 1/4 do tamanho
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])  # Converter BGR para RGB

        face_locations = face_recognition.face_locations(rgb_small_frame)  # Localizar faces no frame
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)  # Obter codificações faciais

        face_names = []  # Lista para armazenar os nomes das faces detectadas
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)  # Verificar se a face é conhecida
            name = "Desconhecido"  # Nome padrão se a face não for reconhecida
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)  # Calcular a distância para faces conhecidas
            best_match_index = np.argmin(face_distances)  # Encontrar o índice da melhor correspondência
            if matches[best_match_index]:  # Verificar se a melhor correspondência é uma face conhecida
                name = known_face_names[best_match_index]  # Obter o nome da face conhecida
            face_names.append(name)  # Adicionar o nome à lista de nomes

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Redimensionar as coordenadas das faces de volta ao tamanho original
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Desenhar um retângulo ao redor da face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Desenhar uma etiqueta com o nome abaixo da face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Exibir a imagem resultante
        cv2.imshow('Video', frame)

        # Pressionar 'q' para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar a captura de vídeo e fechar todas as janelas
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()