from docx import Document
from transformers import pipeline
import os

# Inicializar o pipeline de sumarização
summarizer = pipeline("summarization")

def read_docx(docx_path):
    """
    Lê o texto de um documento .docx.
    :param docx_path: Caminho para o documento .docx
    :return: Texto completo do documento
    """
    document = Document(docx_path)
    full_text = []
    for para in document.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

def summarize_text(text, max_length=130, min_length=30, do_sample=False):
    """
    Função para sumarizar um texto.
    :param text: Texto a ser sumarizado
    :param max_length: Comprimento máximo do resumo
    :param min_length: Comprimento mínimo do resumo
    :param do_sample: Se True, usar amostragem; se False, usar truncagem
    :return: Resumo do texto
    """
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
    return summary[0]['summary_text']

def save_summary_to_txt(summary_text, txt_path):
    """
    Salva o resumo em um arquivo .txt.
    :param summary_text: Texto do resumo
    :param txt_path: Caminho para salvar o arquivo .txt
    """
    with open(txt_path, 'w', encoding='utf-8') as file:
        file.write(summary_text)

if __name__ == "__main__":
    # Caminho para o documento .docx
    docx_path = 'documento.docx'  # Arquivo .docx
    txt_path = 'resumo.txt'  # Nome do arquivo de saída .txt

    # Ler o texto completo do documento
    full_text = read_docx(docx_path)
    
    # Sumarizar o texto completo
    summary = summarize_text(full_text, max_length=200, min_length=50)
    
    # Salvar o resumo em um arquivo .txt
    save_summary_to_txt(summary, txt_path)
    
    print("Sumarização completa. O resumo foi salvo em 'resumo.txt'.")