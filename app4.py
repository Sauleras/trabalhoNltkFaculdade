import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
import string
from unidecode import unidecode  # Importe a biblioteca unidecode

# Download os recursos necessários do NLTK (execute apenas uma vez)
nltk.download('punkt')
nltk.download('stopwords')

# Carregue o arquivo PDF
def load_pdf(pdf_path):
    pdf = PdfReader(open(pdf_path, "rb"))
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Pré-processamento do texto
def preprocess_text(text):
    # Tokenize em sentenças
    sentences = sent_tokenize(text, language='portuguese')  # Usando tokenização em português
    # Remova a pontuação
    cleaned_sentences = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence, language='portuguese')  # Usando tokenização em português
        words = [unidecode(word).lower() for word in words if word.isalnum()]  # Normalize e torne minúsculo
        cleaned_sentences.append(" ".join(words))
    return cleaned_sentences

# Faça uma pergunta
def answer_question(question, sentences):
    question_words = set(unidecode(question.lower()).split())  # Normalize e torne minúsculo
    best_sentence = None
    max_overlap = 0

    for sentence in sentences:
        sentence_words = set(sentence.split())
        overlap = len(question_words.intersection(sentence_words))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sentence = sentence

    return best_sentence

# Função principal
if __name__ == "__main__":
    pdf_path = "texto.pdf"
    pdf_text = load_pdf(pdf_path)
    preprocessed_sentences = preprocess_text(pdf_text)

    while True:
        user_question = input("Faça uma pergunta (ou digite 'sair' para sair): ")
        if user_question.lower() == 'sair':
            break
        answer = answer_question(user_question, preprocessed_sentences)
        print("Resposta:", answer)
