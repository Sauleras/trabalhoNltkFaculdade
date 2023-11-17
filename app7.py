import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
from transformers import pipeline

# Baixar recursos do NLTK
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(texto):
    palavras = word_tokenize(texto.lower())
    stopwords_lista = set(stopwords.words('portuguese'))
    palavras_filtradas = [palavra for palavra in palavras if palavra.isalnum() and palavra not in stopwords_lista]
    return palavras_filtradas

def extrair_texto_do_pdf(caminho_pdf):
    texto_pdf = ""
    with fitz.open(caminho_pdf) as pdf_document:
        for pagina_numero in range(pdf_document.page_count):
            pagina = pdf_document[pagina_numero]
            texto_pdf += pagina.get_text()

    return texto_pdf

# Caminho para o arquivo PDF
caminho_pdf = 'texto.pdf'
segundo_caminho_pdf = 'texto2.pdf'
terceira_caminho_pdf = 'texto3.pdf'

# Extrair texto do PDF
texto_do_pdf = extrair_texto_do_pdf(caminho_pdf)
texto_segundo_documento = extrair_texto_do_pdf(segundo_caminho_pdf)
texto_terceiro_documento = extrair_texto_do_pdf(terceira_caminho_pdf)

# Criar corpus com base no texto extraído do PDF
corpus = {
    'documento1': texto_do_pdf,
    'documento2': texto_segundo_documento,
    'documento3': texto_terceiro_documento
}

# Criar vetorizador TF-IDF
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(corpus.values())
nomes_features = vetorizador.get_feature_names_out()

def vetorizar_pergunta(pergunta):
    pergunta_processada = preprocess_text(pergunta)
    vetor_pergunta = vetorizador.transform([' '.join(pergunta_processada)])
    return vetor_pergunta

def obter_pontuacoes_similaridade(vetor_pergunta, matriz_tfidf):
    pontuacoes_similaridade = cosine_similarity(vetor_pergunta, matriz_tfidf)
    return pontuacoes_similaridade

def recuperar_documento_mais_relevante(pontuacoes_similaridade, corpus):
    indice_mais_similar = pontuacoes_similaridade.argmax()
    documento_mais_similar = list(corpus.keys())[indice_mais_similar]
    return documento_mais_similar

def gerar_resposta_pergunta(pergunta, texto_do_documento):
    question_answering_pipeline = pipeline("question-answering", model="neuralmind/bert-base-portuguese-cased", tokenizer="neuralmind/bert-base-portuguese-cased")
    resposta = question_answering_pipeline(context=texto_do_documento, question=pergunta)
    return resposta['answer']

# Testar o sistema com uma pergunta
pergunta = input("Faça uma pergunta: ")
vetor_pergunta = vetorizar_pergunta(pergunta)
pontuacoes_similaridade = obter_pontuacoes_similaridade(vetor_pergunta, matriz_tfidf)
documento_mais_similar = recuperar_documento_mais_relevante(pontuacoes_similaridade, corpus)

# Obter texto do documento mais similar
texto_do_documento_mais_similar = corpus[documento_mais_similar]

# Gerar resposta para a pergunta
resposta = gerar_resposta_pergunta(pergunta, texto_do_documento_mais_similar)

print(pergunta)
print(f"A resposta para a pergunta é: {resposta}")
