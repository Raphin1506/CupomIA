# Importações principais
from flask import Flask, request, jsonify  # Para criar a API e manipular requisições/respostas
import os  # Para manipulação de arquivos e diretórios
import cv2  # OpenCV para manipulação de imagens
import pytesseract  # OCR (leitura de texto em imagem)
from sklearn.feature_extraction.text import TfidfVectorizer  # Vetorização de texto
from sklearn.linear_model import LogisticRegression  # Modelo de classificação
from sklearn.pipeline import make_pipeline  # Facilita criação do pipeline com vetorizador + modelo
from werkzeug.utils import secure_filename  # Para salvar arquivos com nomes seguros
from PIL import Image  # Manipulação de imagens (opcional nesse código)
import joblib  # Para salvar/carregar modelos treinados (não utilizado aqui, mas pode ser útil futuramente)
from nltk.corpus import stopwords  # Palavras irrelevantes (stopwords) para vetorização
import nltk  # Toolkit de NLP
from flask_cors import CORS  # Libera CORS para requisições entre diferentes origens

# 🔧 Define o caminho do executável do Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"

# 🚀 Criação do app Flask e configuração de CORS
app = Flask(__name__)
CORS(app)  # Permite chamadas de qualquer origem (frontend separado)

# 📁 Pasta para uploads temporários
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 📂 Pastas com imagens para treino do modelo
PASTA_CUPONS = "cupons"           # Imagens que são cupons fiscais
PASTA_NAO_CUPONS = "nao_cupons"   # Imagens que não são cupons fiscais

# 📄 Função para extrair texto de uma imagem com OCR
def extrair_texto(imagem_path):
    img = cv2.imread(imagem_path)               # Lê a imagem
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converte para escala de cinza
    texto = pytesseract.image_to_string(img, lang='por')  # Extrai texto usando OCR com idioma português
    return texto

# 🧠 Função que treina o modelo de Machine Learning com base nas imagens
def treinar_modelo():
    textos, rotulos = [], []

    # Itera sobre as duas categorias de imagens
    for pasta, rotulo in [(PASTA_CUPONS, 1), (PASTA_NAO_CUPONS, 0)]:
        for arquivo in os.listdir(pasta):
            caminho = os.path.join(pasta, arquivo)
            try:
                texto = extrair_texto(caminho).strip()
                if texto: ## só adiciona se texto não for vazio
                    textos.append(texto)
                    rotulos.append(rotulo)
                else:
                    print(f"[AVISO] Texto vazio em {caminho}")
            except Exception as e:
                print(f"[ERRO] Falha ao processar {caminho}: {e}")

    # Verifica se foi possível extrair algum texto
    if not textos:
        raise ValueError("Nenhum texto extraído válido para treinar o modelo!")

    # Baixa e carrega as stopwords (palavras irrelevantes) em português
    nltk.download('stopwords')
    stopwords_portugues = stopwords.words('portuguese')

    # Cria o pipeline: vetorização + regressão logística
    modelo = make_pipeline(
        TfidfVectorizer(stop_words=stopwords_portugues),
        LogisticRegression()
    )

    # Treina o modelo
    modelo.fit(textos, rotulos)
    return modelo

# Treina o modelo ao iniciar o servidor
modelo = treinar_modelo()

# 📥 Rota para verificar se uma imagem enviada é um cupom fiscal
@app.route('/verificar', methods=['POST'])
def verificar_imagem():
    if 'imagem' not in request.files:
        return jsonify({'erro': 'Envie uma imagem com o campo "imagem"'}), 400

    # Salva a imagem recebida
    arquivo = request.files['imagem']
    caminho = os.path.join(UPLOAD_FOLDER, secure_filename(arquivo.filename))
    arquivo.save(caminho)

    # Extrai texto da imagem
    texto = extrair_texto(caminho)

    if not texto.strip():
        return jsonify({'erro': 'Texto não pôde ser extraído da imagem.'}), 400

    # Faz a predição com o modelo treinado
    pred = modelo.predict([texto])[0]  # 0 = não é cupom, 1 = é cupom
    prob = modelo.predict_proba([texto])[0][1]  # Probabilidade de ser cupom

    # Monta resposta JSON
    resultado = {
        "resultado": "É cupom fiscal" if pred == 1 else "Não é cupom fiscal",
        "confiança": round(prob * 100, 2),
        "texto_extraido": texto.strip()
    }

    return jsonify(resultado)

# 🔁 Inicia o servidor Flask em modo de depuração
if __name__ == '__main__':
    app.run(debug=True)
