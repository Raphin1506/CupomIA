from flask import Flask, request, jsonify
import os
import cv2
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from werkzeug.utils import secure_filename
from PIL import Image
import joblib
from nltk.corpus import stopwords
import nltk
from flask_cors import CORS

# 👇 Força o caminho do executável do Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Caminhos das pastas
PASTA_CUPONS = "cupons"
PASTA_NAO_CUPONS = "nao_cupons"

# 🧠 Função para extrair texto
def extrair_texto(imagem_path):
    img = cv2.imread(imagem_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texto = pytesseract.image_to_string(img, lang='por')
    return texto

# 🤖 Treinar modelo com base nas pastas
def treinar_modelo():
    textos, rotulos = [], []
    
    for pasta, rotulo in [(PASTA_CUPONS, 1), (PASTA_NAO_CUPONS, 0)]:
        for arquivo in os.listdir(pasta):
            caminho = os.path.join(pasta, arquivo)
            try:
                texto = extrair_texto(caminho)
                texto = texto.strip()
                if texto:  # só adiciona se texto não for vazio
                    textos.append(texto)
                    rotulos.append(rotulo)
                else:
                    print(f"[AVISO] Texto vazio em {caminho}")
            except Exception as e:
                print(f"[ERRO] Falha em {caminho}: {e}")
    
    if not textos:
        raise ValueError("Nenhum texto extraído válido para treinar o modelo!")

  
    nltk.download('stopwords')
    stopwords_portugues = stopwords.words('portuguese')

    modelo = make_pipeline(TfidfVectorizer(stop_words=stopwords_portugues), LogisticRegression())

    modelo.fit(textos, rotulos)
    return modelo

modelo = treinar_modelo()

# 📥 Rota principal para enviar imagem
@app.route('/verificar', methods=['POST'])
def verificar_imagem():
    if 'imagem' not in request.files:
        return jsonify({'erro': 'Envie uma imagem com o campo "imagem"'}), 400
    
    arquivo = request.files['imagem']
    caminho = os.path.join(UPLOAD_FOLDER, secure_filename(arquivo.filename))
    arquivo.save(caminho)

    texto = extrair_texto(caminho)

    if not texto.strip():
        return jsonify({'erro': 'Texto não pôde ser extraído da imagem. Provavelmente isso não é um cupom fiscal, ou a imagem está corrompida/ilegivel.'}), 400
    
    pred = modelo.predict([texto])[0]
    prob = modelo.predict_proba([texto])[0][1]

    resultado = {
        "resultado": "É cupom fiscal" if pred == 1 else "Não é cupom fiscal",
        "confiança": round(prob * 100, 2),
        "texto_extraido": texto.strip()
    }

    return jsonify(resultado)

# 🚀 Roda a aplicação
if __name__ == '__main__':
    app.run(debug=True)
