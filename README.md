# 🧾 Verificador de Cupons Fiscais com Flask, OCR e Machine Learning

Este projeto consiste em uma API Flask que utiliza OCR (pytesseract) e Machine Learning (scikit-learn) para identificar se uma imagem enviada representa um **cupom fiscal** ou **não**. O sistema realiza a extração de texto da imagem e classifica com base em um modelo treinado com exemplos.

## 🚀 Funcionalidades

- Extração de texto de imagens com o Tesseract OCR.
- Treinamento automático com imagens separadas por categorias.
- Classificação via `LogisticRegression` com vetorização TF-IDF.
- API REST com Flask para envio e análise de imagens.
- Interface web simples para interação com o sistema.

## 📁 Estrutura esperada

projeto/
├── app.py
├── cupons/ # Imagens de cupons fiscais (treinamento)
├── nao_cupons/ # Imagens que não são cupons fiscais (treinamento)
├── uploads/ # Imagens recebidas pela API
├── interface.html # Interface web para teste
└── README.md

## ⚙️ Requisitos

- Python 3.7+
- Tesseract OCR instalado (e caminho configurado no `app.py`)
- Bibliotecas Python:
  - flask
  - opencv-python
  - pytesseract
  - scikit-learn
  - nltk
  - pillow

Instale as dependências com:

pip install flask opencv-python pytesseract scikit-learn nltk pillow

🧠 Treinamento do modelo

O modelo é treinado automaticamente na inicialização do sistema, utilizando as imagens nas pastas:

cupons/ para imagens que representam cupons fiscais

nao_cupons/ para imagens que não são cupons

▶️ Executando o projeto

Certifique-se de que o Tesseract OCR está instalado:

Windows: Download aqui

Configure o caminho do Tesseract no app.py:


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

Execute a API:
python app.py

Acesse a interface HTML (interface.html) via navegador ou use Postman/cURL para testar a rota /verificar.

🌐 Teste com curl
curl -X POST http://localhost:5000/verificar -F "imagem=@caminho/da/imagem.jpg"

💡 Exemplo de resposta JSON

{
  "resultado": "É cupom fiscal",
  "confiança": 94.27,
  "texto_extraido": "SUPERMERCADO ABC\nCNPJ..."
}

📃 Licença
Este projeto é de uso livre para fins acadêmicos e educacionais.
