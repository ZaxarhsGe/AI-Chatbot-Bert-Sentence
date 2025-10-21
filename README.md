# 🤖 AI Chatbot with BERT and Flask  

This project showcases a **freelance chatbot application** built using **Python**, **Flask**, and **BERT-based Sentence Transformers**. It identifies the best-matching responses to user queries in **Greek** and **English**, using a pre-defined set of Q&A pairs.  

The chatbot is fully deployed on **AWS EC2**, exposing a REST API that integrates seamlessly with web frontends such as WordPress.  

---

## 🚀 Key Features  
- **Multilingual Support:** Understands and responds to both Greek and English queries using a multilingual BERT model.  
- **Semantic Matching:** Uses **cosine similarity** between embeddings to find the most relevant response.  
- **REST API Integration:** Flask-based `/chat` endpoint for real-time user interaction via HTTP requests.  
- **Cross-Origin Support:** Configured with **Flask-CORS** for cross-domain communication.  
- **AWS Deployment:** Hosted on an EC2 instance with **Nginx**, **Gunicorn**, and **SSL certificates** for production reliability.  
- **Lightweight & Extendable:** Uses pre-defined Q&A pairs but can be easily connected to databases or third-party APIs.  

---

## 🧠 Tech Stack  
- **Languages:** Python  
- **Frameworks & Libraries:** Flask, PyTorch, Transformers, Sentence Transformers, scikit-learn, NumPy  
- **Cloud & Deployment:** AWS EC2, Nginx, Gunicorn, Certbot (for HTTPS)  

---

## ⚙️ Installation  

1. **Clone the Repository**  
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Set Up Environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run Locally**  
   ```bash
   python app.py
   ```
   Access at `http://localhost:8000`

---

## ☁️ Deployment on AWS  

1. **Launch an EC2 Instance** (e.g., `m5.large`, Ubuntu 20.04).  
2. **Install Dependencies:**  
   ```bash
   sudo apt update
   sudo apt install python3-pip
   pip3 install -r requirements.txt
   ```
3. **Run with Gunicorn:**  
   ```bash
   gunicorn --bind 0.0.0.0:8000 app:app
   ```
4. **Configure Security Groups** to open port `8000`.  
5. **Access Live Chatbot** via your EC2 public DNS or connected domain.  

---

## 🔌 API Endpoints  

### `GET /`  
Serves the main chatbot webpage (`index.html`).  

### `POST /chat`  
**Request:**  
```json
{ "message": "Your question here" }
```  
**Response:**  
```json
{ "response": "Chatbot's answer" }
```  

---

## 🧩 Code Overview  
- **Model:** `paraphrase-multilingual-MiniLM-L12-v2` (Sentence Transformers)  
- **Logic:** Uses cosine similarity between embeddings to match user input with predefined Q&A pairs.  
- **Threshold:** Responses are returned only if similarity > 0.6.  
- **Error Handling:** Graceful fallback for invalid inputs or model loading issues.  

---

## 🛠️ Customization  
- Modify `questions_answers` list for new responses.  
- Adjust similarity threshold in `chatbot_response()` for stricter or looser matching.  
- Replace local data with a database or API-driven source for scalability.  

---

## 🔍 Troubleshooting  
- Verify dependencies with `pip list`.  
- Check console logs for Flask errors.  
- Ensure AWS security groups allow inbound HTTP/HTTPS traffic.  

---

## 🙌 Acknowledgments  
- [Hugging Face Transformers](https://github.com/huggingface/transformers)  
- [Sentence Transformers](https://www.sbert.net/)  
- **AWS** for hosting and deployment infrastructure  

---

## 📧 Contact  
For inquiries or collaboration opportunities:  
**George Zacharis**  
📩 `maxilaria.gr@gmail.com`  
