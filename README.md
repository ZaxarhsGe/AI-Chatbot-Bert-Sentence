# AI-Chatbot-Bert-Sentence
This repository contains a chatbot application that uses BERT-based sentence transformers to find the best-matching responses to user queries. It leverages a pre-defined set of question-answer pairs to respond to questions in both Greek and English. The application is built using Python, Flask, and PyTorch, and is accessible through an HTTPS endpoint.

## Features
- **Multilingual Support:** Handles both Greek and English queries using a BERT-based model.
- **Cosine Similarity Matching:** Uses cosine similarity to match user queries with pre-defined questions.
- **REST API:** Offers a `/chat` endpoint to interact with the chatbot.
- **CORS Support:** Enabled with Flask-CORS for cross-origin resource sharing.
- **Pre-defined Q&A:** Utilizes a hardcoded list of question-answer pairs.

## Prerequisites
- Python 3.x
- Flask
- PyTorch
- NumPy
- scikit-learn
- Transformers
- Sentence Transformers

## Installation
1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. Create and activate a virtual environment (optional):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Add the following packages to `requirements.txt` if not already present:
    ```
    torch
    numpy
    scikit-learn
    transformers
    flask
    flask-cors
    sentence-transformers
    ```

## Usage
1. **Run the application:**
    ```bash
    python app.py
    ```
    The application will start on port 8000 by default.

2. **Access the Chatbot:**
   - Navigate to `http://localhost:8000` to access the web interface.
   - Use the `/chat` endpoint with a `POST` request to interact with the chatbot via JSON.

## API Endpoints
### `GET /`
Returns the main webpage (`index.html`), which serves as the user interface for the chatbot.

### `POST /chat`
- **Request Body:** JSON object containing the user's message:
    ```json
    {
      "message": "Your question here"
    }
    ```
- **Response:** Returns a JSON object with the chatbot's response:
    ```json
    {
      "response": "Chatbot's answer"
    }
    ```

## Code Overview
- **Model:** Uses the `paraphrase-multilingual-MiniLM-L12-v2` model from Sentence Transformers for encoding sentences.
- **Cosine Similarity:** Calculates the similarity between the user's input and pre-defined questions to find the best match.
- **Thresholding:** Only returns a response if the similarity score exceeds a set threshold (0.6).
- **Error Handling:** Includes basic error handling for invalid inputs or model issues.

## Customization
- Add or modify question-answer pairs in the `questions_answers` list.
- Adjust the similarity threshold in the `chatbot_response` function as needed.

## Error Handling
- If an error occurs during sentence encoding or chatbot response generation, an appropriate error message will be returned to the user.

## License
This project is open-source and free to use for educational and commercial purposes.

## Acknowledgments
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for the pre-trained BERT models.
- [Sentence Transformers](https://www.sbert.net/) for providing a straightforward API to encode sentences.

## Troubleshooting
- If the chatbot does not respond as expected, check for errors in the console where the Flask app is running.
- Ensure that all dependencies are installed correctly and that the `requirements.txt` file includes all necessary packages.

## Contact
For further questions or support, please reach out to `maxilaria.gr@gmail.com`.
