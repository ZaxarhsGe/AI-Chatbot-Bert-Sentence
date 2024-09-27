# AI Chatbot with BERT and Flask

This repository contains a chatbot application that uses BERT-based sentence transformers to find the best-matching responses to user queries. It leverages a pre-defined set of question-answer pairs to respond to questions in both Greek and English. The application is built using Python, Flask, and PyTorch, and is deployed using AWS.

## Features
- **Multilingual Support:** Handles both Greek and English queries using a BERT-based model.
- **Cosine Similarity Matching:** Uses cosine similarity to match user queries with pre-defined questions.
- **REST API:** Offers a `/chat` endpoint to interact with the chatbot.
- **CORS Support:** Enabled with Flask-CORS for cross-origin resource sharing.
- **Pre-defined Q&A:** Utilizes a hardcoded list of question-answer pairs.
- **Production Deployment:** Hosted on AWS EC2 for robust and scalable deployment.

## Prerequisites
- Python 3.x
- Flask
- PyTorch
- NumPy
- scikit-learn
- Transformers
- Sentence Transformers
- An active AWS account (for deployment)

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
1. **Run the application locally:**
    ```bash
    python app.py
    ```
    The application will start on port 8000 by default.

2. **Access the Chatbot Locally:**
   - Navigate to `http://localhost:8000` to access the web interface.
   - Use the `/chat` endpoint with a `POST` request to interact with the chatbot via JSON.

## Deployment on AWS
1. **Launch an EC2 Instance:**
   - Log in to your AWS account and navigate to the EC2 Dashboard.
   - Launch a new EC2 instance (e.g., `m5.large` for sufficient processing power).
   - Choose an appropriate AMI (Amazon Machine Image) with Python installed (e.g., Ubuntu 20.04 LTS).
   
2. **Set Up the EC2 Instance:**
   - SSH into the instance:
     ```bash
     ssh -i "your-key.pem" ubuntu@your-ec2-public-dns
     ```
   - Install required packages on the EC2 instance:
     ```bash
     sudo apt update
     sudo apt install python3-pip
     pip3 install -r requirements.txt
     ```

3. **Deploy the Application:**
   - Upload the application files to the EC2 instance using `scp` or a similar method.
   - Start the Flask app with a production-ready server, such as Gunicorn:
     ```bash
     gunicorn --bind 0.0.0.0:8000 app:app
     ```

4. **Configure Security Groups:**
   - Open port 8000 in the EC2 instance's security group to allow traffic to the application.

5. **Access the Application:**
   - Navigate to `http://your-ec2-public-dns:8000` to access the chatbot.

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
- **AWS** for providing a robust platform to deploy and host this chatbot application.

## Troubleshooting
- If the chatbot does not respond as expected, check for errors in the console where the Flask app is running.
- Ensure that all dependencies are installed correctly and that the `requirements.txt` file includes all necessary packages.
- Verify that the correct security groups are configured in AWS to allow traffic to the EC2 instance.

## Contact
For further questions or support, please reach out to `maxilaria.gr@gmail.com`.
