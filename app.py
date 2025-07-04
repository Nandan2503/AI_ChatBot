from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure OpenAI Configuration
# TODO: Replace these with your actual Azure OpenAI credentials
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "YOUR_AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource-name.openai.azure.com/")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4o")  # Your deployment name

# Initialize Azure OpenAI client
client = openai.AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# Store conversation history (in production, use a database)
conversations = {}

def get_conversation_id(request):
    """Get or create a conversation ID for session management"""
    # For simplicity, using IP address. In production, use proper session management
    return request.remote_addr

@app.route('/')
def home():
    return "Azure AI Chatbot Backend is running!"

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        user_message = data['message']
        conversation_id = get_conversation_id(request)
        
        # Initialize conversation history if not exists
        if conversation_id not in conversations:
            conversations[conversation_id] = [
                {"role": "system", "content": "You are a helpful AI assistant. Be concise and friendly in your responses."}
            ]
        
        # Add user message to conversation history
        conversations[conversation_id].append({"role": "user", "content": user_message})
        
        # Keep only last 10 messages to manage token limits
        if len(conversations[conversation_id]) > 10:
            conversations[conversation_id] = conversations[conversation_id][-10:]
        
        logger.info(f"Processing message from {conversation_id}: {user_message}")
        
        # Call Azure OpenAI API
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=conversations[conversation_id],
            max_tokens=500,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        # Extract the assistant's response
        assistant_response = response.choices[0].message.content
        
        # Add assistant response to conversation history
        conversations[conversation_id].append({"role": "assistant", "content": assistant_response})
        
        logger.info(f"Response generated for {conversation_id}")
        
        return jsonify({'response': assistant_response})
    
    except openai.AuthenticationError:
        logger.error("Azure OpenAI authentication failed")
        return jsonify({'error': 'Authentication failed. Please check your Azure OpenAI credentials.'}), 401
    
    except openai.RateLimitError:
        logger.error("Azure OpenAI rate limit exceeded")
        return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
    
    except openai.APIError as e:
        logger.error(f"Azure OpenAI API error: {str(e)}")
        return jsonify({'error': f'API error: {str(e)}'}), 500
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500

@app.route('/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history for debugging"""
    try:
        conversation_id = get_conversation_id(request)
        if conversation_id in conversations:
            del conversations[conversation_id]
        return jsonify({'message': 'Conversation cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'azure_openai_configured': bool(AZURE_OPENAI_API_KEY and AZURE_OPENAI_API_KEY != "YOUR_AZURE_OPENAI_API_KEY")
    })

if __name__ == '__main__':

    # Check if Azure credentials are configured
    if AZURE_OPENAI_API_KEY == "YOUR_AZURE_OPENAI_API_KEY":
        print("WARNING: Please configure your Azure OpenAI credentials in the .env file")
        print("Create a .env file with:")
        print("AZURE_OPENAI_API_KEY=your_actual_api_key")
        print("AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/")
        print("DEPLOYMENT_NAME=your_deployment_name")
    
    app.run(debug=True, host='0.0.0.0', port=5000)