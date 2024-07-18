# app.py
import asyncio
from flask import Flask, request, jsonify, render_template
from chatbot.chatbot import get_answer, initialize_chatbot

app = Flask(__name__)

initialize_chatbot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chat():
    data = request.json
    query = data.get('message', '')

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = loop.run_until_complete(get_answer(query))

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)




