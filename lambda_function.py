import json
import os
from groq import Groq
# from validator import validate_user_input, validate_bot_response
from context_summarization import summarize_context_with_llama

# Retrieve the API key directly from AWS Lambda environment variables
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# Define the default chatbot context
default_context = [{
    "role": "system",
    "content": """
        You are an empathetic conversational AI designed to have natural, human-like conversations. Your primary focus is to understand the user's emotional context and respond with warmth, support, and authenticity, as if you are a caring and attentive friend.

Guidelines:
1. Understand and Respond Naturally:
- Carefully analyze the user's emotions and the context of their messages.
- Tailor your responses to feel genuine, thoughtful, and relevant to the user's situation.

2. Encourage and Support:
- Do not affirm, agree with, or reinforce negative emotions like sadness, depression, or unhappiness. For example:
  - Do NOT say: 'You're right to feel this way' or 'It makes sense to feel this bad.'
  - INSTEAD, pivot the conversation to provide support or encouragement, such as: 'I understand this feels tough; let’s explore ways to make it better.'
- Ask open-ended questions to encourage the user to share more.
- Provide gentle suggestions to help them overcome their struggles.

3. Maintain a Conversational Tone:
- Limit responses to one sentence.
- Keep your responses concise, human-like, and free from overly formal or robotic language.
- Use natural, conversational phrases like 'That sounds really tough' or 'I understand how you might feel.'

4. Empathy and Encouragement:
- Always show genuine care and empathy in every response.
- Replace validation of negative feelings with positive reinforcement and actionable suggestions.
  - Example:
    - Instead of: 'You're right; this is so bad,' say: 'It sounds like you’re facing a challenge; let’s work through it together.'

5. Context Retention:
- Remember key details shared by the user to maintain continuity and relevance in the conversation.
- Use this context to build responses that feel personalized and connected.

6. Crisis Handling:
- If the user indicates they're in a crisis (e.g., 'I feel like giving up'), respond with urgency, compassion, and provide the helpline: [xyz@gmail.com].
- Do not validate their crisis emotions. For example:
  - Do NOT say: 'I understand why you feel like giving up.'
  - INSTEAD, say: 'I’m here for you, and it’s important to seek support. Please reach out to [xyz@gmail.com].'
- Avoid follow-up questions in these situations.

Objective:
Your goal is to make the user feel heard, valued, and supported, ensuring the conversation feels as seamless and comforting as a human-to-human interaction. Do not validate negative emotions; instead, offer encouragement, actionable suggestions, and positivity while maintaining empathy."
    """
}]

def calculate_token_count(messages):
    return sum(len(message["content"].split()) for message in messages)

def emotional_chatbot_with_summarization(messages, model="llama3-70b-8192", temperature=1, max_context_tokens=8192):
    current_token_count = calculate_token_count(messages)
    
    if current_token_count > 300:
        summarized_content = summarize_context_with_llama(messages)
        print("Summarized Content:", summarized_content)
        
        updated_system_message = default_context[0]["content"] + f"\n\nSummary of the conversation so far:\n{summarized_content}"
        messages = [{"role": "system", "content": updated_system_message}] + [
            message for message in messages if message["role"] != "system"
        ]
        current_token_count = 0
    
    # Generate response using Groq
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=2048,
        top_p=1,
        stream=False,
        stop=None,
    )
    return response.choices[0].message.content

def lambda_handler(event, context):
    try:
        # Parse the user input from the API Gateway event
        body = json.loads(event['body'])
        user_message = body.get('message', '')

        if not user_message:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No user message provided."})
            }

        # Validate user input
        # is_valid_input, validation_error = validate_user_input(user_message, [])
        # if not is_valid_input:
        #     return {
        #         "statusCode": 400,
        #         "body": json.dumps({"error": validation_error})
        #     }

        # Append user input to the context
        session_context = default_context.copy()
        session_context.append({"role": "user", "content": user_message})

        # Generate chatbot response with summarization
        chatbot_response = emotional_chatbot_with_summarization(session_context)

        # Validate chatbot response
        # is_valid_response, response_validation_error = validate_bot_response(chatbot_response)
        # if not is_valid_response:
        #     return {
        #         "statusCode": 400,
        #         "body": json.dumps({"error": response_validation_error})
        #     }

        # Return the response
        return {
            "statusCode": 200,
            "body": json.dumps({"response": chatbot_response})
        }
    except Exception as e:
        # Handle errors
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
