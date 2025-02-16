import os
from groq import Groq

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

def summarize_context_with_llama(messages, max_tokens=512):
   
    
    context_text = " ".join([message["content"] for message in messages if message["role"] != "system"])
    
    # Summarization prompt
    summarization_prompt = [
        {"role": "system", "content": "Summarize the following conversation in a concise and meaningful way. My output should be a short summary of the conversation."},
        {"role": "user", "content": context_text}
    ]
    
    response = client.chat.completions.create(
        messages=summarization_prompt,
        model="llama3-70b-8192",
        temperature=0, 
        max_tokens=max_tokens, 
        top_p=1,
        stream=False,
        stop=None,
    )

    summarized_content = response.choices[0].message.content
    return summarized_content
