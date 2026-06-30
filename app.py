import os
import re
import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Hinglish Support Bot", page_icon="💬", layout="centered")
st.title("💬 Customer Support Assistant")
st.markdown("Ask any question about company policies, and get a natural Hinglish response.")

hf_token = st.secrets.get("HF_TOKEN")

# Using the robust, fully-supported base model on the free tier
client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct", 
    token=hf_token
)

def process_customer_support(user_query):
    if not user_query.strip():
        return "Please enter a valid query."
    try:
        retrieved_context = (
            "If you forget your password, click 'Forgot Password' on the login landing page, "
            "enter your registered email address, and open the password reset link sent to your inbox. "
            "The link expires in 15 minutes."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful customer support assistant. You write your internal logic in English inside <think> tags. "
                    "Immediately after the closing </think> tag, write your response in fluent, natural conversational Hinglish with flawless Hindi sentence grammar.\n"
                    "Do not use formal or pure Hindi words. Use casual, helpful Hinglish that an Indian customer care executive would use.\n\n"
                    "Example:\n"
                    "Aap chinta mat kijiye. Aap sabse pehle login page par jaakar 'Forgot Password' par click karein. Phir apne registered email address par aaye hue password reset link ko open karein. Yeh link 15 mins tak valid rahega."
                )
            },
            {
                "role": "user",
                "content": f"Context policy:\n{retrieved_context}\n\nUser Question:\n{user_query}"
            }
        ]
        response = client.chat_completion(messages=messages, max_tokens=512)
        output_text = response.choices[0].message.content
        
        # Cleanly isolate the response text if thinking tags exist
        final_answer = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL).strip()
        return final_answer
    except Exception as e:
        return f"System Error: {str(e)}"

user_input = st.text_input(label="Ask your question here:", placeholder="Type here... (e.g., password bhul gya help)")

if st.button("Submit Query", type="primary") or user_input:
    if not user_input.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Assistant is thinking..."):
            answer = process_customer_support(user_input)
            st.markdown("### Response:")
            st.info(answer)
