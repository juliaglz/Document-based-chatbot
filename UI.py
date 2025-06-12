import streamlit as st
from openai import AzureOpenAI
import os
from ChromaDB import ChromaDBManagerS
from doc_parser import CleanTextExtractor
from doc_chunker import TextChunker
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
#COMPLETE WITH API INFO
)

# Function to interact with Azure OpenAI
def interact_with_ai(user_input):
    message_text = [{"role": "user", "content": user_input}]
    completion = client.chat.completions.create(
        model="TFG",
        messages=message_text,
        temperature=0.7,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None
    )
    ai_response = ""
    for choice in completion.choices:
        for message in choice.message:
            if isinstance(message, tuple) and message[0] == 'content':
                ai_response = message[1]
                break
        if ai_response:
            break
    return ai_response

# Streamlit UI
st.title("AI Conversational Bot")
output_placeholder = st.empty()
# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# Get user input
text_input = st.text_input("Enter your question:")

# Respond to user input
if st.button("Ask"):
    # Database and document handling
    db_path = "path"
    db_manager = ChromaDBManagerS(db_path)
    collection_name = "documents_chroma"
    collection = db_manager.create_collection(collection_name)

    data_directory = ".\wiki2txt"
    documents = []
    ids = []
    existing_ids = set(db_manager.get_ids(collection))
    for filename in os.listdir(data_directory):
        filepath = os.path.join(data_directory, filename)
        extractor = CleanTextExtractor(filepath)
        document_text = extractor.extract_text()
        chunker = TextChunker(document_text)
        chunks = chunker.chunk_by_sentences()

        for i, chunk in enumerate(chunks, start=1):
            chunk_id = f"{filename}_chunk_{i}"
            if chunk_id not in existing_ids:
                documents.append(chunk)
                ids.append(chunk_id)
    if documents:
        db_manager.add_documents(collection, documents, ids)

    # Query the database for context
    results = db_manager.query_collection(collection, text_input, 1)
    context = results['documents']
    context2 = str(context)

    # Get AI's answer
    answer = interact_with_ai("CONTEXT" + context2 + "QUESTION" + text_input)

    # Update chat history
    st.session_state["chat_history"].append({"role": "user", "content": text_input})
    st.session_state["chat_history"].append({"role": "assistant", "content": answer})

    # Refresh the page to update chat history display

    chat_history_html = ""
    for msg in st.session_state.chat_history:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                chat_history_html += f'<div style="text-align:right;"><b>{role}:</b> {content}</div>'
            elif role == "assistant":
                chat_history_html += f'<div style="text-align:left;"><b>{role}:</b> {content}</div>'
            else:
                chat_history_html += f'<div><b>Unknown Role:</b> {content}</div>'
        else:
            chat_history_html += f'<div><b>Unexpected Data Type:</b> {msg}</div>'

    output_placeholder.markdown(chat_history_html, unsafe_allow_html=True)
