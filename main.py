from flask import Flask, request, render_template
import sys
import os
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

documents = []
for file in os.listdir("ai_camp_data"):
    if file.endswith(".pdf"):
        """pdf_path = "./docs/" + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        doc_path = "./docs/" + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())"""
    elif file.endswith('.txt'):
        text_path = "./ai_camp_data/" + file
        loader = TextLoader(text_path)
        documents.extend(loader.load())
        
text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=10)
documents = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-base")

vectordb = FAISS.from_documents(documents, embeddings)

pdf_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo-16k"),
    vectordb.as_retriever(search_type = "similarity_score_threshold", 
                          search_kwargs={'score_threshold': 0.2, 'k': 5}),
    return_source_documents=True,
    verbose=False
)

app = Flask(__name__)

chat_history = []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['name']
        result = pdf_qa({"question": query, "chat_history": chat_history})
        chat_history.append((query, "True"))  # "True" indicates a user message
        #chat_history.append((result["answer"], "False"))  # "False" indicates an AI response
        website_url = (result['source_documents'][0].metadata)['source'].split('/')[2][:-4]
        website_url = website_url.replace("[", "/")
        final_answer = str(result['answer']) + " \n Please find the source at \n " + website_url
        print(final_answer)
        chat_history.append((final_answer, "False"))  # "False" indicates an AI response
        # Get the last message index
        last_message_index = len(chat_history) - 1

        return render_template(
            'index.html',
            chat_history=chat_history,
            question=query,
            greeting=final_answer,
            last_message_index=last_message_index  # Pass it to the template
        )

    return render_template('index.html', chat_history=chat_history, greeting=None)

if __name__ == '__main__':
    app.run(debug=True)