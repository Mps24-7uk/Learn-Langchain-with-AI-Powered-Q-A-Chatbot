

import os
import streamlit as st
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
import pickle
from constant_huggingface import huggingface_key

HUGGINGFACEHUB_API_TOKEN = huggingface_key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN)

with open('faiss_langchain_db.pickle', 'rb') as handle:
      db_m1  = pickle.load(handle)["embedding"]
        
# Streamlit UI setup
st.title("Learn Langchain with AI-Powered Q&A Chatbot")

def main():
  query = st.text_input('Ask Question about Langchain.')


  # Check if the query is non-empty and no cancel button was pressed
#  if query and not st.button("Clear"):
  if query:   
      retriever = db_m1.as_retriever()
      # Build our Langchain chain instance.
      chain = RetrievalQA.from_chain_type( llm=llm,retriever=retriever)
      result = chain.invoke({"query":query})
      st.write(result['result'])


if __name__ == "__main__":
  main()
