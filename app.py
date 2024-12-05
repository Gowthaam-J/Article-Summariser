#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings  # Or the embedding model you used
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings


# Title of the Streamlit app
st.title("Query Input App")

# Step 1: Create three input boxes in Streamlit
query1 = st.text_input("Query 1:")
query2 = st.text_input("Query 2:")
query3 = st.text_input("Query 3:")

# Step 2: Load FAISS index and embeddings
faiss_index_path = "faiss_index"  # The folder where `index.faiss` and `index.pkl` are stored

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name=model_name)

# Load the FAISS vector store from disk
vectorstore = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)  # Allow pickle deserialization
retriever = vectorstore.as_retriever()

# Initialize the Llama model via ChatGroq
llm = ChatGroq(
    temperature=0.7,
    groq_api_key="your_groq_api_key_here",  # Replace with your actual API key
    model_name="llama-3.1-70b-versatile"
)

# Step 3: Define the Langchain prompt template
prompt_template = """
Given the following context:
{context}

Answer the following question:
{question}
"""

# Create the PromptTemplate and LLMChain
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt
)

# Step 4: Logic to process the queries and find answers
if query1 or query2 or query3:
    if st.button("Find Answer"):
        # Combine all queries into a single list
        queries = [q for q in [query1, query2, query3] if q]

        combined_context = ""
        for query in queries:
            # Step 5: Retrieve relevant documents (chunks) for each query
            retrieved_docs = retriever.get_relevant_documents(query)
            
            # Combine the chunks for each query into a single context
            combined_context += "\n".join(
                [f"Source: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content}" for doc in retrieved_docs]
            )

        # Step 6: Run the LLMChain with the combined context and query
        result = llm_chain.run({
            "context": combined_context,
            "question": "Summarize the key information from the above context."
        })

        # Step 7: Output the final answer in the Streamlit app
        st.write("Answer:", result)

        # Optionally, display the retrieved chunks as well for transparency
        st.write("Retrieved Chunks:")
        for doc in retrieved_docs:
            st.write(f"Source: {doc.metadata.get('source', 'N/A')}")
            st.write(f"Content: {doc.page_content}")

else:
    st.warning("Please fill at least one query to find answers.")


# In[ ]:




