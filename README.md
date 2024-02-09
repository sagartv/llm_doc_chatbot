## LLM Document Expert

A Streamlit App that uses Langchain, OpenAI Embeddings, GPT 3.5-Turbo, and Pinecone Vector Databases to process a user-provided document.
The document is chunked, and then converted to word embeddings using OpenAI Embeddings. The embeddings are inserted into a Pinecone Index which is deleted after runtime.
Langchain is used to retrieve information through the QA

Upload the document in the sidebar: .pdf, .docx, and .txt files are supported.
You can also control chunk size to improve the quality of the responses.


Use streamlit run doc_chat.py to run the app, upload the document, and then proceed to chat with the doc. Don't forget to Delete the Pinecone Index at the end of the session. 
