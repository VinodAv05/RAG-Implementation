**Use Case Summary**

By combining local document retrieval with generative capabilities of advanced language models, this RAG-based system ensures that generated responses are both relevant and grounded in trusted, domain-specific information. This is particularly useful in scenarios where precise, context-aware information retrieval from custom data sources is critical.

**Project Overview**

This project demonstrates the use of Retrieval-Augmented Generation (RAG) — a powerful technique that combines information retrieval with text generation — applied to a set of documents located in the ml-doc-notes folder. The objective is to enhance the performance and relevance of generative AI models by grounding responses in custom document content.

**Implementation Details**

**_1. Setting Up the Vector Database_**

To facilitate efficient retrieval, we use ChromaDB, a lightweight and open-source vector database. Instructions for setting up and populating the database are provided in the create_vector_db.py script. This setup includes initializing the database and preparing it to store embedded document chunks.

**_2. Loading and Preprocessing Documents_**

The source documents, which are in Markdown format, are ingested into ChromaDB. During this stage:

The documents are chunked into smaller segments to allow for granular retrieval.

Each chunk is embedded into a high-dimensional vector space using embedding models via LangChain, ensuring semantic search capabilities.
This process allows the system to later retrieve document sections that are most relevant to a given query.

**_3. Generating Context and Answers_**

Once the vector store is populated, we perform the following during query time:

The user input is converted into an embedding.

A semantic search is executed over the ChromaDB collection to retrieve the most relevant document chunks.

These chunks are then supplied as context to a Google GenAI model (such as Gemini or PaLM) to generate accurate and contextually-grounded responses.

Additionally, users are given the flexibility to:

Restrict the search to only the ingested custom documents (i.e., ml-doc-notes), or

Expand the search to include the broader Google GenAI knowledge base for more comprehensive answers.
