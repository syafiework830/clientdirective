from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from langchain_openai.chat_models.azure import AzureChatOpenAI
import dotenv
import os, re  
from typing import List, Any
#from function_libraries import create_rag_chain_with_score, format_rag_response_new_grouped
from langchain_community.vectorstores import AzureSearch
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from operator import itemgetter
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from pydantic import BaseModel, Field

dotenv.load_dotenv()

def bot_model(text, indexname, chat_history):
    """
    Creates a RAG-based chatbot model using Azure OpenAI and Azure Cognitive Search.
    
    Args:
        text (str): The input text/question from the user
        indexname (str): The name of the Azure Search index to use
        chat_history (list): List of previous chat interactions
        
    Returns:
        tuple: (answer, references) where answer is the bot's response and references are the sources
    """
    # Get environment variables
    AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
    CDR_AZURE_COGNITIVE_SEARCH_SERVICE_NAME = os.environ.get("CDR_AZURE_COGNITIVE_SEARCH_SERVICE_NAME")
    CDR_AZURE_COGNITIVE_SEARCH_API_KEY = os.environ.get("CDR_AZURE_COGNITIVE_SEARCH_API_KEY")

    # Initialize embedding model
    embedding = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version="2023-05-15",
        dimensions= 1536
    )
    
    # Initialize chat model
    #model = AzureChatOpenAI(
    #    model="gpt-4o-mini-2024-07-18",
    #    azure_deployment="gpt-4o-mini",
    #    api_key=AZURE_API_KEY,
    #    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    #    api_version="2023-03-15-preview"
    #)

    model = AzureChatOpenAI(model="gpt-4o-2024-08-06",
            azure_deployment="gpt-4o",
            api_key='8dd26d2d746041fe90a8892f68806c3d',
            azure_endpoint="https://ocr-chatbot-mccalla.openai.azure.com/",
            api_version="2024-08-01-preview"
        )
    
    # Set up Azure Cognitive Search
    vector_store_address = f"https://{CDR_AZURE_COGNITIVE_SEARCH_SERVICE_NAME}.search.windows.net"
    vectorstore = AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=CDR_AZURE_COGNITIVE_SEARCH_API_KEY,
        index_name=indexname,
        embedding_function=embedding,
        search_type='semantic_hybrid'
    )
    
    def get_response(chain, question: str, chat_history: list = None):
        """Get response from the chain."""
        if chat_history is None:
            chat_history = []
        return chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
    
    class cited_answer(BaseModel):
        """Answer the user question based only on the given sources, and cite the sources used."""
        answer: str = Field(
            ...,
            description="The answer to the user question, which is based only on the given sources.",
        )
        citations: List[int] = Field(
            ...,
            description="The integer IDs of the SPECIFIC sources which justify the answer.",
        )

    def create_rag_chain_with_score(vectorstore, llm_model):
        """
        Create a RAG chain with chat history and dynamic citations based on similarity score threshold.
        Citations are automatically included only when similarity score >= 0.65
        """
        
        # Simple query detection prompt
        simple_query_prompt = ChatPromptTemplate.from_messages([
            ("system", """Your name is McCalla Client Directive Chatbots. Determine if this is a simple query that doesn't require document lookup.
            Simple queries include:
            - Greetings (hi, hello, hey)
            - Basic math (1+1, 2*3)
            - Simple chitchat
            Return only "true" for simple queries or "false" for queries needing document lookup."""),
            ("human", "{question}")
        ])
        
        # Initialize the LLM with citation tool
        llm_with_tools = llm_model.bind_tools(
            [cited_answer],
            tool_choice="cited_answer",
        )
        output_parser = JsonOutputKeyToolsParser(key_name="cited_answer", first_tool_only=True)
        
        # Create prompts
        base_prompt = ChatPromptTemplate.from_messages([
            ("system", 
                """
                Your name is McCalla Client Directive Chatbots. Your task is to answer the {question} based on the following context: {context}.

                The data you've been fed is confidential therefore, you must provide the best answer to the {question} based on the {context}. 
                You may receive {question} that is not related to the {context} but still you need to answer without any hesitant. 

                You also need to be polite and professional when answering.

                Format the response such that each bullet point ('-', '▪', '•') or numbered point ('1.', '2.') starts on a new line but make sure to differentiate the list heading. 
                Do not add extra blank lines.
                
                Example:
                    
                There are many types of cars, and they can be categorized in different ways based on various factors like purpose, size, design, and features. Here are some common types:

                1. Sedan - A standard car with a trunk and seating for 4-5 people.
                2. SUV (Sport Utility Vehicle) - Larger, with higher ground clearance and often capable of off-road driving.
                3. Hatchback - A car with a rear door that swings upward, offering a larger cargo space.
                """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
            ])

        
        simple_response_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        def format_docs(docs: List[Document]) -> str:
            """Format documents into a simple string."""
            return "\n\n".join([doc.page_content for doc in docs])
        
        def format_docs_with_id(docs: List[Document]) -> str:
            """Format documents with source IDs and metadata."""
            formatted = [
                f"Source ID: {i}\nArticle Title: {doc.metadata.get('title', 'No title')}\nArticle Snippet: {doc.page_content}"
                for i, doc in enumerate(docs)
            ]
            return "\n\n" + "\n\n".join(formatted)
        
        def retrieve_with_history(input_dict):
            """Retrieve documents considering chat history and similarity scores."""
            llm_response = llm_model.invoke(
                retriever_prompt.format(
                    question=input_dict["question"],
                    chat_history=input_dict["chat_history"]
                )
            )
            
            # Get documents with similarity scores
            docs_with_scores = vectorstore.similarity_search_with_relevance_scores(llm_response.content)
            
            # Filter and process documents
            filtered_docs = []
            max_similarity = 0.0
            
            for doc, score in docs_with_scores:
                doc.metadata['similarity_score'] = score
                filtered_docs.append(doc)
                max_similarity = max(max_similarity, score)
            
            return filtered_docs, max_similarity >= 0.60  # Return docs and whether max similarity meets threshold
        
        def handle_simple_query(input_dict):
            """Handle simple queries and complex queries with dynamic citation inclusion."""
            is_simple = llm_model.invoke(
                simple_query_prompt.format(question=input_dict["question"])
            ).content.strip().lower() == "true"
            
            if is_simple:
                # For simple queries, return direct response without citations
                response = llm_model.invoke(
                    simple_response_prompt.format(
                        question=input_dict["question"],
                        chat_history=input_dict["chat_history"]
                    )
                )
                return {
                    "cited_answer": {
                        "answer": response.content,
                        "citations": []
                    },
                    "docs": []
                }
            
            # For complex queries, check similarity scores
            docs, should_include_citations = retrieve_with_history(input_dict)
            
            # If any document has similarity >= 0.7, include citations
            if should_include_citations:
                context = format_docs_with_id(docs)
                response = llm_with_tools.invoke(
                    base_prompt.format(
                        context=context,
                        question=input_dict["question"],
                        chat_history=input_dict["chat_history"]
                    )
                )
                parsed_response = output_parser.invoke(response)
                return {
                    "cited_answer": parsed_response,
                    "docs": docs
                }
            else:
                # For low similarity scores, don't include citations
                context = format_docs(docs) if docs else ""
                response = llm_model.invoke(
                    base_prompt.format(
                        context=context,
                        question=input_dict["question"],
                        chat_history=input_dict["chat_history"]
                    )
                )
                return {
                    "answer": response.content,
                    "docs": docs
                }
        
        # Build the main chain
        chain = RunnableParallel(
            {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history")
            }
        ).assign(
            output=RunnableLambda(handle_simple_query)
        ).pick(["output"])
        
        return chain
    
    def format_rag_response_new_grouped(response):
        """
        Format RAG response with answer and multiple citations in a readable format.
        Ensures proper spacing for bullet points and numbered lists.
        """
        output = response['output']
        formatted_citations = ""

        def format_text(text):
            """
            Apply consistent formatting for bullet points and numbered lists.
            """
            # Add extra newline before bullets, dashes, and numbered lists
            text = re.sub(r"\n(\d+)\.", r"\n\1.", text)  # Handles numbered lists like 1., 2., 3.
            text = re.sub(r"\n([•▪-])", r"\n\1", text)   # Handles bullets like •, ▪, or -
            return text

        if 'cited_answer' in output:
            answer = format_text(output['cited_answer']['answer'])
            citations = output['cited_answer']['citations']
            docs = output['docs']

            if citations and len(citations) > 0:
                for idx, citation_idx in enumerate(citations):
                    if citation_idx < len(docs):
                        doc = docs[citation_idx]
                        citation = format_text(doc.page_content.replace('\n', ' ').strip())

                        source_snippet = doc.metadata.get('source', 'Unknown source')
                        date_retrieved = source_snippet.split(" --- ")[0].replace("_", "/")
                        source = source_snippet.split(" --- ")[1]
                        page = doc.metadata.get('pageNum', 'Unknown page')
                        similarity_score = doc.metadata.get('similarity_score', None)
                        score_display = f" (Similarity: {similarity_score:.2%})" if similarity_score is not None else ""

                        formatted_citations += f"\n📌 <span style='background-color:#f7dc6f; color:black; padding:3px;'><b>Reference {idx + 1}{score_display}</b></span> \n"
                        formatted_citations += f"@Tag4DateRetrieved@: {date_retrieved}\n"
                        formatted_citations += f"@Tag4Source@: {source}\n"
                        formatted_citations += f"@Tag4pagenum@: {page}\n"
                        formatted_citations += f"@Tag4Citation@: {citation}\n\n"
                        formatted_citations += "=" * 75
                        formatted_citations += "\n\n"
                
                return answer, formatted_citations.strip()
            
            return answer, formatted_citations
        else:
            answer = format_text(output['answer'])
            return answer, formatted_citations

    # Initialize and use the chain
    chain = create_rag_chain_with_score(
        vectorstore=vectorstore,
        llm_model=model
    )

    # Get and format response
    response = get_response(
        chain=chain,
        question=text,
        chat_history=chat_history
    )

    format_response = format_rag_response_new_grouped(response)
    
    print(format_response)

    return format_response