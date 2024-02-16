import os
import qdrant_client
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain,LLMChain, RetrievalQA

# Set page configuration for Streamlit
st.set_page_config(page_title="Chat with the Chat Bot",
                   page_icon="🤖",
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)

# Title for the app
st.title("Welcome To GSM infoBot")

# Initialize the chat messages history
if "messages" not in st.session_state.keys():
  st.session_state.messages = [{
      "role":
      "assistant",
      "content":
      "Need Info? Ask Me Questions about GSM Mall's Features"
  }]

os.environ['OPENAI_API_KEY'] = st.secrets.openai_key
QDRANT_HOST = st.secrets.QDRANT_HOST
QDRANT_API_KEY = st.secrets.QDRANT_API_KEY

@st.cache(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        client = qdrant_client.QdrantClient(
            url=QDRANT_HOST,
            api_key=QDRANT_API_KEY,
        )
        embeddings = OpenAIEmbeddings()
        vector_store = Qdrant(
            client=client,
            collection_name="gsm_demo.0.0.4",
            embeddings=embeddings
        )
        print("Connection established!")
        return vector_store


prompt_template = """
Your friendly assistant is here to help! Remember, always provide clear, concise, and friendly responses within 10-15 words. value User time and aim to provide clear and concise responses. Maintain a positive and professional tone. Encourage users to visit the store subtly, without being pushy. Dont hallucinate. Let's make every interaction a delightful experience! 😊

You will be given a context of the conversation made so far followed by a customer's question,
give the answer to the question using the context.
The answer should be short, straight and to the point. If you don't know the answer, reply that the answer is not available.

Context: {context}

Question: {question}
Answer:"""

PROMPT = PromptTemplate(
template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = { "prompt" : PROMPT }
#build your LLM
llm = ChatOpenAI(temperature=0)

template = (
                """Combine the chat history and follow up question into
                a standalone question.
                If chat hsitory is empty, use the follow up question as it is.
                Chat History: {chat_history}
                Follow up question: {question}"""
            )
        # TRY TO ADD THE INPUT VARIABLES
prompt = PromptTemplate.from_template(template)
        # question_generator_chain = LLMChain(llm=llm, prompt=prompt
if "chat_engine" not in st.session_state.keys():
  vectore_store = load_data()
  dr = vectore_store.as_retriever()
  # Initialize the chat engine
  st.session_state.chat_engine = ConversationalRetrievalChain.from_llm(
            llm = llm,
            chain_type = "stuff",
            memory = ConversationSummaryMemory(llm = llm, memory_key='chat_history', input_key='question', output_key= 'answer', return_messages=True),
            retriever = dr,
            condense_question_prompt = prompt,
            return_source_documents=False,
            combine_docs_chain_kwargs=chain_type_kwargs,
        )

if prompt := st.text_input("Your question"):
  # Prompt for user input and save to chat history
  st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
  # Display the prior chat messages
  with st.chat_message(message["role"]):
    st.write(message["content"])

# If the last message is not from the assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
  with st.chat_message("assistant"):
    with st.spinner("Thinking..."):
      response = st.session_state.chain(
                {"context": st.session_state.chat_engine.memory.buffer, "question": customer_prompt}, return_only_outputs=True)
      st.write(response.response)
      message = {"role": "assistant", "content": response.response}
      st.session_state.messages.append(
          message)  # Add response to message history
