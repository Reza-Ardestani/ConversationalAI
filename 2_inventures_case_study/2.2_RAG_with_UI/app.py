#initial files
import streamlit as st

st.title("RAG Bot")
with st.chat_message("assistant"):
  st.write("Hello! Please wait till I spin up 🔄 ...")

if 'initialized' not in st.session_state:
  # python does not have block-level scope within conditional
  # statements (if, for, while, etc.). Instead, Python uses function-level
  # scope. This means that variables defined inside an if statement
  # are still accessible outside of that statement, provided they are
  # within the same function or global scope. But, streamlit only remembers
  # the st.session_state in subsequent runs. That's why we need to store
  # rag_chain in state
  from mylibs import *
  st.session_state.initialized = True
  # Initialize chat history
  if "messages" not in st.session_state:
    st.session_state.messages = []

  doc_text = [] # loading the txt files into a list of strings
  for i in range(1,187):
    with open(f'./sessions_txt/{i}.txt', 'r') as file:
      data = file.read()
      doc_text.append(data)

  raw_documents = []
  for i in range(len(doc_text)):
    raw_documents.append(Document(page_content=doc_text[i], metadata={"source": "local", 'id':i+1}))

  embd = OpenAIEmbeddings()
  model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", max_tokens=4096)
  leaf_texts = doc_text
  # unpickle results.pkl into results
  with open("results.pkl", "rb") as f:
      results = pickle.load(f)
  # Initialize all_texts with leaf_texts
  all_texts = leaf_texts.copy()

  # Iterate through the results to extract summaries from each level and add them to all_texts
  for level in sorted(results.keys()):
    # Extract summaries from the current level's DataFrame
    summaries = results[level][1]["summaries"].tolist()
    # Extend all_texts with the summaries from the current level
    all_texts.extend(summaries)

  # Now, use all_texts to build the vectorstore with Chroma
  vectorstore = Chroma.from_texts(texts=all_texts, embedding=embd)
  retriever = vectorstore.as_retriever()

  # Prompt
  prompt = hub.pull("rlm/rag-prompt")


  # Post-processing (joining all docs retrieved after vector search)
  def format_docs(docs):
      return "\n\n".join(doc.page_content for doc in docs)

  # Chain
  st.session_state.rag_chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt
      | model
      | StrOutputParser()
  )
else:
  # Retrieve the chain from session state
  rag_chain = st.session_state.rag_chain


# Display chat messages from history on app return
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


# React to user input
if prompt := st.chat_input("What's up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({'role': "user", "content": prompt})

    # Invoke the RAG chain to get the response
    ans = rag_chain.invoke(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(ans)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant",
                                      "content": ans})
