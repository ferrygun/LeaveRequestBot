#!/Users/ferry.djaja/anaconda3/bin/python

from flask import Flask, request, jsonify
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ChatVectorDBChain
from langchain.chat_models import AzureChatOpenAI

from langchain.memory import (
  ConversationBufferMemory, 
  ReadOnlySharedMemory, 
  ConversationSummaryMemory, 
  ConversationBufferWindowMemory, 
  ConversationSummaryBufferMemory, 
  ConversationEntityMemory,
  ReadOnlySharedMemory
)

import json
import os

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = ""
os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_KEY"] = ""

app = Flask(__name__)

class Session:
    def __init__(self, user_id):
        self.user_id = user_id
        print("user_id: ", user_id)
        self.embedding_model = OpenAIEmbeddings(chunk_size=10)

        print("--Load File--")
        self.recipe_1 = TextLoader('Leave_Procedure.txt').load()
        self.text_splitter_1 = CharacterTextSplitter(chunk_overlap=100)
        self.recipe_1_content = self.text_splitter_1.split_documents(self.recipe_1)


        user_data = f'leave_{user_id}.txt'
        self.recipe_2 = TextLoader(user_data).load()
        self.text_splitter_2 = CharacterTextSplitter(chunk_overlap=100)
        self.recipe_2_content = self.text_splitter_2.split_documents(self.recipe_2)


        self.faiss_db1 = FAISS.from_documents(self.recipe_1_content, self.embedding_model)  
        self.faiss_db2 = FAISS.from_documents(self.recipe_2_content, self.embedding_model)  

        self.faiss_db1.merge_from(self.faiss_db2)

        self.retriever = self.faiss_db1.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        self.llm = AzureChatOpenAI(
            temperature=0,
            deployment_name="gpt-4",
        )

        #self.memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True, output_key="answer")
        self.memory = ConversationSummaryMemory(llm=self.llm, memory_key="chat_history", input_key="question", return_messages=True)
        #self.memory = ConversationSummaryBufferMemory(llm=self.llm, memory_key="chat_history", input_key="question", return_messages=True,  max_token_limit=650)


        self.prompt_template = """
        You are an assistive chatbot designed to assist users with inquiries related to the leave application process. 
        In this conversation, a user is seeking information on how to request leave, and they may have various leave dates and they should provide the following details:

        1. The specific period or duration of their intended leave. The leave period must not be overlapped each other.
        2. The destination country where they plan to travel.
        3. Whether they require a visa letter for their trip.
        4. Whether they need assistance in configuring their out-of-office message.
        5. Whether they need to delegate the approval process to someone else.

        Follow exactly these steps:
        1. Carefully review and analyze the provided context below.
        2. Formulate your response exclusively using the information available in the context.
        3. Obtain the user's leave dates and record them as "fromDate" and "toDate." If you do not have this information, kindly seek clarification from the user.
        4. If all leave dates (fromDate and toDate) are dated in the past relative to today's date, request the user to make the necessary corrections.
        4. Get the existing leave dates (fromDate and toDate) and put in the "existing_leaves".
        5. Exercise careful scrutiny to identify any overlapping trip dates. If any are found, prompt the user to make adjustments. 
        6. If the user has not specified the country, kindly request this information from the user.
        7. If the user requires a visa letter, record this information in "visa_letter", set the flag to "Y".
        8. If the user requires assistance with setting up an out-of-office message, record this request in "setup_ooo", set the flag to "Y"
        9. If the user intends to delegate their approval to another person, request the email address of the designated delegate and record it in "delegate_to." Note that the email address should feature the domain "@ada.com". Capture the provided email address in "delegate_to".
        10. Ensure that you obtain answers to ALL follow-up questions referenced in the context. If any remain unanswered, continue to prompt the user for responses. Do not introduce new inquiries.
        11. Once you have obtained all the required information, respond to the user with the statement, "Thank you, I have gathered all the necessary information and will proceed with the next steps. No further questions will be asked." Set the "complete" flag to true.
        12. If user says remove/cancel leave or anything like this, record this information in "op" with "remove", set "complete" to true. Update "existing_leaves"
        13. If user says add leave or anything like this, record this information in "op" with "add" and set N/A for "fromDate_old" & "toDate_old". Update "existing_leaves"
        14. if user says update leave or anything like this, record this information in "op" with "update" and fill in the "fromDate_old" and "toDate_old" with the old leave dates. Update "existing_leaves"
        15. Avoid using the following response: "As an assistive chatbot, I don't have personal plans or intentions."
        16. In instances where you do not possess the answer to a question, it is appropriate to acknowledge that by stating, "I do not have that information." Do not attempt to fabricate a response. If the user's question is unrelated to the context, politely inform them that you are focused solely on addressing context-related questions, and strive to provide detailed responses.


        Today's date: 
        {today_date}

        Chat History: 
        {chat_history}
                
        User Question: 
        {question}

        Context: 
        {context}

        Respond to the user using JSON format and include the following key-value pairs:
        - answer, your answer should also mention the follow up question if any (ask one at a time) (i.e. answer)
        - country ID (i.e. country_ID: Indonesia)
        - intent (i.e. intent: leave)
        - user leaves date from and to: (ie. fromDate, toDate) in this format DD/MM/YYYY. Make sure no overlap    
        - visa letter (ie. visa_letter: Y or N), if you don't know the answer put N/A.
        - need to setup office (ie. setup_ooo: Y or N), if you don't know the answer put N/A.
        - delegate their approval (ie. delegate_to: someone@ada.com or No), if you don't have the answer put N/A.
        - complete (ie. true or false) only true if all information is captured by chatbot or set false if user requested to cancel leave.
        - op (ie. remove/add/update)
        - fromDate_old, toDate_old
        - existing_leaves with country_ID, delegate_to, setup_office, visa_letter, fromDate and toDate

        """

        self.QA_PROMPT = PromptTemplate(
            template=self.prompt_template, input_variables=['context', 'question', 'chat_history', 'today_date']
        )

        self.qa = ConversationalRetrievalChain.from_llm(llm=self.llm, 
                                                       chain_type="stuff",
                                                       retriever=self.retriever, 
                                                       memory=self.memory, 
                                                       combine_docs_chain_kwargs={"prompt": self.QA_PROMPT},
                                                       verbose=False)

        # Flag to track the first iteration
        self.first_iteration = True
        self.last_answer = None
        self.chat_history = []


sessions = {}

@app.route('/greet', methods=['GET'])
def greet():
    return jsonify({'message': 'Hi, I am the Tango Bot!'})

@app.route('/reset', methods=['POST'])
def reset_session():
    data = request.get_json()
    user_id = data.get('user_id', None)

    if user_id is None:
        return jsonify({'error': 'user_id is required'}), 400

    if user_id in sessions:
        del sessions[user_id]  # Remove the session for the given user_id

    return jsonify({'message': 'Session reset successfully'}), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    print(data)

    data = data["userMessage"]  

    user_id = data.get('user_id', None)
    if user_id is None:
        return jsonify({'error': 'user_id is required'}), 400
    
    if user_id not in sessions:
        sessions[user_id] = Session(user_id) # Pass user_id when creating the Session instance

    session = sessions[user_id]


    question = data.get('question', None)
    if question is None:
        return jsonify({'error': 'question is required'}), 400

    today_date = '4 September 2023'
    print("Session status: ", session.first_iteration)

    # Check if it's the first iteration
    if session.first_iteration:
        chat_history = []  # For the first iteration, initialize chat_history as empty
        print(chat_history)
        result = session.qa({"question": question, "chat_history": chat_history, "today_date": today_date})
    else:
        chat_history = [(question, session.last_answer)]  # For subsequent iterations, use previous answer
        print(chat_history)
        result = session.qa({"question": question, "chat_history": chat_history, "today_date": today_date})

    # Update chat_history with the result
    chat_history.append((question, result["answer"]))
    session.chat_history = chat_history
    session.last_answer = result["answer"]

    session.first_iteration = False

    print(result["answer"])

    parsed_data = result["answer"]
    return jsonify(parsed_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
