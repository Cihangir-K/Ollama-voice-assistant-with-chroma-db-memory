import speech_recognition as sr
import pyttsx3
import os
from openai import OpenAI
import datetime
import platform
import unicodedata

from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

from langdetect import detect

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.tools import DuckDuckGoSearchRun

# faster whisper implementation
from faster_whisper import WhisperModel
import time


wake_word = 'jarvis'
# wake_word = 'pudding'

listening_for_wake_word = True

whisper_size ='tiny'
num_cores = os.cpu_count()
whisper_model = WhisperModel(
    whisper_size,
    device = 'cpu',
    compute_type='int8',
    cpu_threads=num_cores,
    num_workers=num_cores

)
def wav_to_text(audio_path):
    segments, _ =whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text



def detect_language(text):
    try:
        language = detect(text)
        return language
    except:
        return "Unknown"

    # # Example usage
    # sentence = "Merhaba, nasılsınız?"
    # language = detect_language(sentence)
    # print("Detected language:", language)


def remove_non_ascii_characters(s):
    """
    Remove non-ASCII characters from the string.
    """
    return ''.join(c for c in s if unicodedata.category(c)[0] != 'C')


# HEADER = '\033[95m'
# MAVI = '\033[94m'
# YESIL = '\033[92m'
# SARI = '\033[93m'
# KIRMIZI = '\033[91m'
# BEYAZ = '\033[0m'
# BOLD = '\033[1m'
# UNDERLINE = '\033[4m'


client = OpenAI(
                base_url = 'http://localhost:11434/v1',
                api_key='ollama',
    )


# model_local =ChatOllama(model='cas/minicpm-3b-openhermes-2.5-v2') 
# model_local =ChatOllama(model='stablelm-zephyr')
model_local =ChatOllama(model='dolphin-llama3')
# model_local =ChatOllama(model='openhermes') 
# model_local =ChatOllama(model='mistral')
# model_local =ChatOllama(model='experiment26')
# model_local =ChatOllama(model='gemma')
# model_local =ChatOllama(model='gemma:2b-instruct')
# model_local =ChatOllama(model='llava')
# model_local =ChatOllama(model='adrienbrault/nous-hermes2pro:Q5_K_S')
# model_local =ChatOllama(model='gemma:instruct')



#google gemini api 
# model_local = "gemini"

# safety_settings = [
#   {
#     "category": "HARM_CATEGORY_HARASSMENT",
#     "threshold": "BLOCK_NONE"
#   },
#   {
#     "category": "HARM_CATEGORY_HATE_SPEECH",
#     "threshold": "BLOCK_NONE"
#     # "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#   },
#   {
#     "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#     "threshold": "BLOCK_NONE"
#   },
#   {
#     "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#     "threshold": "BLOCK_NONE"
#   },
# ]

if(model_local == "gemini"):
    model_local = ChatGoogleGenerativeAI(
                                model="gemini-pro", 
                                temperature=0.1, 
                                convert_system_message_to_human=True,
                                safety_settings= None
                            )
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

chat_history = []

file = open('C:/Users/E.C.E/anaconda3/envs/GLaDOS/ck/history.txt', 'a', encoding="utf-8")
now = datetime.datetime.now()
file.write(str(now)+"\n")




#*****************************************************************

def beep():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 500)  # frequency in Hz and duration in milliseconds
    else:
        # For Unix/Linux-based systems
        os.system("echo -n '\a';sleep 0.2;echo -n '\a'")  # You may adjust sleep duration
# beep()


# Yapay zeka uygulamasina erismek icin kullanilacak fonksiyon aynı zamanda RAG işini de yapan kodlar burada
def get_ai_response_w_db(input):
    try:
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        output=""

        # things_to_split=str(input)
        things_to_split=str(remove_non_ascii_characters(input))
        # print("things_to_split: ",things_to_split)
        docs_splits = text_splitter.split_text(things_to_split)
        docs_splits2 = text_splitter.create_documents(docs_splits)
        # print("docs_splits:", docs_splits)

        #2. Convert documents to Embeddings and store them
        vectorestore = Chroma.from_documents(
                documents=docs_splits2,
                collection_name="rag-chroma",
                embedding=embeddings.ollama.OllamaEmbeddings(model="nomic-embed-text"),persist_directory="./chroma_db",
        )
        retriever = vectorestore.as_retriever()
        # print("retriever: ",retriever)


        #old rag chains, its without chat history

        # after_rag_template = """"You are a helpful AI bot named David. when you recieve English content you answer in English, 
        #                         when you recieve Turkish content you answer in Turkish. 
        #                         Here is our chat history :{context}. Dont mention about chat history if not necessary. 
        #                         If context doesn't specify about the topic please continue the last conversation.
        #                         Give short answers that no more than 4 sentences.,
        #                         "human", "Hello, how are you doing?"
        #                         "ai", "I'm doing well, thanks!"
        #                         Question: {question}
        # """
        # after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
        # after_rag_chain = (
        #     {"context": retriever, "question": RunnablePassthrough()}
        #     | after_rag_prompt
        #     | model_local
        #     | StrOutputParser()

        # )

        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        contextualize_q_chain = contextualize_q_prompt | model_local | StrOutputParser()



        contextualize_q_chain.invoke(
            {
                "chat_history": [
                    HumanMessage(content="What does LLM stand for?"),
                    AIMessage(content="Large language model"),
                ],
                "question": "What is meant by large",
            }
        )


        qa_system_prompt = """You are an helpful ai assistant for question-answering tasks. \
        If you know the answer, answer the question.\
        If you don't know the answer try to use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        If it's not in context you don't need to stick to provided context. \
        Use three sentences maximum and keep the answer concise.\
        {context}"""



        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        
        def contextualized_question(input: dict):
            if input.get("chat_history"):
                return contextualize_q_chain
            else:
                return input["question"]


        rag_chain = (
            RunnablePassthrough.assign(
                context=contextualized_question | retriever 
            )
            | qa_prompt
            | model_local
        )    
        
        # chat_history = []   #this part is taken out of the while loop 
        question = input
        output = rag_chain.invoke({"question": question, "chat_history": chat_history})
        chat_history.extend([HumanMessage(content=question), output])



        print('\033[92m'+"ai answer: ",output.content)

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        # print("came to split")
        # print("output.content: ",output.content)
        things_to_split=str("user said: "+input+"\n"+"AI said: "+output.content+"\n")
        # print("things_to_split: ",things_to_split)

        docs_splits = text_splitter.split_text(things_to_split)
        docs_splits2 = text_splitter.create_documents(docs_splits)
        # print("docs_splits:", docs_splits)

        #2. Convert documents to Embeddings and store them

        vectorestore = Chroma.from_documents(
                documents=docs_splits2,
                collection_name="rag-chroma",
                embedding=embeddings.ollama.OllamaEmbeddings(model="nomic-embed-text"),persist_directory="./chroma_db",
        )
        retriever = vectorestore.as_retriever()

        return output.content
    
    except Exception as e:
        print("No answer from Ollama_2")
        print("Error:", e)
        return ()

# Yapay zeka uygulamasina erismek icin kullanilacak fonksiyon
def get_ai_response(input_text):

            # Initialize ollama 
        try:

            response = client.chat.completions.create(
                # model="phi",
                # model="mistral",
                model="openhermes",
                messages=[
                    {"role": "system","content": "you are a helpful ai asistant named David. Answer questions briefly, in a sentence or less. And also this is our previous conversation:" + str(chat_history)+"please dont repeat previous conversations every time."},
                    {"role": "user","content": input_text}
                ],
            )

#*******************
            if response and response.choices:
                # print('ollama:', response.choices[0].message.content)
                return response
            else:
                print("No answer from Ollama_1")
                return ()
        except Exception as e:
            print("No answer from Ollama_2")
            print("Error:", e)
            return ()


# Recognizer
recognizer = sr.Recognizer()

#(text-to-speech) 
voice = pyttsx3.init('sapi5')

voices = voice.getProperty('voices') # sesleri almak için 
voice.setProperty('voice', voices[0].id) # türkçe dil için 1 ingilizce için 0erkek ve 2bayan

chat_history = []  # Sohbet gecmisini saklamak icin bos bir liste olusturun





choise =input("Do you want voice control (answer as 'yes' or 'no'): ")

print("input is : ",choise)

if "yes" == str(choise):
    print("You can speak for voice recognition.")
        

    while True:
        try:
            # Mikrofondan ses al
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)  # Calibrate the recognizer
                print("Listened for ambient noise ...")
                # beep()
                print('\033[91m'+"Dinliyorum...")
                audio_data = recognizer.listen(source) 
                # audio_data = recognizer.listen_in_background(source)# listen'dan devşirdik

            # Ses verisini metne cevir
            # faster whisper kullanımı için audio wav olarak kaydediliyior. ardından wav_to_text ile text çıkarılıyor


            wake_audio_path = 'wake_detect.wav'
            with open(wake_audio_path, 'wb') as f:
                f.write(audio_data.get_wav_data())
                text_input = wav_to_text(wake_audio_path)
                print('text_input: ',text_input)

            user_input =text_input
            # user_input = recognizer.recognize_whisper(audio_data)
            # user_input = recognizer.recognize_sphinx(audio_data)
            # user_input = recognizer.recognize_google(audio_data)
            print('\033[94m'+"Kullanici Girdisi:", user_input)
            
            ai_name = wake_word
            # ai_name = "jarvis"
            kapat=1
            if ai_name in user_input.lower():

                # hello boss 
                voice.say("Hello! I am listening")
                voice.setProperty('rate', 145)  # speed of reading 145 lower number slower speech

                voice.runAndWait()
                # tekrar dinlemeye başla 

                # beep()
                while kapat == 1:
                    
                    with sr.Microphone() as source:
                        recognizer.adjust_for_ambient_noise(source)  # Calibrate the recognizer
                        print("Listening to answer.")
                        audio_data = recognizer.listen(source)

                        awake_audio_path = 'awake.wav'
                        with open(awake_audio_path, 'wb') as f:
                            f.write(audio_data.get_wav_data())
                            text_input = wav_to_text(awake_audio_path)
                            # print('text_input: ',text_input)

                        # user_input = recognizer.recognize_whisper(audio_data)
                        user_input =text_input
                        print('\033[93m'+"user_input ",user_input)

                        close_control= user_input.lower()
                    if "close "+ ai_name in close_control:
                        kapat=0
                        print('\033[91m'+"Kapat geldi")                    
                        voice.say("Good byee!")
                        voice.setProperty('rate', 145)  # speed of reading 145
                        voice.runAndWait()
                        pass

                    elif "exit" in user_input.lower():
                        kapat=0
                        voice.setProperty('voice', voices[0].id) # türkçe dil için 1 ingilizce için 0erkek ve 2bayan
                        voice.say("Exiting! Good byee!")
                        voice.setProperty('rate', 145)  # speed of reading 145
                        voice.runAndWait()
                        print("Exit geldi, Cikis yapiliyor...")
                        break

                    elif "thank you" in user_input.lower():
                        kapat=0
                        voice.setProperty('voice', voices[0].id) # türkçe dil için 1 ingilizce için 0erkek ve 2bayan
                        voice.say("Your welcome! Good byee!")
                        voice.setProperty('rate', 145)  # speed of reading 145
                        voice.runAndWait()
                        print('\033[91m'+"Exit geldi, Cikis yapiliyor...")
                        break
                    else:
                        kapat=1
                        # file.write(str(user_input)+"\n")
                        file.write(str(remove_non_ascii_characters(user_input))+"\n")
                        # Yapay zeka uygulamasindan yanit al

                        # response=get_ai_response_w_db(user_input)
                        if detect_language(user_input) in ["tr", "en"]:

                            response=get_ai_response_w_db(remove_non_ascii_characters(user_input))
                        else:
                            response=("Dil anlaşılamadı")
                        # print("response of ai raw:",response)
                        # print('\033[93m'+"Yapay Zeka Yaniti:", response)

                        # Sohbet gecmisine inputu ekle
                        # chat_history.append(remove_non_ascii_characters(user_input))        
                        # Sohbet gecmisine yaniti ekle

                        # chat_history.append(response)

                        try:
                            # print(str(response))
                            for_writer = remove_non_ascii_characters(response)  # Implement remove_non_ascii_characters function as needed

                            file.write(str(for_writer)+"\n")
                        except sr.RequestError as e:
                            print('\033[91m'+"Yazmada Sorun yasandi",e)
                            # print("Yazmada Sorun yasandi; {0}".format(e))  
                            pass                 


                        # Yaniti sesli olarak kullaniciya iletiyoruz
                        if detect_language(response) == "tr":

                            voice.setProperty('voice', voices[1].id) # türkçe dil için 1 ingilizce için 0erkek ve 2bayan
                            voice.say(response)
                            voice.setProperty('rate', 145)  # speed of reading 145 
                            voice.runAndWait()
                        elif detect_language(response) == "en":
                            voice.setProperty('voice', voices[0].id) # türkçe dil için 1 ingilizce için 0erkek ve 2bayan
                            voice.say(response)
                            voice.setProperty('rate', 145)  # speed of reading 145 
                            voice.runAndWait()
                        else:
                            voice.setProperty('voice', voices[0].id) # türkçe dil için 1 ingilizce için 0erkek ve 2bayan
                            voice.say(response)
                            voice.setProperty('rate', 145)  # speed of reading 145 
                            voice.runAndWait()



                        pass

            elif "exit" in user_input.lower():
                voice.say("Exiting! Good byee!")
                voice.setProperty('voice', voices[0].id) # türkçe dil için 1 ingilizce için 0erkek ve 2bayan
                voice.setProperty('rate', 145)  # speed of reading 145
                voice.runAndWait()
                print('\033[91m'+"Exit geldi, Cikis yapiliyor...")
                file.close()
                break
            
            else:
                print("not send to AI.")
        except sr.UnknownValueError:
            print("Anlasilamadi")
        except sr.RequestError as e:
            print("Sorun yasandi; {0}".format(e))
        except KeyboardInterrupt:
            print("Cikis yapiliyor...")
            file.close()
            break


else:
    print("You can write your prompt.")

    while True:
        
        input_ =input()
        if input_ != "exit":
            print('\033[93m'+input_)
            user_input = input_
            # file.write(str(user_input)+"\n")
            file.write(str(remove_non_ascii_characters(user_input))+"\n")
            # Yapay zeka uygulamasindan yanit al

            # response=get_ai_response_w_db(user_input)
            if detect_language(user_input) in ["tr", "en"]: # DİKKAT PAS GEÇİLDİ
                # print("detect_language(user_input): ",detect_language(user_input))

                response=get_ai_response_w_db(remove_non_ascii_characters(user_input))
            else:
                # print("detect_language(user_input): ",detect_language(user_input))
                # response=("Dil anlaşılamadı")
                response=get_ai_response_w_db(remove_non_ascii_characters(user_input))
                # print(response)
            # print("response of ai raw:",response)
            # print('\033[93m'+"Yapay Zeka Yaniti:", response)

            # Sohbet gecmisine inputu ekle
            # chat_history.append(remove_non_ascii_characters(user_input))        
            # Sohbet gecmisine yaniti ekle

            # chat_history.append(response)

            try:
                # print(str(response))
                for_writer = remove_non_ascii_characters(response)  # Implement remove_non_ascii_characters function as needed

                # for_writer = response

                file.write(str(for_writer)+"\n")
            except sr.RequestError as e:
                print('\033[91m'+"Yazmada Sorun yasandi",e)
                # print("Yazmada Sorun yasandi; {0}".format(e))  
                pass                 

        else:
            break