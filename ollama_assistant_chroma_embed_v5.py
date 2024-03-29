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

# model_local =ChatOllama(model='stablelm-zephyr')
model_local =ChatOllama(model='openhermes')
# model_local =ChatOllama(model='mistral')

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

        after_rag_template = """"You are a helpful AI bot named David. when you recieve English content you answer in English, 
                                when you recieve Turkish content you answer in Turkish. 
                                Here is our chat history :{context}. Dont mention about chat history if not necessary. 
                                If context doesn't specify about the topic please continue the last conversation.
                                Give short answers that no more than 4 sentences.,
                                "human", "Hello, how are you doing?"
                                "ai", "I'm doing well, thanks!"
                                Question: {question}
        """
        after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
        after_rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | after_rag_prompt
            | model_local
            | StrOutputParser()

        )
        output=after_rag_chain.invoke(input)
        print('\033[92m'+"ai answer: ",output)

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        things_to_split=str("user said: "+input+"\n"+"AI said: "+output+"\n")
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

        return output
    
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

            # Ses verisini metne cevir
            user_input = recognizer.recognize_whisper(audio_data)
            # user_input = recognizer.recognize_sphinx(audio_data)
            # user_input = recognizer.recognize_google(audio_data)
            print('\033[94m'+"Kullanici Girdisi:", user_input)
            
            # ai_name = "jarvis"
            ai_name = "david"
            kapat=1
            if ai_name in user_input.lower():

                # hello boss 
                voice.say("Hello! I am listening")
                voice.setProperty('rate', 145)  # speed of reading 145 lower number slower speech

                voice.runAndWait()
                # tekrar dinlemeye başla 

                beep()
                while kapat == 1:
                    
                    with sr.Microphone() as source:
                        recognizer.adjust_for_ambient_noise(source)  # Calibrate the recognizer
                        print("Listening to answer.")
                        audio_data = recognizer.listen(source)
                        user_input = recognizer.recognize_whisper(audio_data)
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
            if detect_language(user_input) in ["tr", "en"]:

                response=get_ai_response_w_db(remove_non_ascii_characters(user_input))
            else:
                response=("Dil anlaşılamadı")
                print(response)
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

        else:
            break