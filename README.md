# Ollama-voice-assistant-with-chroma-db-memory


pip install langchain langchain-community langchain-core
pip install chromadb


speech_recognition pyttsx3 openai platform unicodedata

pip install langdetect

ollama pull nomic-embed-text
ollama run mistral 

activate glados

cd C:\Users\E.C.E\anaconda3\envs\GLaDOS\ck

notes for turkish tts:
eeozcan

1
17 Oca
Soru 3. Sene Önce Sorulmuş Ama Çözümün Türkçe Kaynağı Olmadığı İçin Paylaştım Herkese Kolay Gelsin.
Öncelikle Ayarlar > Zaman ve Dil > Konuşma Bölümünden Türkçe Ses(Tolga) Paketini İndir.
Ardından Win + r İle Çalıştır Penceresini Aç > Regedit Yaz > Enter > HKEY_LOCAL_MACHINE/SOFTWARE/WOW6432Node/Microsoft/ Speech_OneCore/Voices/Tokens Dizinine Git Burada Yüklediğin Sesler Görünecek > Ses Dosyasına Sağ Tıkla > Ver’e Tıkla > Dosyayı Herhangi Bir Yere Kaydet > Bu Dosyayı Notepad İle Aç > Ve En Önemlisi Burada

[HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\Microsoft\Speech_OneCore\Voices\Tokens\MSTTS_V110_trTR_Tolga]
Bu Bölümü
[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MSTTS_V110_trTR_Tolga] Bununla Değiştir
[HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\Microsoft\Speech_OneCore\Voices\Tokens\MSTTS_V110_trTR_Tolga\Attributes]
Bu Bölümü
[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MSTTS_V110_trTR_Tolga\Attributes] Bununla Değiştir
Sonra Kaydet Çık > Ve Bu Dosyaya Tıkla, Herşeye Evet Diyip Yükle > Başarılı Mesajını Aldığında > print(engine.getProperty(‘voices’)) Artık Burada Ses Dosyasını Görebilirsin.

Aslında Genel Olarak Yaptığımız Şu. İndirilen Seslerin Hepsini 3. Parti Uygulamalarda Kullanamıyorsun Microsoftun 3. Parti Uygulamalara İzin Verdiği Sesler HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\ Bölümünde Bulunur. Bizde Ses Paketini 3. Parti Olarak Kullanacağımız İçin Sesi İndirim 3.Parti Sesler Bölümüne Yükledik.
