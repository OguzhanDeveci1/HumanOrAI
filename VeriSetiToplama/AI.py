import pandas as pd
import google.generativeai as genai
import csv
import os
import time

# 1. API Ayarlarını Yapın
genai.configure(api_key="AIzaSyA1yHWg1mcp-_I1b9EL5UAd0D62ILVIqSs")  # Buraya kendi API anahtarınızı girin
model = genai.GenerativeModel('gemini-2.0-flash')

# 2. Dosya Yolları
input_file = 'large_human_dataset.csv'
output_file = 'large_ai_generated_dataset.csv'

# 3. Çıktı Dosyası Hazırlığı (Eğer yoksa oluştur)
if not os.path.exists(output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Title', 'Main_Category', 'Text', 'Label'])

# 4. İnsan Verisini Oku
df_human = pd.read_csv(input_file)

print(f"AI üretimi başladı. Hedef: {len(df_human)} makale.")

# 5. Üretim Döngüsü
for index, row in df_human.iterrows():
    title = row['Title']

    # Prompt: Kesinlikle ekstra açıklama istemediğimizi belirtiyoruz
    prompt = f"""Write a professional, academic-style article about '{title}'. 
    The article should be between 500-700 words long. 
    Focus on technical depth and use a formal tone. 
    Avoid clichés like 'In conclusion' or 'In today's world'. 
    IMPORTANT: Provide ONLY the article text itself. Do not include any introductory remarks, conversational fillers, or headers like 'Here is the article'."""

    try:
        # API Çağrısı
        response = model.generate_content(prompt)
        ai_text = response.text.strip()

        # Anlık Kayıt
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([title, row['Main_Category'], ai_text, 0])  # Label 0: AI

        if (index + 1) % 10 == 0:
            print(f"Tamamlanan: {index + 1} / {len(df_human)}")

        # API Rate Limit (Ücretsiz kota kullanıyorsanız mola vermekte fayda var)
        time.sleep(0.1)

    except Exception as e:
        print(f"Hata ({title}): {e}")
        time.sleep(5)  # Hata durumunda biraz daha uzun bekle