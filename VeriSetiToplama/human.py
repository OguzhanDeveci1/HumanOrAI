import wikipediaapi
import pandas as pd
import csv
import os
import re
import time

# DAHA GENİŞ ANA KATEGORİ LİSTESİ
# Bu listeyi ne kadar geniş tutarsanız veri o kadar çeşitli olur.
main_categories = [
    "Category:Main topic classifications",  # Wikipedia'nın en üst ana dalı
    "Category:Natural sciences", "Category:History", "Category:Sociology",
    "Category:Technology", "Category:Arts", "Category:Philosophy",
    "Category:Law", "Category:Health", "Category:Economics", "Category:Geography",
    "Category:Literature", "Category:Politics", "Category:Psychology", "Category:Religion"
]

wiki_en = wikipediaapi.Wikipedia(
    user_agent='LargeDatasetCollector/2.0 (yourname@example.com)',
    language='en'
)

output_file = 'large_human_dataset.csv'

# CSV Başlıkları
if not os.path.exists(output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Title', 'Main_Category', 'Text', 'Label'])

collected_titles = set()  # Aynı makaleyi tekrar çekmemek için


def scrape_category_recursive(category_obj, main_cat_name, level=0, max_level=2):
    """Kategorinin alt dallarına sızarak makale toplar."""
    global collected_titles

    # Çok derine inip kaybolmamak için sınır (level)
    if level > max_level:
        return

    members = category_obj.categorymembers

    for member in members.values():
        # Eğer bu bir makaleyse ve daha önce çekmediysek
        if member.ns == wikipediaapi.Namespace.MAIN and member.title not in collected_titles:
            try:
                text = member.text
                text = re.split(r'==\s*(References|See also|External links|Notes)\s*==', text)[0]
                text = re.sub(r'\[\d+\]', '', text)
                text = re.sub(r'\s+', ' ', text).strip()

                words = text.split()
                if len(words) >= 450:  # Kaliteli ve uzun metinler
                    final_text = " ".join(words[:650])

                    with open(output_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([member.title, main_cat_name, final_text, 1])

                    collected_titles.add(member.title)
                    if len(collected_titles) % 10 == 0:
                        print(f"Toplam Veri: {len(collected_titles)} | Son eklenen: {member.title}")
            except:
                continue

        # Eğer bu bir alt kategoriyse, içine gir (Recursive call)
        elif member.ns == wikipediaapi.Namespace.CATEGORY:
            scrape_category_recursive(member, main_cat_name, level + 1, max_level)




for cat_name in main_categories:
    print(f"\n--- {cat_name} ve alt dalları taranıyor ---")
    cat_page = wiki_en.page(cat_name)
    if cat_page.exists():
        scrape_category_recursive(cat_page, cat_name)