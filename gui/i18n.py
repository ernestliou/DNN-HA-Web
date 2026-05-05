import os
import json
import locale

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCALES_DIR = os.path.join(project_root, "locale")
translations = {}

for lang_code in ["zh_TW", "en_US"]:
    loc_file = os.path.join(LOCALES_DIR, f"{lang_code}.json")
    if os.path.exists(loc_file):
        with open(loc_file, "r", encoding="utf-8") as f:
            translations[lang_code] = json.load(f)

if "en_US" not in translations:
    translations["en_US"] = {} # Fallback

try:
    sys_lang, _ = locale.getdefaultlocale()
    if sys_lang and sys_lang in translations:
        default_lang = sys_lang
    else:
        default_lang = "en_US"
except:
    default_lang = "en_US"

def get_i18n(lang=None):
    if lang is None:
        lang = default_lang
    return translations.get(lang, translations.get("en_US", {}))
