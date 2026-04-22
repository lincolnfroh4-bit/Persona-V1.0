import locale
import json
import os
from pathlib import Path


I18N_DIR = Path(__file__).resolve().parent / "i18n"


def load_language_list(language):
    with (I18N_DIR / f"{language}.json").open("r", encoding="utf-8") as f:
        language_list = json.load(f)
    return language_list


class I18nAuto:
    def __init__(self, language=None):
        if language in ["Auto", None]:
            language = locale.getdefaultlocale()[
                0
            ]  # getlocale can't identify the system's language ((None, None))
        if not (I18N_DIR / f"{language}.json").exists():
            language = "en_US"
        self.language = language
        # print("Use Language:", language)
        self.language_map = load_language_list(language)

    def __call__(self, key):
        return self.language_map.get(key, key)

    def print(self):
        print("Use Language:", self.language)
