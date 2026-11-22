import inspect
from collections import namedtuple

if not hasattr(inspect, "getargspec"):
    _ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    def _getargspec(func):
        spec = inspect.getfullargspec(func)
        return _ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = _getargspec

import sqlite3
from sqlite3 import Error
import pymorphy2
import tokenize_uk as tok


SUBCORPUS_SIZE = 1000
NUM_SUBCORPORA = 20
MIN_TOTAL_TOKENS = SUBCORPUS_SIZE * NUM_SUBCORPORA  # 20 000 слововживань


class Word:
    s_morph = pymorphy2.MorphAnalyzer(lang="uk")

    def __init__(self, form: str, lemma: str, pos: str, freq: int, sub_freqs: list[int]):
        self._form = form
        self._lemma = lemma
        self._pos = pos
        self._freq = freq
        self._sub_freqs = sub_freqs  # список із 20 частот по підвибірках №1..№20

    @staticmethod
    def from_token(token: str, freq: int, sub_freqs: list[int]):
        parsed = Word.s_morph.parse(token)[0]
        pos = parsed.tag.POS if parsed.tag.POS else "UNKN"
        return Word(token, parsed.normal_form, pos, freq, sub_freqs)

    def show(self):
        print(
            f"Словоформа: {self._form} | Лема: {self._lemma} | "
            f"Частина мови: {self._pos} | Частота: {self._freq}"
        )

    def get_tuple(self) -> tuple:
        return (self._form, self._lemma, self._pos, self._freq, *self._sub_freqs)


class Sample:
    def __init__(self, text: str, max_tokens: int = MIN_TOTAL_TOKENS):
        self._text = text
        self._tokens = tok.tokenize_words(self._text)
        all_words = [s.lower() for s in self._tokens if s and s[0].isalpha()]

        if len(all_words) < max_tokens:
            raise ValueError(
                f"Після токенізації отримано {len(all_words)} слововживань, "
                f"а потрібно мінімум {max_tokens} (20 підвибірок по 1000)."
            )

        # Беремо рівно 20 000 слововживань (або інший заданиий обсяг)
        self._words_list = all_words[:max_tokens]

        # Ділимо на 20 підвибірок по 1000 слововживань
        self._subcorpora = [
            self._words_list[i * SUBCORPUS_SIZE : (i + 1) * SUBCORPUS_SIZE]
            for i in range(NUM_SUBCORPORA)
        ]

    def get_words(self) -> list[Word]:
        words_set = set()
        words = []

        # Попередньо рахуємо частоти у кожній підвибірці для швидшої роботи
        sub_counters = []
        for sub in self._subcorpora:
            counts = {}
            for w in sub:
                counts[w] = counts.get(w, 0) + 1
            sub_counters.append(counts)

        for word_str in self._words_list:
            if word_str in words_set:
                continue
            words_set.add(word_str)

            sub_freqs = [cnt.get(word_str, 0) for cnt in sub_counters]
            total_freq = sum(sub_freqs)
            words.append(Word.from_token(word_str, total_freq, sub_freqs))

        return words

    def show(self, words: list[Word], limit: int = 20):
        for word in words[:limit]:
            word.show()


class SQL:
    def __init__(self, db_file: str):
        self._connection = SQL.create_connection(db_file)
        self._cursor = self._connection.cursor()

    @staticmethod
    def create_connection(db_file: str):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
        except Error as e:
            print(e)
        return conn

    def create_word_freq_table(self):
        sub_cols_def = ",\n                ".join(
            f"p{i} INTEGER" for i in range(1, NUM_SUBCORPORA + 1)
        )
        create_sql = f"""
            DROP TABLE IF EXISTS WordFreq;
            CREATE TABLE WordFreq (
                word_id INTEGER PRIMARY KEY AUTOINCREMENT,
                form TEXT,
                lemma TEXT,
                pos TEXT,
                freq INTEGER,
                {sub_cols_def}
            );
        """
        self._cursor.executescript(create_sql)
        self._connection.commit()

    def insert_into_freq_table(self, word: Word):
        placeholders = ", ".join(["?"] * (4 + NUM_SUBCORPORA))
        sub_cols_names = ", ".join(f"p{i}" for i in range(1, NUM_SUBCORPORA + 1))
        insert_sql = f"""
            INSERT INTO WordFreq (form, lemma, pos, freq, {sub_cols_names})
            VALUES ({placeholders});
        """
        self._cursor.execute(insert_sql, word.get_tuple())

    def generate_word_freq_table(self, words: list[Word]):
        self.create_word_freq_table()
        for word in words:
            self.insert_into_freq_table(word)
        self._connection.commit()

    def build_summary_tables(self):
        """Формує БД_ЧС_лем та БД_ЧС_ЧМ на основі WordFreq."""
        sum_cols_sql = ",\n                ".join(
            f"SUM(p{i}) AS p{i}" for i in range(1, NUM_SUBCORPORA + 1)
        )
        script = f"""
            DROP TABLE IF EXISTS LemmaFreq;
            CREATE TABLE LemmaFreq AS
            SELECT
                lemma,
                SUM(freq) AS abs_freq,
                {sum_cols_sql}
            FROM WordFreq
            GROUP BY lemma
            ORDER BY abs_freq DESC;

            DROP TABLE IF EXISTS PosFreq;
            CREATE TABLE PosFreq AS
            SELECT
                pos,
                SUM(freq) AS abs_freq,
                {sum_cols_sql}
            FROM WordFreq
            GROUP BY pos
            ORDER BY abs_freq DESC;
        """
        self._cursor.executescript(script)
        self._connection.commit()

    def select_pos_freq(self):
        query = """
            SELECT pos, SUM(freq) AS total_occurrences, COUNT(word_id) AS unique_forms
            FROM WordFreq
            GROUP BY pos
            ORDER BY total_occurrences DESC;
        """
        rows = self._cursor.execute(query).fetchall()
        for row in rows:
            print(row)

    def close(self):
        if self._connection:
            self._connection.close()


if __name__ == "__main__":
    datasets = [
        ("data/text1.txt", "data/freq_text1.db"),
        ("data/text2.txt", "data/freq_text2.db"),
    ]

    for txt_path, db_path in datasets:
        text_content = None
        for enc in ("utf-8", "cp1251", "utf-16"):
            try:
                with open(txt_path, "r", encoding=enc) as f:
                    text_content = f.read()
                break
            except UnicodeDecodeError:
                text_content = None

        if text_content is None:
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                text_content = f.read()

        sample = Sample(text_content)
        words_list = sample.get_words()
        sample.show(words_list, limit=10)

        db = SQL(db_path)
        db.generate_word_freq_table(words_list)
        db.build_summary_tables()
        print(f"--- Статистика частин мови для {db_path} ---")
        db.select_pos_freq()
        db.close()