from collections import namedtuple
import inspect

if not hasattr(inspect, "getargspec"):
    _ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    def _getargspec(func):
        spec = inspect.getfullargspec(func)
        return _ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = _getargspec

import os
import numpy as np
import pymorphy2
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import tokenize_uk as tok

morph = pymorphy2.MorphAnalyzer(lang="uk")

STOP_POS = {"PREP", "CONJ", "PRCL", "INTJ", "NPRO", "NUMR", "PRED", "UNKN"}

EXTRA_STOP_WORDS = {
    "бути", "це", "цей", "той", "тога", "такий", "свій", "соб", "себе",
    "весь", "увесь", "всякий", "один", "інший", "який", "хто", "що",
    "де", "коли", "куди", "звідки", "чому", "як", "так", "там", "тут",
    "вже", "ще", "теж", "також", "просто", "можна", "треба", "мати",
    "могти", "стати", "робити", "сказати", "говорити", "тюля", "пепа",
    "до", "за", "про", "на", "в", "у", "з", "із", "зі", "від", "під",
    "по", "при", "між", "через", "без", "для", "після", "перед",
    "він", "вона", "воно", "вони", "я", "ти", "ми", "ви",
    "його", "йога", "її", "їх", "мене", "мен", "мені", "тобі", "тебе",
    "нам", "нас", "вам", "вас", "ним", "ними", "ній", "ніх", "цього", "того",
}


def resolve_path(filename: str) -> str:
    data_path = os.path.join("data", filename)
    return data_path if os.path.exists(data_path) else filename


def load_subcorpora_words(filepath: str, chunk_size: int = 1000, chunks_count: int = 20) -> list[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()
    tokens = tok.tokenize_words(raw_text)
    words = [t.lower().strip() for t in tokens if t and t.strip() and t.strip()[0].isalpha()]
    subdocs = []
    for i in range(0, min(len(words), chunk_size * chunks_count), chunk_size):
        chunk = words[i : i + chunk_size]
        if chunk:
            subdocs.append(" ".join(chunk))
    return subdocs


def load_subcorpora_lemmas(filepath: str, chunk_size: int = 1000, chunks_count: int = 20) -> list[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()
    tokens = tok.tokenize_words(raw_text)
    lemmas = []
    for token in tokens:
        word = token.lower().strip()
        if not word or not word[0].isalpha() or len(word) < 2:
            continue
        parsed = morph.parse(word)[0]
        pos = parsed.tag.POS if parsed.tag.POS else "UNKN"
        lemma = parsed.normal_form.lower().strip()
        if (
            pos in STOP_POS
            or word in EXTRA_STOP_WORDS
            or lemma in EXTRA_STOP_WORDS
            or len(lemma) < 2
        ):
            continue
        lemmas.append(lemma)

    subdocs = []
    for i in range(0, min(len(lemmas), chunk_size * chunks_count), chunk_size):
        chunk = lemmas[i : i + chunk_size]
        if chunk:
            subdocs.append(" ".join(chunk))
    return subdocs


def compute_tfidf_summary(documents: list[str], doc_names: list[str], title: str, top_n: int = 10):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    print(f"\n=== {title} ===")
    print("Розмірність матриці TF-IDF (підвибірки x терміни):", tfidf_matrix.shape)

    feature_names = vectorizer.get_feature_names_out()
    dense = tfidf_matrix.toarray()

    for idx, name in enumerate(doc_names):
        mean_scores = dense[idx * 20 : (idx + 1) * 20].mean(axis=0)
        top_pairs = [
            (word, round(float(score), 4))
            for score, word in sorted(zip(mean_scores, feature_names), reverse=True)
            if score > 0
        ][:top_n]
        print(f"Топ-{top_n} для {name}: {top_pairs}")


def cluster_subcorpora(subdocs: list[str], labels: list[str], cluster_num: int = 2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(subdocs)

    kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init=10)
    kmeans.fit(tfidf_matrix)

    print("\n=== Кластеризація підвибірок (KMeans, 2 кластери) ===")
    for cluster_id in range(cluster_num):
        indices = np.where(kmeans.labels_ == cluster_id)[0]
        t1_cnt = sum(1 for i in indices if labels[i].startswith("Т1_"))
        t2_cnt = sum(1 for i in indices if labels[i].startswith("Т2_"))
        print(f"Cluster {cluster_id + 1}: {len(indices)} підвибірок (Т1: {t1_cnt}, Т2: {t2_cnt})")
        print("  ", ", ".join(labels[i] for i in indices))
        print("-" * 40)


if __name__ == "__main__":
    p1 = resolve_path("text1.txt")
    p2 = resolve_path("text2.txt")

    raw_1 = load_subcorpora_words(p1, chunk_size=1000, chunks_count=20)
    raw_2 = load_subcorpora_words(p2, chunk_size=1000, chunks_count=20)

    # 1. Спочатку TF-IDF по словоформах (без чистки / зі стоп-словами)
    compute_tfidf_summary(
        raw_1 + raw_2,
        ["Текст 1 (20 підвибірок)", "Текст 2 (20 підвибірок)"],
        "TF-IDF по словоформах (до лематизації та видалення стоп-слів)",
        top_n=10,
    )

    # 2. Далі TF-IDF після лематизації та видалення стоп-слів
    lem_1 = load_subcorpora_lemmas(p1, chunk_size=1000, chunks_count=20)
    lem_2 = load_subcorpora_lemmas(p2, chunk_size=1000, chunks_count=20)

    compute_tfidf_summary(
        lem_1 + lem_2,
        ["Текст 1 (20 підвибірок)", "Текст 2 (20 підвибірок)"],
        "TF-IDF після лематизації та видалення стоп-слів",
        top_n=10,
    )

    # 3. Кластеризація KMeans на 2 кластери по 20 підвибірках кожного тексту
    all_subdocs = raw_1 + raw_2
    all_labels = [f"Т1_п{i+1}" for i in range(len(raw_1))] + [f"Т2_п{i+1}" for i in range(len(raw_2))]
    cluster_subcorpora(all_subdocs, all_labels, cluster_num=2)