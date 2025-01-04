import pandas as pd
import re
from math import log, sqrt


class JournalSearchModel:
    def __init__(self, dataset_path, stopwords_path):
        self.file_path = dataset_path
        self.stopword_path = stopwords_path
        # Load dataset
        (
            self.abstracts,
            self.titles,
            self.authors,
            self.years,
            self.urls,
            self.dataset,
        ) = self.load_dataset()
        # Load stopwords
        self.stopwords = self.load_stopwords(stopwords_path)

        # Preprocess documents
        self.processed_docs = [
            self.preprocessing(abstract) for abstract in self.abstracts
        ]

        # Build inverted index
        self.inverted_index = self.build_index(self.processed_docs)

        # Calculate TF-IDF for documents
        self.tfidf_matrix, self.idf = self.calculate_tfidf(self.processed_docs)

    def load_dataset(self):
        dataset = pd.read_excel(self.file_path)
        return (
            dataset["Abstrak"].tolist(),
            dataset["Judul Jurnal"].tolist(),
            dataset["Penulis"].tolist(),
            dataset["Tahun Terbit"].tolist(),
            dataset["Link Jurnal"].tolist(),
            dataset,
        )

    def load_stopwords(self, file_path):
        with open(file_path, "r") as file:
            stopwords_list = file.read().splitlines()
        return set(stopwords_list)

    def case_folding(self, text):
        text = "".join([char if not char.isdigit() else " " for char in text])
        text = "".join(
            [char if char.isalnum() or char.isspace() else " " for char in text]
        )
        return text.lower()

    def tokenization(self, text):
        return text.split()

    def stopword_removal(self, tokens):
        return [word for word in tokens if word not in self.stopwords]

    def preprocessing(self, text):
        if not isinstance(text, str):
            text = str(text)
        text = self.case_folding(text)
        tokens = self.tokenization(text)
        return self.stopword_removal(tokens)

    def build_index(self, documents):
        term_doc_pairs = []
        for doc_id, tokens in enumerate(documents):
            for token in tokens:
                term_doc_pairs.append((token, doc_id))

        term_doc_pairs = sorted(term_doc_pairs, key=lambda x: (x[0], x[1]))

        index = {}
        for term, doc_id in term_doc_pairs:
            if term not in index:
                index[term] = []
            if doc_id not in index[term]:
                index[term].append(doc_id)

        return index

        def encode_vb(self, number):
            bytes_ = []
            while True:
                byte = number & 127
                bytes_.insert(0, byte)
                if number < 128:
                    break
            number >>= 7

        bytes_[-1] |= 128
        return bytes_

    def compress_posting_list(self, inverted_index):
        compressed_index = {}
        for term, posting_list in inverted_index.items():
            gaps = [posting_list[0]] + [
                posting_list[i] - posting_list[i - 1]
                for i in range(1, len(posting_list))
            ]
            compressed_bytes = []
            for gap in gaps:
                compressed_bytes.extend(self.encode_vb(gap))
            compressed_index[term] = compressed_bytes
        return compressed_index

    def calculate_tf(self, doc_tokens):
        tf = {}
        for token in doc_tokens:
            tf[token] = tf.get(token, 0) + 1
        for token in tf:
            tf[token] /= len(doc_tokens)
        return tf

    def calculate_idf(self, documents):
        idf = {}
        total_docs = len(documents)
        for doc_tokens in documents:
            for token in set(doc_tokens):
                idf[token] = idf.get(token, 0) + 1
        for token in idf:
            idf[token] = log(total_docs / idf[token])
        return idf

    def calculate_tfidf(self, documents):
        tfidf = []
        idf = self.calculate_idf(documents)
        for doc_tokens in documents:
            tf = self.calculate_tf(doc_tokens)
            tfidf.append({token: tf[token] * idf[token] for token in tf})
        return tfidf, idf

    def process_query(self, query):
        tokens = self.preprocessing(query)
        tf = self.calculate_tf(tokens)
        query_vector = {token: tf[token] * self.idf.get(token, 0) for token in tokens}
        return tokens, query_vector

    def retrieve_by_inverted_index(self, query_tokens):
        matched_doc_ids = set()
        for token in query_tokens:
            if token in self.inverted_index:
                matched_doc_ids.update(self.inverted_index[token])
        return [(doc_id, self.titles[doc_id]) for doc_id in matched_doc_ids]

    def rank_documents(self, query_vector, matched_docs):
        ranked_results = []
        for doc_id, _ in matched_docs:
            dot_product = sum(
                self.tfidf_matrix[doc_id].get(token, 0) * query_vector.get(token, 0)
                for token in query_vector
            )
            doc_norm = sqrt(
                sum(value**2 for value in self.tfidf_matrix[doc_id].values())
            )
            query_norm = sqrt(sum(value**2 for value in query_vector.values()))
            if doc_norm * query_norm == 0:
                similarity = 0
            else:
                similarity = dot_product / (doc_norm * query_norm)
            ranked_results.append((doc_id, similarity))

        ranked_results.sort(key=lambda x: x[1], reverse=True)
        return ranked_results
