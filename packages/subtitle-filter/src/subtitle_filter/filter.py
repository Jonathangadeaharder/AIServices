from pathlib import Path

import pandas as pd
import pysrt


class VocabFilter:
    """Filter subtitle segments by vocabulary level using spaCy lemmatization."""

    @staticmethod
    def load_vocab(vocab_dir: str, levels: list[str]) -> set[str]:
        vocab: set[str] = set()
        for level in levels:
            path = Path(vocab_dir) / f"{level}_vokabeln.csv"
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path, header=None)
                words = df.iloc[:, 0].astype(str).str.strip().str.lower().tolist()
                vocab.update(words)
            except Exception:
                continue
        return vocab

    @staticmethod
    def filter_subs(
        subs: pysrt.SubRipFile,
        nlp: object,
        vocab_set: set[str],
    ) -> pysrt.SubRipFile:
        """Keep segments that contain at least one word NOT in vocab_set."""
        filtered = pysrt.SubRipFile()
        for sub in subs:
            doc = nlp(sub.text)  # type: ignore[operator]
            lemmas: list[str] = []
            for token in doc:
                if (
                    token.is_punct
                    or token.is_stop
                    or token.pos_ in ("PROPN", "NUM", "INTJ", "X", "SPACE")
                ):
                    continue
                lemma = token.lemma_.lower().strip()
                if lemma:
                    lemmas.append(lemma)
            if lemmas and not all(lem in vocab_set for lem in lemmas):
                filtered.append(sub)
        # Re-index
        for i, sub in enumerate(filtered, 1):
            sub.index = i
        return filtered
