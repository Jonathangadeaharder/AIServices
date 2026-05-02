from __future__ import annotations


class MarianTranslator:
    """Batch translator using CTranslate2-optimized MarianMT models."""

    def __init__(
        self,
        ct2_model_path: str,
        model_name: str = "Helsinki-NLP/opus-mt-tc-big-de-es",
    ):
        import ctranslate2
        import transformers

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.translator = ctranslate2.Translator(ct2_model_path)

    def translate_batch(self, texts: list[str], batch_size: int = 32) -> list[str]:
        if not texts:
            return []
        translations: list[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            source_tokens = [
                self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
                for text in batch
            ]
            results = self.translator.translate_batch(source_tokens)
            for result in results:
                target_ids = self.tokenizer.convert_tokens_to_ids(
                    result.hypotheses[0]
                )
                translations.append(
                    self.tokenizer.decode(target_ids, skip_special_tokens=True)
                )
        return translations
