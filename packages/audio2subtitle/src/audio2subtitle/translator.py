import ctranslate2
import transformers


class MarianTranslator:
    """Batch translator using CTranslate2-optimized MarianMT models. Fast CPU inference."""

    def __init__(self, model_name: str, ct2_model_path: str):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.translator = ctranslate2.Translator(ct2_model_path)

    def translate_batch(self, texts: list[str], batch_size: int = 32) -> list[str]:
        """Translate a list of texts in batches. Returns list of translated strings."""
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
                target_tokens = result.hypotheses[0]
                target_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
                translations.append(self.tokenizer.decode(target_ids, skip_special_tokens=True))
        return translations
