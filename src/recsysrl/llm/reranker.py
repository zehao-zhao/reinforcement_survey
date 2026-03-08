from dataclasses import dataclass


@dataclass
class LLMReranker:
    enabled: bool = False

    def rerank(self, candidates, user_text=""):
        if not self.enabled:
            return candidates
        # placeholder deterministic rerank by candidate id descending
        return sorted(candidates, reverse=True)
