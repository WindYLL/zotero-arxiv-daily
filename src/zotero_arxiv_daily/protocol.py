from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json

RawPaperItem = TypeVar("RawPaperItem")


def _request_llm(
    openai_client: OpenAI,
    llm_params: dict,
    messages: list[dict],
) -> str:
    api_mode = llm_params.get("api_mode", "chat_completion")
    generation_kwargs = dict(llm_params.get("generation_kwargs", {}))

    if api_mode == "chat_completion":
        response = openai_client.chat.completions.create(
            messages=messages,
            **generation_kwargs,
        )
        return response.choices[0].message.content

    if api_mode == "response":
        max_tokens = generation_kwargs.pop("max_tokens", None)

        if (
            max_tokens is not None
            and "max_output_tokens" not in generation_kwargs
        ):
            generation_kwargs["max_output_tokens"] = max_tokens

        response = openai_client.responses.create(
            input=messages,
            **generation_kwargs,
        )
        return response.output_text

    raise ValueError(
        f"Unsupported llm.api_mode: {api_mode}. "
        "Expected 'chat_completion' or 'response'."
    )


@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

    def _generate_tldr_with_llm(
        self,
        openai_client: OpenAI,
        llm_params: dict,
    ) -> str:
        lang = llm_params.get("language", "Chinese")

        if not self.abstract:
            logger.warning(
                f"No abstract is provided for {self.url}"
            )
            return "Failed to generate TLDR. No abstract is provided."

        prompt = f"""
            Translate the following academic paper abstract into {lang} faithfully and accurately.

            Requirements:
            1. Translate the original abstract only. Do not summarize, paraphrase, expand, explain, or comment on it.
            2. Do not omit any important information from the original text.
            3. Do not add any conclusions, interpretations, or background information that are not present in the original abstract.
            4. Preserve the precise meaning of technical terms.
            5. Keep common technical abbreviations such as BEV, V2X, LiDAR, and Transformer in English.
            6. Preserve all numerical values, dataset names, model names, metric names, and experimental results accurately.
            7. Use concise, natural, and publication-quality academic Chinese.
            8. Output only the translated abstract. Do not add headings such as "Chinese Translation" or "Abstract Translation".

            Original Abstract:
            {self.abstract}
            """

        tldr = _request_llm(
            openai_client,
            llm_params,
            [
                {
                    "role": "system",
                    "content": (
                        "You are a professional academic translator. "
                        f"Your task is to faithfully and accurately translate "
                        f"English academic abstracts into {lang}. "
                        "Do not summarize, rewrite, expand, explain, or add "
                        "any information that is not present in the original text."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        return tldr

    def generate_tldr(
        self,
        openai_client: OpenAI,
        llm_params: dict,
    ) -> str:
        try:
            tldr = self._generate_tldr_with_llm(
                openai_client,
                llm_params,
            )
            self.tldr = tldr
            return tldr

        except Exception as e:
            logger.warning(
                f"Failed to generate tldr of {self.url}: {e}"
            )

            # If translation fails, fall back to the original abstract.
            tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(
        self,
        openai_client: OpenAI,
        llm_params: dict,
    ) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = (
                "Given the beginning of a paper, extract the affiliations "
                "of the authors in a python list format, which is sorted by "
                "the author order. If there is no affiliation found, return "
                "an empty list '[]':\n\n"
                f"{self.full_text}"
            )

            # Use gpt-4o tokenizer for estimation.
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]
            prompt = enc.decode(prompt_tokens)

            affiliations = _request_llm(
                openai_client,
                llm_params,
                [
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant who perfectly extracts "
                            "affiliations of authors from a paper. "
                            "You should return a python list of affiliations "
                            "sorted by the author order, like "
                            "[\"TsingHua University\", \"Peking University\"]. "
                            "If an affiliation is consisted of multi-level "
                            "affiliations, like 'Department of Computer Science, "
                            "TsingHua University', you should return the top-level "
                            "affiliation 'TsingHua University' only. "
                            "Do not contain duplicated affiliations. "
                            "If there is no affiliation found, you should return "
                            "an empty list []. "
                            "You should only return the final list of affiliations, "
                            "and do not return any intermediate results."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            affiliations = re.search(
                r"\[.*?\]",
                affiliations,
                flags=re.DOTALL,
            ).group(0)
            affiliations = json.loads(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]

            return affiliations

        return None

    def generate_affiliations(
        self,
        openai_client: OpenAI,
        llm_params: dict,
    ) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(
                openai_client,
                llm_params,
            )
            self.affiliations = affiliations
            return affiliations

        except Exception as e:
            logger.warning(
                f"Failed to generate affiliations of {self.url}: {e}"
            )
            self.affiliations = None
            return None


@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
