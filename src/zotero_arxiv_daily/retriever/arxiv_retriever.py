from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
from types import SimpleNamespace
import multiprocessing
import os
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    
    
    def _retrieve_raw_papers(self) -> list[SimpleNamespace]:
        query = "+".join(self.config.source.arxiv.category)
        feed_url = f"https://rss.arxiv.org/atom/{query}"

        logger.info(f"Fetching arXiv RSS: {feed_url}")
        feed = feedparser.parse(feed_url)

        status = feed.get("status", 200)
        if status >= 400:
            raise RuntimeError(
                f"arXiv RSS returned HTTP {status}"
            )

        if "Feed error for query" in feed.feed.get("title", ""):
            raise ValueError(f"Invalid arXiv query: {query}")

        if feed.get("bozo") and not feed.entries:
            raise RuntimeError(
                f"Failed to parse arXiv RSS: "
                f"{feed.get('bozo_exception')}"
            )

        include_cross_list = self.config.source.arxiv.get(
            "include_cross_list", False
        )

        allowed_types = (
            {"new", "cross"}
            if include_cross_list
            else {"new"}
        )

        raw_papers = []
        seen_ids = set()

        for entry in feed.entries:
            announce_type = entry.get(
                "arxiv_announce_type", "new"
            )

            if announce_type not in allowed_types:
                continue

            entry_id = entry.get("id", "")
            if not entry_id.startswith("oai:arXiv.org:"):
                logger.warning(
                    f"Skipping invalid arXiv RSS ID: {entry_id}"
                )
                continue

            paper_id = entry_id.removeprefix("oai:arXiv.org:")

            if paper_id in seen_ids:
                continue

            summary = entry.get("summary", "")
            _, separator, abstract = summary.partition(
                "Abstract:"
            )

            if not separator or not abstract.strip():
                logger.warning(
                    f"Skipping paper without abstract: {paper_id}"
                )
                continue

            author_string = entry.get("author", "")
            author_names = [
                name.strip()
                for name in author_string.split(",")
                if name.strip()
            ]

            paper = SimpleNamespace(
                title=entry.get("title", "").strip(),
                authors=[
                    SimpleNamespace(name=name)
                    for name in author_names
                ],
                summary=abstract.strip(),
                entry_id=f"https://arxiv.org/abs/{paper_id}",
                pdf_url=f"https://arxiv.org/pdf/{paper_id}",
                source_url=lambda pid=paper_id: (
                    f"https://arxiv.org/e-print/{pid}"
                ),
            )

            raw_papers.append(paper)
            seen_ids.add(paper_id)

        if self.config.executor.debug:
            raw_papers = raw_papers[:10]

        if feed.entries and not raw_papers:
            relevant = any(
                entry.get("arxiv_announce_type", "new")
                in allowed_types
                for entry in feed.entries
            )
            if relevant:
                raise RuntimeError(
                    "RSS entries found, but no valid papers parsed"
                )

        self._raw_by_url = {
            paper.entry_id: paper
            for paper in raw_papers
        }

        logger.info(
            f"Retrieved metadata for {len(raw_papers)} "
            "arXiv papers directly from RSS"
        )

        return raw_papers

   
    
    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        # Only construct metadata before reranking.
        return Paper(
            source=self.name,
            title=raw_paper.title,
            authors=[a.name for a in raw_paper.authors],
            abstract=raw_paper.summary,
            url=raw_paper.entry_id,
            pdf_url=raw_paper.pdf_url,
            full_text=None,
        )

    def load_full_text(self, paper: Paper) -> None:
        # Retrieve full text only for selected papers.
        raw_paper = getattr(
            self, "_raw_by_url", {}
        ).get(paper.url)

        if raw_paper is None:
            logger.warning(
                f"No raw arXiv metadata for {paper.url}"
            )
            return

        full_text = extract_text_from_tar(raw_paper)

        if full_text is None:
            full_text = extract_text_from_html(raw_paper)

        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)

        paper.full_text = full_text


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
