from pydantic import BaseModel, Field


class SubTask(BaseModel):
    description: str = Field(description="What this sub-task investigates")
    keywords: list[str] = Field(description="Search keywords for API queries")


class RawResult(BaseModel):
    source: str = Field(description="URL or API identifier")
    title: str = Field(description="Title of the result")
    snippet: str = Field(description="Abstract or excerpt")
    sub_task_idx: int = Field(default=0, description="Index of originating sub-task")
    authors: list[str] = Field(default=[], description="Author names")
    published: str = Field(default="", description="Publication date")


def _format_citation_author(authors: list[str]) -> str:
    if not authors:
        return "Unknown"
    if len(authors) == 1:
        return authors[0]
    return f"{authors[0]} et al."


class ProcessedFinding(BaseModel):
    summary: str = Field(description="Concise summary")
    relevance_score: int = Field(description="Relevance 0-3", ge=0, le=3)
    source: str = Field(description="Source URL or citation")
    source_url: str = Field(default="", description="Clickable link to source")
    title: str = Field(default="", description="Original title")
    year: str = Field(default="", description="Publication year")
    citation_author: str = Field(default="", description="First author formatted for Harvard citation")


class ReportAssets(BaseModel):
    markdown_path: str = Field(description="Path to saved .md report")
    json_path: str = Field(description="Path to saved .json data")
    title: str = Field(description="Report title")
