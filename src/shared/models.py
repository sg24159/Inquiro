from pydantic import BaseModel, Field


class SubTask(BaseModel):
    description: str = Field(description="What this sub-task investigates")
    keywords: list[str] = Field(description="Search keywords for API queries")


class RawResult(BaseModel):
    source: str = Field(description="URL or API identifier")
    title: str = Field(description="Title of the result")
    snippet: str = Field(description="Abstract or excerpt")
    sub_task_idx: int = Field(default=0, description="Index of originating sub-task")


class ProcessedFinding(BaseModel):
    summary: str = Field(description="Concise summary")
    relevance_score: float = Field(description="Relevance 0-1", ge=0, le=1)
    source: str = Field(description="Source URL or citation")
    title: str = Field(default="", description="Original title")


class ReportAssets(BaseModel):
    markdown_path: str = Field(description="Path to saved .md report")
    json_path: str = Field(description="Path to saved .json data")
    title: str = Field(description="Report title")
