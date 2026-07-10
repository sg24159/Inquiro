from pydantic import BaseModel, ValidationError

from shared.models import ProcessedFinding, RawResult, ReportAssets, SubTask


class PlannerInput(BaseModel):
    query: str


class PlannerOutput(BaseModel):
    sub_tasks: list[SubTask]


class RetrieverInput(BaseModel):
    sub_tasks: list[SubTask]


class RetrieverOutput(BaseModel):
    raw_results: list[RawResult]


class ProcessorInput(BaseModel):
    raw_results: list[RawResult]


class ProcessorOutput(BaseModel):
    processed_findings: list[ProcessedFinding]
    synthesized_answer: str = ""


class WriterInput(BaseModel):
    query: str
    sub_tasks: list[SubTask]
    processed_findings: list[ProcessedFinding]
    synthesized_answer: str = ""


class WriterOutput(BaseModel):
    report: ReportAssets | None = None


def validate_contract(data: dict, model: type[BaseModel]) -> list[str]:
    try:
        model(**data)
        return []
    except ValidationError as e:
        msgs = []
        for err in e.errors():
            field = ".".join(str(p) for p in err["loc"])
            msgs.append(
                f"  [Contract] {model.__name__}.{field}: {err['msg']} "
                f"(got {err.get('input', 'N/A')!r})"
            )
        return msgs
