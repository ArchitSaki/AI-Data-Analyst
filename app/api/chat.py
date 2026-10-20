
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.ai_analyst import (
    analyze_question,
    generate_business_summary,
)
from app.services.data_loader import CSVValidationError, load_csv
import uuid

from app.services.business_insights import generate_business_insights
from app.services.dataset_store import (
    save_dataset,
    save_business_insights,
)

router = APIRouter(prefix="/api", tags=["AI Analyst"])



@router.post("/ask")
async def ask_question(
    question: str = Form(...),
    file: UploadFile | None = File(default=None),
    files: list[UploadFile] | None = File(default=None),
    conversation_id: str | None = Form(default=None),
):
    uploads = files or ([file] if file is not None else [])

    if not uploads:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least one CSV file.",
        )

    try:
        datasets = {}

        for uploaded in uploads:
            if (
                not uploaded.filename
                or not uploaded.filename.lower().endswith(".csv")
            ):
                raise HTTPException(
                    status_code=400,
                    detail="All uploaded files must be CSV files.",
                )

            if uploaded.filename in datasets:
                raise HTTPException(
                    status_code=400,
                    detail=f"Duplicate filename: {uploaded.filename}",
                )

            contents = await uploaded.read()
            datasets[uploaded.filename] = load_csv(contents)

        first_dataframe = next(iter(datasets.values()))

        return analyze_question(
            first_dataframe,
            question,
            conversation_id=conversation_id,
            datasets=datasets if len(datasets) > 1 else None,
        )

    except CSVValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    finally:
        for uploaded in uploads:
            await uploaded.close()


@router.post("/insights")
async def get_business_insights(
    file: UploadFile = File(...),
    conversation_id: str | None = Form(default=None),
):
    """Generate a business insights report for an uploaded CSV."""

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    try:
        contents = await file.read()
        dataframe = load_csv(contents)

        dataset_id = conversation_id or str(uuid.uuid4())

        save_dataset(dataset_id, dataframe)
        report = generate_business_insights(dataframe)

        try:
            report["business_summary"] = generate_business_summary(report)
        except Exception:
            report["business_summary"] = (
                "The AI-generated summary is temporarily unavailable. "
                "Please review the calculated findings and their evidence below."
            )

        save_business_insights(dataset_id, report)

        return {
            "conversation_id": dataset_id,
            **report,
            "analysis_type": "business_insights",
        }

    except CSVValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    finally:
        await file.close()
