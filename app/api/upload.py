from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.data_loader import (
    CSVValidationError,
    load_csv,
    profile_dataframe,
)

router = APIRouter(prefix="/api", tags=["Data Upload"])


@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a file with a .csv extension.",
        )

    try:
        contents = await file.read()
        dataframe = load_csv(contents)
        profile = profile_dataframe(dataframe)

        return {
            "filename": file.filename,
            "profile": profile,
        }

    except CSVValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    finally:
        await file.close()
