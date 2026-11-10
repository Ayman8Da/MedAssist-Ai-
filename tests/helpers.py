from __future__ import annotations

from fastapi.testclient import TestClient

METFORMIN_DOC = """SYNTHETIC TEST DOCUMENT

METFORMIN AND RENAL FUNCTION

Metformin is cleared by the kidneys. In patients with impaired renal function,
accumulation may increase the risk of lactic acidosis.

Assessment should include eGFR, the trend of renal function, hydration status,
liver disease, alcohol use, hypoxia, and acute illness.

General principles: metformin is contraindicated when eGFR is below 30
mL/min/1.73m2. Initiation is not recommended between 30 and 45. Dose review is
required. Withhold metformin during acute illness, dehydration, hypoxia, or
before iodinated contrast when advised by the local protocol.

Warning signs of lactic acidosis include malaise, respiratory distress,
abdominal pain, and altered mental status. Urgent assessment is required.

Missing information: recent eGFR, baseline renal function, concurrent
nephrotoxic drugs, contrast exposure, sepsis, and pregnancy.
"""

DIABETES_DOC = """SYNTHETIC TEST DOCUMENT

DIABETES AND SUSPECTED RESPIRATORY INFECTION

Adults with diabetes may have a higher risk of severe respiratory infection.
Hyperglycaemia can impair immune function.

Clinical assessment should include vital signs, oxygen saturation, mental
status, hydration, glucose trend, renal function, and a focused respiratory
examination.

Missing information includes symptom duration, fever, vaccination status,
recent antibiotics, pregnancy, immunosuppression, and prior hospitalisation.

Red flags include hypoxia, hypotension, confusion, and increased work of
breathing. These require urgent in-person assessment.
"""


def upload_document(
    client: TestClient,
    *,
    text: str = METFORMIN_DOC,
    filename: str = "metformin.txt",
    source: str = "synthetic-test",
    title: str | None = None,
    publication_date: str | None = None,
    version: str | None = None,
    category: str | None = None,
    document_key: str | None = None,
    content_type: str = "text/plain",
):
    data: dict[str, str] = {"source": source}
    if title is not None:
        data["title"] = title
    if publication_date is not None:
        data["publication_date"] = publication_date
    if version is not None:
        data["version"] = version
    if category is not None:
        data["category"] = category
    if document_key is not None:
        data["document_key"] = document_key
    return client.post(
        "/api/documents/upload",
        files={"file": (filename, text.encode("utf-8"), content_type)},
        data=data,
    )
