from enum import Enum

from pydantic import BaseModel


class DocumentSource(str, Enum):
    WHATSAPP = "whatsapp"
    CRM = "crm"
    WEB = "web"
    MANUAL = "manual"


class DocumentType(str, Enum):
    INE = "INE"
    PASAPORTE = "PASAPORTE"
    LICENCIA = "LICENCIA"


class ReceiptValidationRequest(BaseModel):
    client_id: str
    source: DocumentSource = DocumentSource.MANUAL


class IdentityValidationRequest(BaseModel):
    client_id: str
    document_type: DocumentType
