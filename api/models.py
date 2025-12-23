"""
API Models - Pydantic schemas for request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


# ============== Enums ==============

class JobStatus(str, Enum):
    """Job processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class OutputFormat(str, Enum):
    """Supported output formats"""
    MARKDOWN = "md"
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    JSON = "json"


class AudioFormat(str, Enum):
    """Supported audio formats"""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"
    WEBM = "webm"
    M4A = "m4a"


# ============== Request Models ==============

class TranscriptionRequest(BaseModel):
    """Request for transcription"""
    language: str = Field("en", description="Language code (e.g., en, hi, te)")
    enable_diarization: bool = Field(False, description="Enable speaker diarization")
    enhance_audio: bool = Field(True, description="Apply audio enhancement")
    
    class Config:
        schema_extra = {
            "example": {
                "language": "en",
                "enable_diarization": False,
                "enhance_audio": True
            }
        }


class AnalysisRequest(BaseModel):
    """Request for text analysis"""
    text: str = Field(..., description="Text to analyze", min_length=10)
    title: str = Field("Document", description="Document title")
    use_ai: bool = Field(True, description="Use AI for enhanced analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Instagram is a photo and video sharing social networking service...",
                "title": "Instagram Overview",
                "use_ai": True
            }
        }


class GenerateRequest(BaseModel):
    """Request for document generation"""
    text: str = Field(..., description="Text to convert to document", min_length=10)
    title: str = Field("EchoNotes Document", description="Document title")
    format: OutputFormat = Field(OutputFormat.HTML, description="Output format")
    use_ai: bool = Field(True, description="Use AI for enhanced content")
    include_full_content: bool = Field(True, description="Include full text in document")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Instagram is a photo and video sharing social networking service...",
                "title": "Instagram Overview",
                "format": "html",
                "use_ai": True,
                "include_full_content": True
            }
        }


class ProcessRequest(BaseModel):
    """Request for full pipeline processing"""
    title: str = Field("EchoNotes Document", description="Document title")
    format: OutputFormat = Field(OutputFormat.HTML, description="Output format")
    use_ai: bool = Field(True, description="Use AI for enhanced content")
    language: str = Field("en", description="Language code")
    enhance_audio: bool = Field(True, description="Apply audio enhancement")
    enable_diarization: bool = Field(False, description="Enable speaker diarization")


# ============== Response Models ==============

class JobResponse(BaseModel):
    """Response for job creation"""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    message: str = Field(..., description="Status message")
    created_at: str = Field(..., description="Job creation timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "abc12345",
                "status": "processing",
                "message": "Processing started. Check status with /api/status/{job_id}",
                "created_at": "2024-01-15T10:30:00"
            }
        }


class TranscriptionResult(BaseModel):
    """Result of transcription"""
    text: str = Field(..., description="Transcribed text")
    confidence: float = Field(..., description="Confidence score (0-1)")
    duration: float = Field(..., description="Audio duration in seconds")
    word_count: int = Field(..., description="Number of words")
    segments: List[Dict[str, Any]] = Field(default=[], description="Timestamped segments")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Hello, this is a test transcription.",
                "confidence": 0.95,
                "duration": 5.2,
                "word_count": 6,
                "segments": [
                    {"start": 0.0, "end": 2.5, "text": "Hello, this is"}
                ]
            }
        }


class ConceptResult(BaseModel):
    """A key concept"""
    term: str
    definition: str
    importance: float
    frequency: int


class QuestionResult(BaseModel):
    """A study question"""
    question: str
    type: str
    difficulty: str
    hint: Optional[str] = None


class AnalysisResult(BaseModel):
    """Result of text analysis"""
    title: str
    executive_summary: str
    key_sentences: List[str]
    concepts: List[Dict[str, Any]]
    questions: List[Dict[str, Any]]
    related_topics: List[str]
    word_count: int
    reading_time: float
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Instagram Overview",
                "executive_summary": "Instagram is a social media platform...",
                "key_sentences": ["Instagram is a photo sharing service..."],
                "concepts": [{"term": "Instagram", "importance": 0.9}],
                "questions": [{"question": "What is Instagram?", "difficulty": "easy"}],
                "related_topics": ["Social Media", "Technology"],
                "word_count": 150,
                "reading_time": 0.8
            }
        }


class AIEnhancedResult(BaseModel):
    """AI-enhanced content"""
    simplified_explanation: str
    eli5_explanation: str
    key_takeaways: List[str]
    examples: List[str]
    faq: List[Dict[str, str]]
    vocabulary: List[Dict[str, str]]


class GenerationResult(BaseModel):
    """Result of document generation"""
    job_id: str
    status: JobStatus
    document_path: str
    format: str
    ai_enhanced: bool
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "abc12345",
                "status": "completed",
                "document_path": "/api/download/abc12345",
                "format": "html",
                "ai_enhanced": True
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    modules: Dict[str, bool]
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "modules": {
                    "audio_processor": True,
                    "speech_transcriber": True,
                    "nlp_analyzer": True
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: str
    status_code: int
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "Invalid audio file format",
                "status_code": 400
            }
        }


class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str
    status: JobStatus
    created_at: str
    updated_at: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class JobListResponse(BaseModel):
    """List of jobs response"""
    jobs: List[Dict[str, Any]]
    total: int
