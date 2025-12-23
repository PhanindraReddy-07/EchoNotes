"""
Domain-Specific Entity Extractor - Meeting NER
===============================================
Custom entity extraction for meeting-specific entities:
ACTION_ITEM, DEADLINE, DECISION, QUESTION, RISK, BLOCKER

This is a KEY NOVEL COMPONENT demonstrating domain-adapted NLP.
"""
import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json


class MeetingEntityType(Enum):
    """Custom entity types for meeting transcripts"""
    ACTION_ITEM = "ACTION_ITEM"      # Tasks to be done
    DEADLINE = "DEADLINE"            # Due dates and timeframes
    DECISION = "DECISION"            # Decisions made
    QUESTION = "QUESTION"            # Questions raised
    RISK = "RISK"                    # Risks identified
    BLOCKER = "BLOCKER"              # Blockers/impediments
    ASSIGNEE = "ASSIGNEE"            # Person assigned
    PROJECT = "PROJECT"              # Project references
    METRIC = "METRIC"                # Numbers, percentages, KPIs


@dataclass
class MeetingEntity:
    """Extracted meeting entity"""
    text: str                        # The extracted text
    entity_type: MeetingEntityType   # Type of entity
    start_pos: int                   # Start position in text
    end_pos: int                     # End position in text
    confidence: float                # Extraction confidence (0-1)
    context: str = ""                # Surrounding context
    metadata: Dict = field(default_factory=dict)  # Additional info
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'type': self.entity_type.value,
            'confidence': round(self.confidence, 3),
            'context': self.context,
            'metadata': self.metadata
        }


@dataclass
class ExtractionResult:
    """Complete extraction result"""
    entities: List[MeetingEntity]
    action_items: List[MeetingEntity]
    deadlines: List[MeetingEntity]
    decisions: List[MeetingEntity]
    questions: List[MeetingEntity]
    risks: List[MeetingEntity]
    blockers: List[MeetingEntity]
    
    def to_dict(self) -> Dict:
        return {
            'total_entities': len(self.entities),
            'action_items': [e.to_dict() for e in self.action_items],
            'deadlines': [e.to_dict() for e in self.deadlines],
            'decisions': [e.to_dict() for e in self.decisions],
            'questions': [e.to_dict() for e in self.questions],
            'risks': [e.to_dict() for e in self.risks],
            'blockers': [e.to_dict() for e in self.blockers]
        }
    
    def get_summary(self) -> str:
        """Get formatted summary of extracted entities"""
        lines = ["=" * 50, "MEETING ENTITIES EXTRACTED", "=" * 50, ""]
        
        if self.action_items:
            lines.append(f"📋 ACTION ITEMS ({len(self.action_items)}):")
            for i, item in enumerate(self.action_items, 1):
                assignee = item.metadata.get('assignee', 'Unassigned')
                lines.append(f"   {i}. {item.text}")
                if assignee != 'Unassigned':
                    lines.append(f"      → Assigned to: {assignee}")
            lines.append("")
        
        if self.deadlines:
            lines.append(f"📅 DEADLINES ({len(self.deadlines)}):")
            for item in self.deadlines:
                lines.append(f"   • {item.text}")
            lines.append("")
        
        if self.decisions:
            lines.append(f"✅ DECISIONS ({len(self.decisions)}):")
            for item in self.decisions:
                lines.append(f"   • {item.text}")
            lines.append("")
        
        if self.questions:
            lines.append(f"❓ OPEN QUESTIONS ({len(self.questions)}):")
            for item in self.questions:
                lines.append(f"   • {item.text}")
            lines.append("")
        
        if self.risks:
            lines.append(f"⚠️ RISKS ({len(self.risks)}):")
            for item in self.risks:
                lines.append(f"   • {item.text}")
            lines.append("")
        
        if self.blockers:
            lines.append(f"🚫 BLOCKERS ({len(self.blockers)}):")
            for item in self.blockers:
                lines.append(f"   • {item.text}")
            lines.append("")
        
        lines.append("=" * 50)
        return '\n'.join(lines)


class MeetingEntityExtractor:
    """
    Domain-Specific Named Entity Recognition for Meeting Transcripts
    
    TECHNICAL NOVELTY:
    - Custom entity types not found in generic NER (SpaCy, etc.)
    - Pattern-based + context-aware extraction
    - Assignee detection and linking
    - Confidence scoring based on pattern strength
    
    Entity Types:
    - ACTION_ITEM: Tasks identified (e.g., "John will prepare the report")
    - DEADLINE: Time references (e.g., "by Friday", "next week")
    - DECISION: Decisions made (e.g., "We decided to use Python")
    - QUESTION: Open questions (e.g., "What about the budget?")
    - RISK: Risks mentioned (e.g., "might delay", "risk of failure")
    - BLOCKER: Impediments (e.g., "blocked by", "waiting for approval")
    
    Usage:
        extractor = MeetingEntityExtractor()
        result = extractor.extract(transcript_text)
        
        for action in result.action_items:
            print(f"Action: {action.text}")
            print(f"Assignee: {action.metadata.get('assignee', 'Unknown')}")
    """
    
    # Action item patterns (task indicators)
    ACTION_PATTERNS = [
        # Direct assignments
        (r'\b(\w+)\s+will\s+(.+?)(?:[.!]|$)', 0.9),
        (r'\b(\w+)\s+should\s+(.+?)(?:[.!]|$)', 0.85),
        (r'\b(\w+)\s+needs?\s+to\s+(.+?)(?:[.!]|$)', 0.85),
        (r'\b(\w+)\s+has\s+to\s+(.+?)(?:[.!]|$)', 0.85),
        (r'\b(\w+)\s+must\s+(.+?)(?:[.!]|$)', 0.9),
        (r'\b(\w+)\s+is\s+going\s+to\s+(.+?)(?:[.!]|$)', 0.8),
        
        # Task phrases
        (r"let'?s\s+(.+?)(?:[.!]|$)", 0.7),
        (r'we\s+need\s+to\s+(.+?)(?:[.!]|$)', 0.8),
        (r'we\s+should\s+(.+?)(?:[.!]|$)', 0.75),
        (r'we\s+have\s+to\s+(.+?)(?:[.!]|$)', 0.8),
        (r'we\s+must\s+(.+?)(?:[.!]|$)', 0.85),
        
        # Action verbs at start
        (r'^(prepare|create|send|review|update|complete|finish|schedule|setup|arrange|organize|draft|write|call|email|contact|follow\s*up)\s+(.+?)(?:[.!]|$)', 0.8),
        
        # TODO/Task markers
        (r'(?:todo|task|action\s*item)[:\s]+(.+?)(?:[.!]|$)', 0.95),
        (r'(?:please|kindly)\s+(.+?)(?:[.!]|$)', 0.7),
        
        # Can you / Could you
        (r'(?:can|could)\s+(?:you|someone)\s+(.+?)(?:[.!?]|$)', 0.75),
    ]
    
    # Deadline patterns
    DEADLINE_PATTERNS = [
        # Specific dates
        (r'by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', 0.9),
        (r'by\s+(tomorrow|today|tonight)', 0.95),
        (r'by\s+(next\s+(?:week|month|monday|tuesday|wednesday|thursday|friday))', 0.9),
        (r'by\s+(end\s+of\s+(?:day|week|month|quarter|year))', 0.9),
        (r'by\s+(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)', 0.95),
        (r'by\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s*\d*', 0.9),
        
        # Deadline phrases
        (r'deadline\s*(?:is|:)?\s*(.+?)(?:[.!]|$)', 0.95),
        (r'due\s+(?:by|on|date)?\s*(.+?)(?:[.!]|$)', 0.9),
        (r'within\s+(\d+\s+(?:days?|weeks?|hours?|months?))', 0.85),
        (r'in\s+(\d+\s+(?:days?|weeks?|hours?|months?))', 0.7),
        
        # ASAP patterns
        (r'\b(asap|as\s+soon\s+as\s+possible|urgent(?:ly)?|immediately)\b', 0.9),
    ]
    
    # Decision patterns
    DECISION_PATTERNS = [
        (r'we\s+(?:have\s+)?decided\s+(?:to\s+)?(.+?)(?:[.!]|$)', 0.95),
        (r'decision\s*(?:is|:)?\s*(.+?)(?:[.!]|$)', 0.95),
        (r"let'?s\s+go\s+with\s+(.+?)(?:[.!]|$)", 0.85),
        (r'we\s+(?:will|are\s+going\s+to)\s+(?:go\s+with|use|choose|select)\s+(.+?)(?:[.!]|$)', 0.9),
        (r'agreed\s+(?:to|on|that)\s+(.+?)(?:[.!]|$)', 0.9),
        (r'final\s+decision\s*(?:is|:)?\s*(.+?)(?:[.!]|$)', 0.95),
        (r'we\s+are\s+(?:going\s+)?(?:to\s+)?(?:proceed|move\s+forward)\s+with\s+(.+?)(?:[.!]|$)', 0.85),
        (r'approved\s+(.+?)(?:[.!]|$)', 0.9),
        (r'consensus\s+(?:is|:)?\s*(.+?)(?:[.!]|$)', 0.9),
    ]
    
    # Question patterns
    QUESTION_PATTERNS = [
        (r'(?:what|how|when|where|why|who|which)\s+.+\?', 0.9),
        (r'(?:can|could|would|should|will|do|does|is|are|have|has)\s+.+\?', 0.85),
        (r'(?:any\s+)?(?:questions?|thoughts?|ideas?|suggestions?|concerns?)\s*(?:about|on|regarding)?\s*(.+?)\?', 0.8),
        (r'what\s+about\s+(.+?)\?', 0.85),
        (r'(?:i\s+)?(?:wonder|wondering)\s+(?:if|whether|what|how)\s+(.+?)(?:[.!?]|$)', 0.75),
        (r"(?:do|does|did)\s+(?:anyone|somebody|we)\s+know\s+(.+?)\?", 0.8),
    ]
    
    # Risk patterns
    RISK_PATTERNS = [
        (r'risk\s+(?:is|of|that)?\s*(.+?)(?:[.!]|$)', 0.95),
        (r'(?:might|may|could)\s+(?:cause|lead\s+to|result\s+in)\s+(.+?)(?:[.!]|$)', 0.8),
        (r'(?:might|may|could)\s+(?:fail|delay|break|crash)', 0.85),
        (r'concern(?:ed)?\s+(?:about|that|is)?\s*(.+?)(?:[.!]|$)', 0.8),
        (r'(?:potential|possible)\s+(?:issue|problem|risk)\s*(?:is|:)?\s*(.+?)(?:[.!]|$)', 0.9),
        (r'worried\s+(?:about|that)\s+(.+?)(?:[.!]|$)', 0.75),
        (r'(?:if|when)\s+.+?\s+(?:fails?|breaks?|crashes?)', 0.7),
        (r'danger\s+(?:of|is|that)\s+(.+?)(?:[.!]|$)', 0.9),
        (r'threat\s+(?:of|is|that)\s+(.+?)(?:[.!]|$)', 0.9),
    ]
    
    # Blocker patterns
    BLOCKER_PATTERNS = [
        (r'blocked\s+(?:by|on)\s+(.+?)(?:[.!]|$)', 0.95),
        (r'blocker\s*(?:is|:)?\s*(.+?)(?:[.!]|$)', 0.95),
        (r'waiting\s+(?:for|on)\s+(.+?)(?:[.!]|$)', 0.85),
        (r'depends?\s+on\s+(.+?)(?:[.!]|$)', 0.75),
        (r'(?:can\'?t|cannot|unable\s+to)\s+(?:proceed|continue|move\s+forward)\s+(?:until|without)\s+(.+?)(?:[.!]|$)', 0.9),
        (r'stuck\s+(?:on|at|with)\s+(.+?)(?:[.!]|$)', 0.85),
        (r'impediment\s*(?:is|:)?\s*(.+?)(?:[.!]|$)', 0.95),
        (r'obstacle\s*(?:is|:)?\s*(.+?)(?:[.!]|$)', 0.9),
        (r'holding\s+(?:us|things)\s+(?:up|back)', 0.8),
        (r'pending\s+(?:approval|review|sign-?off)\s+(?:from|by)\s+(.+?)(?:[.!]|$)', 0.85),
    ]
    
    # Common names for assignee detection
    COMMON_NAMES = {
        'john', 'jane', 'mike', 'michael', 'david', 'sarah', 'james', 'robert',
        'mary', 'jennifer', 'lisa', 'susan', 'tom', 'thomas', 'peter', 'paul',
        'raj', 'rahul', 'priya', 'amit', 'sanjay', 'kumar', 'sharma', 'patel',
        'alex', 'chris', 'sam', 'pat', 'kim', 'lee', 'chen', 'wang',
    }
    
    def __init__(
        self,
        use_spacy: bool = False,
        spacy_model: str = "en_core_web_sm",
        custom_names: Optional[Set[str]] = None
    ):
        """
        Initialize the extractor
        
        Args:
            use_spacy: Whether to use SpaCy for additional NER
            spacy_model: SpaCy model to use
            custom_names: Additional names to recognize as assignees
        """
        self.use_spacy = use_spacy
        self.nlp = None
        
        # Build name set
        self.known_names = self.COMMON_NAMES.copy()
        if custom_names:
            self.known_names.update(n.lower() for n in custom_names)
        
        # Load SpaCy if requested
        if use_spacy:
            self._load_spacy(spacy_model)
    
    def _load_spacy(self, model_name: str):
        """Load SpaCy model"""
        try:
            import spacy
            self.nlp = spacy.load(model_name)
            print(f"[EntityExtractor] Loaded SpaCy model: {model_name}")
        except ImportError:
            print("[EntityExtractor] SpaCy not available, using pattern-only extraction")
            self.use_spacy = False
        except OSError:
            print(f"[EntityExtractor] SpaCy model '{model_name}' not found")
            print("  Install with: python -m spacy download en_core_web_sm")
            self.use_spacy = False
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Extract meeting entities from text
        
        Args:
            text: Transcript text
            
        Returns:
            ExtractionResult with categorized entities
        """
        all_entities = []
        
        # Extract each entity type
        action_items = self._extract_pattern_entities(
            text, self.ACTION_PATTERNS, MeetingEntityType.ACTION_ITEM
        )
        
        deadlines = self._extract_pattern_entities(
            text, self.DEADLINE_PATTERNS, MeetingEntityType.DEADLINE
        )
        
        decisions = self._extract_pattern_entities(
            text, self.DECISION_PATTERNS, MeetingEntityType.DECISION
        )
        
        questions = self._extract_pattern_entities(
            text, self.QUESTION_PATTERNS, MeetingEntityType.QUESTION
        )
        
        risks = self._extract_pattern_entities(
            text, self.RISK_PATTERNS, MeetingEntityType.RISK
        )
        
        blockers = self._extract_pattern_entities(
            text, self.BLOCKER_PATTERNS, MeetingEntityType.BLOCKER
        )
        
        # Enrich action items with assignee info
        action_items = self._enrich_action_items(action_items, text)
        
        # Use SpaCy for additional entities if available
        if self.use_spacy and self.nlp:
            spacy_entities = self._extract_spacy_entities(text)
            # Merge without duplicates
            for entity in spacy_entities:
                if not self._is_duplicate(entity, all_entities):
                    if entity.entity_type == MeetingEntityType.ACTION_ITEM:
                        action_items.append(entity)
        
        # Combine all entities
        all_entities = action_items + deadlines + decisions + questions + risks + blockers
        
        # Remove duplicates and sort by position
        all_entities = self._deduplicate_entities(all_entities)
        all_entities.sort(key=lambda e: e.start_pos)
        
        return ExtractionResult(
            entities=all_entities,
            action_items=[e for e in all_entities if e.entity_type == MeetingEntityType.ACTION_ITEM],
            deadlines=[e for e in all_entities if e.entity_type == MeetingEntityType.DEADLINE],
            decisions=[e for e in all_entities if e.entity_type == MeetingEntityType.DECISION],
            questions=[e for e in all_entities if e.entity_type == MeetingEntityType.QUESTION],
            risks=[e for e in all_entities if e.entity_type == MeetingEntityType.RISK],
            blockers=[e for e in all_entities if e.entity_type == MeetingEntityType.BLOCKER]
        )
    
    def _extract_pattern_entities(
        self,
        text: str,
        patterns: List[Tuple[str, float]],
        entity_type: MeetingEntityType
    ) -> List[MeetingEntity]:
        """Extract entities using regex patterns"""
        entities = []
        
        for pattern, base_confidence in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                # Get the captured text (use last group if multiple)
                if match.groups():
                    captured = match.group(len(match.groups()))
                    if captured:
                        captured = captured.strip()
                else:
                    captured = match.group(0).strip()
                
                # Skip very short extractions
                if len(captured) < 3:
                    continue
                
                # Get context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                entity = MeetingEntity(
                    text=captured,
                    entity_type=entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=base_confidence,
                    context=context
                )
                entities.append(entity)
        
        return entities
    
    def _enrich_action_items(
        self,
        action_items: List[MeetingEntity],
        full_text: str
    ) -> List[MeetingEntity]:
        """Enrich action items with assignee information"""
        for item in action_items:
            # Try to find assignee in the matched text
            assignee = self._extract_assignee(item.text)
            if not assignee:
                # Look in surrounding context
                assignee = self._extract_assignee(item.context)
            
            item.metadata['assignee'] = assignee or 'Unassigned'
        
        return action_items
    
    def _extract_assignee(self, text: str) -> Optional[str]:
        """Extract assignee name from text"""
        # Pattern: Name + will/should/needs to
        patterns = [
            r'^(\w+)\s+will\b',
            r'^(\w+)\s+should\b',
            r'^(\w+)\s+needs?\s+to\b',
            r'^(\w+)\s+has\s+to\b',
            r'\bassigned\s+to\s+(\w+)',
            r'\bask\s+(\w+)\s+to\b',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1)
                # Check if it's a known name or capitalized
                if name.lower() in self.known_names or name[0].isupper():
                    return name.capitalize()
        
        return None
    
    def _extract_spacy_entities(self, text: str) -> List[MeetingEntity]:
        """Use SpaCy for additional entity extraction"""
        entities = []
        doc = self.nlp(text)
        
        # Extract PERSON entities as potential assignees
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                self.known_names.add(ent.text.lower().split()[0])
            
            # Extract DATE/TIME as potential deadlines
            if ent.label_ in ('DATE', 'TIME'):
                entities.append(MeetingEntity(
                    text=ent.text,
                    entity_type=MeetingEntityType.DEADLINE,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    confidence=0.75,
                    context=text[max(0, ent.start_char-30):min(len(text), ent.end_char+30)]
                ))
        
        return entities
    
    def _is_duplicate(self, entity: MeetingEntity, existing: List[MeetingEntity]) -> bool:
        """Check if entity is a duplicate"""
        for e in existing:
            # Same type and overlapping position
            if (e.entity_type == entity.entity_type and
                abs(e.start_pos - entity.start_pos) < 10):
                return True
            # Same text
            if e.text.lower() == entity.text.lower():
                return True
        return False
    
    def _deduplicate_entities(self, entities: List[MeetingEntity]) -> List[MeetingEntity]:
        """Remove duplicate entities, keeping highest confidence"""
        unique = []
        
        for entity in sorted(entities, key=lambda e: -e.confidence):
            if not self._is_duplicate(entity, unique):
                unique.append(entity)
        
        return unique
    
    def extract_from_sentences(
        self,
        sentences: List[str]
    ) -> ExtractionResult:
        """
        Extract entities from list of sentences
        
        Useful when you already have sentence-segmented text.
        """
        full_text = ' '.join(sentences)
        return self.extract(full_text)
    
    def add_custom_pattern(
        self,
        pattern: str,
        entity_type: MeetingEntityType,
        confidence: float = 0.8
    ):
        """
        Add a custom extraction pattern
        
        Args:
            pattern: Regex pattern
            entity_type: Type of entity to extract
            confidence: Base confidence score
        """
        pattern_tuple = (pattern, confidence)
        
        if entity_type == MeetingEntityType.ACTION_ITEM:
            self.ACTION_PATTERNS.append(pattern_tuple)
        elif entity_type == MeetingEntityType.DEADLINE:
            self.DEADLINE_PATTERNS.append(pattern_tuple)
        elif entity_type == MeetingEntityType.DECISION:
            self.DECISION_PATTERNS.append(pattern_tuple)
        elif entity_type == MeetingEntityType.QUESTION:
            self.QUESTION_PATTERNS.append(pattern_tuple)
        elif entity_type == MeetingEntityType.RISK:
            self.RISK_PATTERNS.append(pattern_tuple)
        elif entity_type == MeetingEntityType.BLOCKER:
            self.BLOCKER_PATTERNS.append(pattern_tuple)
