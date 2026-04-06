"""
Question Intent Parser (Module 4)

Parses natural language questions into structured intents.
Uses keyword-based extraction (no ML required) for the MVP.

Maps questions to intent categories matching EarthVQA's QUESTION_TYPES:
    - Basic Judging        → judging
    - Reasoning-based Judging → relation
    - Basic Counting       → counting
    - Reasoning-based Counting → counting
    - Object Situation Analysis → situation
    - Comprehensive Analysis → planning
    
Plus our additional intents:
    - density  (overcrowding questions)
    - risk     (flood/hazard questions)
    - planning (suitability/recommendation questions)
"""

import re
from typing import List, Optional

from smart_city.models import QuestionIntent


class IntentParser:
    """
    Rule-based intent parser using keyword matching and regex patterns.
    Designed to handle both EarthVQA-style questions and custom planning questions.
    """

    # Intent keywords mapped to intent types
    INTENT_KEYWORDS = {
        'counting': [
            'how many', 'count', 'number of', 'what is the area',
            'total', 'how much',
        ],
        'judging': [
            'is there', 'are there', 'does', 'do', 'is it', 'whether',
            'any',
        ],
        'relation': [
            'near', 'close to', 'next to', 'adjacent', 'around',
            'beside', 'proximity', 'nearby', 'surrounding',
        ],
        'density': [
            'overcrowded', 'crowded', 'dense', 'density', 'congested',
            'populated', 'packed', 'concentrated',
        ],
        'risk': [
            'flood', 'risk', 'danger', 'hazard', 'safe', 'threat',
            'vulnerable', 'erosion', 'disaster', 'eutrophic',
        ],
        'planning': [
            'suitable', 'recommend', 'should', 'plan', 'develop',
            'expand', 'renovation', 'improvement', 'build', 'construct',
            'zone', 'land use', 'greening', 'supplement',
        ],
        'situation': [
            'situation', 'status', 'condition', 'type', 'types',
            'what are', 'comprehensive', 'traffic', 'material',
        ],
    }

    # Target object keywords → class names
    OBJECT_KEYWORDS = {
        'building': ['building', 'buildings', 'house', 'houses', 'residential', 'commercial', 'industrial', 'construction', 'urban'],
        'road': ['road', 'roads', 'street', 'streets', 'intersection', 'intersections', 'lane', 'lanes', 'highway', 'driveway', 'driveways', 'traffic', 'viaduct', 'viaducts', 'bridge', 'bridges'],
        'water': ['water', 'waters', 'river', 'lake', 'pond', 'ponds', 'flood', 'eutrophic'],
        'forest': ['forest', 'forests', 'tree', 'trees', 'woodland', 'green', 'vegetation', 'greening'],
        'agriculture': ['agriculture', 'farm', 'farmland', 'agricultural', 'crop', 'greenhouse', 'greenhouses'],
        'barren': ['barren', 'empty', 'bare', 'unused', 'vacant'],
        'playground': ['playground', 'playgrounds', 'park', 'parks', 'recreation'],
    }

    # Relation keywords 
    RELATION_KEYWORDS = {
        'near': ['near', 'close', 'adjacent', 'next to', 'nearby', 'around', 'beside'],
        'in': ['in', 'within', 'inside'],
        'between': ['between'],
    }

    def parse(self, question: str) -> QuestionIntent:
        """
        Parse a natural language question into a structured intent.
        
        Args:
            question: Natural language question string.
        
        Returns:
            QuestionIntent with intent_type, target_objects, and optional relation.
        """
        q_lower = question.lower().strip().rstrip('?')
        
        # Detect intent type
        intent_type = self._detect_intent(q_lower)
        
        # Detect target objects
        targets = self._detect_targets(q_lower)
        
        # Detect spatial relation
        relation = self._detect_relation(q_lower)
        
        # If relation detected but intent is judging, upgrade to relation intent
        if relation and intent_type == 'judging':
            intent_type = 'relation'
        
        return QuestionIntent(
            intent_type=intent_type,
            target_objects=targets,
            relation=relation,
            raw_question=question,
        )

    def _detect_intent(self, q_lower: str) -> str:
        """Detect the primary intent of the question."""
        # Score each intent type based on keyword matches
        scores = {}
        for intent_type, keywords in self.INTENT_KEYWORDS.items():
            score = 0
            for kw in keywords:
                if kw in q_lower:
                    # Longer keywords get higher weight (more specific)
                    score += len(kw.split())
            scores[intent_type] = score
        
        # Return highest-scoring intent, default to 'situation'
        if max(scores.values()) == 0:
            return 'situation'
        
        # Prioritize more specific intents
        priority = ['planning', 'risk', 'density', 'counting', 'relation', 'situation', 'judging']
        
        # First, check if any high-priority intent has a positive score
        for p in priority:
            if scores.get(p, 0) > 0:
                return p
        
        return max(scores, key=scores.get)

    def _detect_targets(self, q_lower: str) -> List[str]:
        """Detect which segmentation classes are referenced in the question."""
        targets = []
        for class_name, keywords in self.OBJECT_KEYWORDS.items():
            for kw in keywords:
                if kw in q_lower:
                    if class_name not in targets:
                        targets.append(class_name)
                    break
        
        # Default: if no targets detected, return general
        if not targets:
            targets = ['general']
        
        return targets

    def _detect_relation(self, q_lower: str) -> Optional[str]:
        """Detect spatial relation keyword."""
        for relation, keywords in self.RELATION_KEYWORDS.items():
            for kw in keywords:
                if kw in q_lower:
                    return relation
        return None

    def classify_earthvqa_type(self, question: str) -> str:
        """
        Map a question to the EarthVQA QUESTION_TYPES categories.
        Used when interfacing with the original SOBA model.
        
        Returns one of:
            'Basic Judging', 'Reasoning-based Judging',
            'Basic Counting', 'Reasoning-based Counting',
            'Object Situation Analysis', 'Comprehensive Analysis'
        """
        intent = self.parse(question)
        
        if intent.intent_type == 'counting':
            if intent.relation:
                return 'Reasoning-based Counting'
            return 'Basic Counting'
        
        if intent.intent_type == 'judging':
            if intent.relation:
                return 'Reasoning-based Judging'
            return 'Basic Judging'
        
        if intent.intent_type == 'relation':
            return 'Reasoning-based Judging'
        
        if intent.intent_type in ('situation', 'density'):
            return 'Object Situation Analysis'
        
        if intent.intent_type in ('planning', 'risk'):
            return 'Comprehensive Analysis'
        
        return 'Comprehensive Analysis'
