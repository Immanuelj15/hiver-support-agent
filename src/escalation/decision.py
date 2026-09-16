"""
src/escalation/decision.py

Data classes and enums for deterministic customer-support escalation decisions.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional

class EscalationAction(str, Enum):
    AUTO_HANDLE = "AUTO_HANDLE"
    ESCALATE = "ESCALATE"

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class EscalationDecision:
    action: EscalationAction
    should_escalate: bool
    reason: str
    risk_level: RiskLevel
    triggered_rule: Optional[str] = None
