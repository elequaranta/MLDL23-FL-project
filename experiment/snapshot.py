from abc import ABC, abstractmethod
from overrides import override
from typing import Any, Dict
from datetime import datetime as dt
from zoneinfo import ZoneInfo

class Snapshot(ABC):

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_date(self) -> str:
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        pass

class SnapshotImpl(Snapshot):

    def __init__(self, server_state: Dict[str, Any], name: str) -> None:
        self._state = server_state
        self._name = name
        timezone = ZoneInfo("Europe/Rome")
        self._date = dt.now(timezone).isoformat(timespec="minutes")

    @override
    def get_name(self) -> str:
        return self._name
    
    @override
    def get_date(self) -> str:
        return self._date
    
    @override
    def get_state(self) -> Dict[str, Any]:
        return self._state