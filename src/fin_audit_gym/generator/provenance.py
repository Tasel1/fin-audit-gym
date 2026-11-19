"""Provenance tracking engine for synthetic financial data lineage."""

from collections import defaultdict
from typing import Any


class ProvenanceTracker:
    """Tracks lineage, causation, and references across synthetic generation events."""

    def __init__(self) -> None:
        self.transactions: dict[str, dict[str, Any]] = {}
        self.entity_map: dict[str, list[str]] = defaultdict(list)

    def record_transaction(
        self,
        transaction_id: str,
        tx_type: str,
        entity_ids: list[str],
        ledger_entry_ids: list[str],
        timestamp: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a single causal business transaction event."""
        record = {
            "transaction_id": transaction_id,
            "tx_type": tx_type,
            "entity_ids": list(entity_ids),
            "ledger_entry_ids": list(ledger_entry_ids),
            "timestamp": timestamp,
            "metadata": dict(metadata) if metadata else {},
        }
        self.transactions[transaction_id] = record

        for entity_id in entity_ids:
            if transaction_id not in self.entity_map[entity_id]:
                self.entity_map[entity_id].append(transaction_id)

    def get_lineage(self, entity_id: str) -> list[dict[str, Any]]:
        """Retrieve full causal history for a specific entity ID."""
        txn_ids = self.entity_map.get(entity_id, [])
        return [self.transactions[tid] for tid in txn_ids if tid in self.transactions]

    def to_dict(self) -> dict[str, Any]:
        """Export internal graph to a serializable dictionary."""
        return {
            "transactions": self.transactions,
            "entity_map": dict(self.entity_map),
        }
