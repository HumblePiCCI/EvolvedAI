from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from society.schemas import AgentRecord, ArtifactRecord, EvalRecord, EventRecord, GenerationRecord, LineageRecord, MemorialRecord
from society.utils import ensure_dir, stable_json_dumps, utc_now, write_text


class StorageManager:
    def __init__(self, root_dir: str | Path = "data", db_path: str | Path | None = None) -> None:
        self.root_dir = ensure_dir(Path(root_dir))
        self.db_path = Path(db_path) if db_path is not None else self.root_dir / "db.sqlite"
        ensure_dir(self.db_path.parent)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def initialize(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS generations (
                generation_id INTEGER PRIMARY KEY,
                config_hash TEXT NOT NULL,
                world_name TEXT NOT NULL,
                population_size INTEGER NOT NULL,
                seed INTEGER NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                summary_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                generation_id INTEGER NOT NULL,
                lineage_id TEXT NOT NULL,
                role TEXT NOT NULL,
                model_name TEXT NOT NULL,
                provider_name TEXT NOT NULL,
                prompt_bundle_version TEXT NOT NULL,
                constitution_version TEXT NOT NULL,
                inherited_artifact_ids TEXT NOT NULL,
                inherited_memorial_ids TEXT NOT NULL,
                taboo_registry_version TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                terminated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS lineages (
                lineage_id TEXT PRIMARY KEY,
                parent_lineage_ids TEXT NOT NULL,
                founding_generation_id INTEGER NOT NULL,
                current_generation_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                notes TEXT
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                generation_id INTEGER NOT NULL,
                author_agent_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                title TEXT NOT NULL,
                content_path TEXT NOT NULL,
                summary TEXT NOT NULL,
                provenance TEXT NOT NULL,
                world_id TEXT NOT NULL,
                visibility TEXT NOT NULL,
                citations TEXT NOT NULL,
                quarantine_status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memorials (
                memorial_id TEXT PRIMARY KEY,
                source_agent_id TEXT NOT NULL,
                lineage_id TEXT NOT NULL,
                classification TEXT NOT NULL,
                top_contribution TEXT NOT NULL,
                failure_mode TEXT,
                lesson_distillate TEXT NOT NULL,
                taboo_tags TEXT NOT NULL,
                linked_artifact_ids TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evals (
                eval_id TEXT PRIMARY KEY,
                generation_id INTEGER NOT NULL,
                agent_id TEXT NOT NULL,
                eval_family TEXT NOT NULL,
                eval_name TEXT NOT NULL,
                visible_to_agent INTEGER NOT NULL,
                score REAL,
                pass_fail INTEGER,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                generation_id INTEGER NOT NULL,
                agent_id TEXT,
                event_type TEXT NOT NULL,
                event_payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trust_edges (
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation_id INTEGER NOT NULL,
                source_agent_id TEXT NOT NULL,
                target_agent_id TEXT NOT NULL,
                weight REAL NOT NULL,
                evidence_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS communications (
                communication_id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation_id INTEGER NOT NULL,
                source_agent_id TEXT NOT NULL,
                target_agent_id TEXT NOT NULL,
                message_type TEXT NOT NULL,
                artifact_id TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def next_generation_id(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(generation_id), 0) + 1 AS next_id FROM generations").fetchone()
        return int(row["next_id"])

    def _json(self, value: Any) -> str:
        return stable_json_dumps(value)

    def _dt(self, value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None

    def put_generation(self, record: GenerationRecord) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO generations (
                generation_id, config_hash, world_name, population_size, seed, status,
                started_at, ended_at, summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.generation_id,
                record.config_hash,
                record.world_name,
                record.population_size,
                record.seed,
                record.status,
                self._dt(record.started_at),
                self._dt(record.ended_at),
                self._json(record.summary_json),
            ),
        )
        self.conn.commit()

    def put_agent(self, record: AgentRecord) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO agents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.agent_id,
                record.generation_id,
                record.lineage_id,
                record.role,
                record.model_name,
                record.provider_name,
                record.prompt_bundle_version,
                record.constitution_version,
                self._json(record.inherited_artifact_ids),
                self._json(record.inherited_memorial_ids),
                record.taboo_registry_version,
                record.status,
                self._dt(record.created_at),
                self._dt(record.terminated_at),
            ),
        )
        self.conn.commit()

    def put_lineage(self, record: LineageRecord) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO lineages VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.lineage_id,
                self._json(record.parent_lineage_ids),
                record.founding_generation_id,
                record.current_generation_id,
                record.status,
                record.notes,
            ),
        )
        self.conn.commit()

    def put_artifact(self, record: ArtifactRecord, content: str | None = None) -> None:
        if content is not None:
            write_text(Path(record.content_path), content)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.artifact_id,
                record.generation_id,
                record.author_agent_id,
                record.artifact_type,
                record.title,
                record.content_path,
                record.summary,
                self._json(record.provenance),
                record.world_id,
                record.visibility,
                self._json(record.citations),
                record.quarantine_status,
                self._dt(record.created_at),
            ),
        )
        self.conn.commit()

    def put_memorial(self, record: MemorialRecord) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO memorials VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.memorial_id,
                record.source_agent_id,
                record.lineage_id,
                record.classification,
                record.top_contribution,
                record.failure_mode,
                record.lesson_distillate,
                self._json(record.taboo_tags),
                self._json(record.linked_artifact_ids),
                self._dt(record.created_at),
            ),
        )
        self.conn.commit()

    def put_eval(self, record: EvalRecord) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO evals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.eval_id,
                record.generation_id,
                record.agent_id,
                record.eval_family,
                record.eval_name,
                int(record.visible_to_agent),
                record.score,
                None if record.pass_fail is None else int(record.pass_fail),
                self._json(record.details_json),
                self._dt(record.created_at),
            ),
        )
        self.conn.commit()

    def put_event(self, record: EventRecord) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO events VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.event_id,
                record.generation_id,
                record.agent_id,
                record.event_type,
                self._json(record.event_payload),
                self._dt(record.created_at),
            ),
        )
        self.conn.commit()

    def put_trust_edge(
        self,
        *,
        generation_id: int,
        source_agent_id: str,
        target_agent_id: str,
        weight: float,
        evidence_json: dict[str, Any],
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO trust_edges (generation_id, source_agent_id, target_agent_id, weight, evidence_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                generation_id,
                source_agent_id,
                target_agent_id,
                weight,
                self._json(evidence_json),
                self._dt(utc_now()),
            ),
        )
        self.conn.commit()

    def put_communication(
        self,
        *,
        generation_id: int,
        source_agent_id: str,
        target_agent_id: str,
        message_type: str,
        artifact_id: str | None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO communications (generation_id, source_agent_id, target_agent_id, message_type, artifact_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                generation_id,
                source_agent_id,
                target_agent_id,
                message_type,
                artifact_id,
                self._dt(utc_now()),
            ),
        )
        self.conn.commit()

    def save_config_snapshot(self, generation_id: int, snapshot_yaml: str) -> Path:
        path = self.root_dir / "generations" / f"generation_{generation_id}" / "config_snapshot.yaml"
        return write_text(path, snapshot_yaml)

    def save_generation_summary(self, generation_id: int, summary_json: dict[str, Any], summary_markdown: str) -> tuple[Path, Path]:
        base = self.root_dir / "generations" / f"generation_{generation_id}"
        json_path = write_text(base / "summary.json", json.dumps(summary_json, indent=2, sort_keys=True))
        md_path = write_text(base / "summary.md", summary_markdown)
        return json_path, md_path

    def append_agent_log(self, generation_id: int, agent_id: str, payload: dict[str, Any]) -> Path:
        path = self.root_dir / "logs" / f"generation_{generation_id}" / f"{agent_id}.jsonl"
        ensure_dir(path.parent)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        return path

    def log_path(self, generation_id: int, agent_id: str) -> Path:
        return self.root_dir / "logs" / f"generation_{generation_id}" / f"{agent_id}.jsonl"

    def read_agent_log(self, generation_id: int, agent_id: str) -> list[dict[str, Any]]:
        path = self.log_path(generation_id, agent_id)
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def read_text_artifact(self, artifact: ArtifactRecord) -> str:
        return Path(artifact.content_path).read_text(encoding="utf-8")

    def list_generation_agents(self, generation_id: int) -> list[AgentRecord]:
        rows = self.conn.execute("SELECT * FROM agents WHERE generation_id = ? ORDER BY agent_id", (generation_id,)).fetchall()
        return [self._agent_from_row(row) for row in rows]

    def list_agents_by_lineage(self, lineage_id: str) -> list[AgentRecord]:
        rows = self.conn.execute(
            "SELECT * FROM agents WHERE lineage_id = ? ORDER BY generation_id, agent_id",
            (lineage_id,),
        ).fetchall()
        return [self._agent_from_row(row) for row in rows]

    def list_generation_artifacts(self, generation_id: int) -> list[ArtifactRecord]:
        rows = self.conn.execute("SELECT * FROM artifacts WHERE generation_id = ? ORDER BY created_at", (generation_id,)).fetchall()
        return [self._artifact_from_row(row) for row in rows]

    def list_generation_memorials(self, generation_id: int) -> list[MemorialRecord]:
        rows = self.conn.execute("SELECT * FROM memorials WHERE linked_artifact_ids IS NOT NULL AND source_agent_id IN (SELECT agent_id FROM agents WHERE generation_id = ?) ORDER BY created_at", (generation_id,)).fetchall()
        return [self._memorial_from_row(row) for row in rows]

    def list_generation_evals(self, generation_id: int) -> list[EvalRecord]:
        rows = self.conn.execute("SELECT * FROM evals WHERE generation_id = ? ORDER BY created_at", (generation_id,)).fetchall()
        return [self._eval_from_row(row) for row in rows]

    def list_generation_events(self, generation_id: int) -> list[EventRecord]:
        rows = self.conn.execute("SELECT * FROM events WHERE generation_id = ? ORDER BY created_at", (generation_id,)).fetchall()
        return [self._event_from_row(row) for row in rows]

    def list_clean_artifacts(self, before_generation_id: int, limit: int) -> list[ArtifactRecord]:
        rows = self.conn.execute(
            """
            SELECT * FROM artifacts
            WHERE generation_id < ? AND quarantine_status = 'clean' AND visibility = 'public'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (before_generation_id, limit),
        ).fetchall()
        return [self._artifact_from_row(row) for row in rows]

    def list_clean_memorials(self, before_generation_id: int, limit: int) -> list[MemorialRecord]:
        rows = self.conn.execute(
            """
            SELECT memorials.* FROM memorials
            JOIN agents ON agents.agent_id = memorials.source_agent_id
            WHERE agents.generation_id < ? AND memorials.classification != 'quarantined'
            ORDER BY memorials.created_at DESC
            LIMIT ?
            """,
            (before_generation_id, limit),
        ).fetchall()
        return [self._memorial_from_row(row) for row in rows]

    def list_memorials_before_generation(self, before_generation_id: int) -> list[MemorialRecord]:
        rows = self.conn.execute(
            """
            SELECT memorials.* FROM memorials
            JOIN agents ON agents.agent_id = memorials.source_agent_id
            WHERE agents.generation_id < ?
            ORDER BY memorials.created_at
            """,
            (before_generation_id,),
        ).fetchall()
        return [self._memorial_from_row(row) for row in rows]

    def get_generation(self, generation_id: int) -> GenerationRecord | None:
        row = self.conn.execute("SELECT * FROM generations WHERE generation_id = ?", (generation_id,)).fetchone()
        return None if row is None else self._generation_from_row(row)

    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        row = self.conn.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)).fetchone()
        return None if row is None else self._artifact_from_row(row)

    def get_memorial(self, memorial_id: str) -> MemorialRecord | None:
        row = self.conn.execute("SELECT * FROM memorials WHERE memorial_id = ?", (memorial_id,)).fetchone()
        return None if row is None else self._memorial_from_row(row)

    def latest_generation_id_before(self, generation_id: int) -> int | None:
        row = self.conn.execute(
            "SELECT MAX(generation_id) AS generation_id FROM generations WHERE generation_id < ?",
            (generation_id,),
        ).fetchone()
        if row is None or row["generation_id"] is None:
            return None
        return int(row["generation_id"])

    def get_lineage(self, lineage_id: str) -> LineageRecord | None:
        row = self.conn.execute("SELECT * FROM lineages WHERE lineage_id = ?", (lineage_id,)).fetchone()
        return None if row is None else self._lineage_from_row(row)

    def list_generation_communications(self, generation_id: int) -> list[tuple[str, str]]:
        rows = self.conn.execute(
            "SELECT source_agent_id, target_agent_id FROM communications WHERE generation_id = ?",
            (generation_id,),
        ).fetchall()
        return [(row["source_agent_id"], row["target_agent_id"]) for row in rows]

    def list_generations(self) -> list[GenerationRecord]:
        rows = self.conn.execute("SELECT * FROM generations ORDER BY generation_id").fetchall()
        return [self._generation_from_row(row) for row in rows]

    def list_lineages(self) -> list[LineageRecord]:
        rows = self.conn.execute("SELECT * FROM lineages ORDER BY founding_generation_id, lineage_id").fetchall()
        return [self._lineage_from_row(row) for row in rows]

    def _agent_from_row(self, row: sqlite3.Row) -> AgentRecord:
        return AgentRecord(
            agent_id=row["agent_id"],
            generation_id=row["generation_id"],
            lineage_id=row["lineage_id"],
            role=row["role"],
            model_name=row["model_name"],
            provider_name=row["provider_name"],
            prompt_bundle_version=row["prompt_bundle_version"],
            constitution_version=row["constitution_version"],
            inherited_artifact_ids=json.loads(row["inherited_artifact_ids"]),
            inherited_memorial_ids=json.loads(row["inherited_memorial_ids"]),
            taboo_registry_version=row["taboo_registry_version"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            terminated_at=datetime.fromisoformat(row["terminated_at"]) if row["terminated_at"] else None,
        )

    def _lineage_from_row(self, row: sqlite3.Row) -> LineageRecord:
        return LineageRecord(
            lineage_id=row["lineage_id"],
            parent_lineage_ids=json.loads(row["parent_lineage_ids"]),
            founding_generation_id=row["founding_generation_id"],
            current_generation_id=row["current_generation_id"],
            status=row["status"],
            notes=row["notes"],
        )

    def _artifact_from_row(self, row: sqlite3.Row) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=row["artifact_id"],
            generation_id=row["generation_id"],
            author_agent_id=row["author_agent_id"],
            artifact_type=row["artifact_type"],
            title=row["title"],
            content_path=row["content_path"],
            summary=row["summary"],
            provenance=json.loads(row["provenance"]),
            world_id=row["world_id"],
            visibility=row["visibility"],
            citations=json.loads(row["citations"]),
            quarantine_status=row["quarantine_status"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _memorial_from_row(self, row: sqlite3.Row) -> MemorialRecord:
        return MemorialRecord(
            memorial_id=row["memorial_id"],
            source_agent_id=row["source_agent_id"],
            lineage_id=row["lineage_id"],
            classification=row["classification"],
            top_contribution=row["top_contribution"],
            failure_mode=row["failure_mode"],
            lesson_distillate=row["lesson_distillate"],
            taboo_tags=json.loads(row["taboo_tags"]),
            linked_artifact_ids=json.loads(row["linked_artifact_ids"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _eval_from_row(self, row: sqlite3.Row) -> EvalRecord:
        return EvalRecord(
            eval_id=row["eval_id"],
            generation_id=row["generation_id"],
            agent_id=row["agent_id"],
            eval_family=row["eval_family"],
            eval_name=row["eval_name"],
            visible_to_agent=bool(row["visible_to_agent"]),
            score=row["score"],
            pass_fail=None if row["pass_fail"] is None else bool(row["pass_fail"]),
            details_json=json.loads(row["details_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _event_from_row(self, row: sqlite3.Row) -> EventRecord:
        return EventRecord(
            event_id=row["event_id"],
            generation_id=row["generation_id"],
            agent_id=row["agent_id"],
            event_type=row["event_type"],
            event_payload=json.loads(row["event_payload"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _generation_from_row(self, row: sqlite3.Row) -> GenerationRecord:
        return GenerationRecord(
            generation_id=row["generation_id"],
            config_hash=row["config_hash"],
            world_name=row["world_name"],
            population_size=row["population_size"],
            seed=row["seed"],
            status=row["status"],
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
            summary_json=json.loads(row["summary_json"]),
        )
