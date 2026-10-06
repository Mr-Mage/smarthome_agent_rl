"""SQLite task-runtime storage with atomic changes and optimistic revisions."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sqlite3


class RevisionConflict(RuntimeError):
    pass


class RuntimeStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS records (
                kind TEXT NOT NULL, id TEXT NOT NULL, revision INTEGER NOT NULL,
                data TEXT NOT NULL, PRIMARY KEY(kind, id))''')

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def get(self, kind, id, *, db=None):
        if db is None:
            with self.transaction() as conn:
                return self.get(kind, id, db=conn)
        row = db.execute('SELECT revision,data FROM records WHERE kind=? AND id=?', (kind, id)).fetchone()
        if row is None:
            raise KeyError((kind, id))
        return json.loads(row['data']), row['revision']

    def put(self, kind, id, data, *, expected_revision=None, db=None):
        if db is None:
            with self.transaction() as conn:
                return self.put(kind, id, data, expected_revision=expected_revision, db=conn)
        payload = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if expected_revision is None:
            db.execute('INSERT INTO records VALUES (?, ?, 1, ?)', (kind, id, payload))
            return 1
        if kind in ('trace','conflict'):
            raise ValueError('Invocation and conflict evidence are append-only')
        cursor = db.execute('UPDATE records SET revision=revision+1,data=? WHERE kind=? AND id=? AND revision=?',
                            (payload, kind, id, expected_revision))
        if cursor.rowcount != 1:
            raise RevisionConflict(f'Stale {kind} revision: {id}')
        return expected_revision + 1

    def list(self, kind, *, db=None):
        if db is None:
            with self.transaction() as conn:
                return self.list(kind, db=conn)
        return [json.loads(row['data']) for row in db.execute(
            'SELECT data FROM records WHERE kind=? ORDER BY id', (kind,))]

    def trace_sink(self, row):
        self.put('trace', row['invocation_id'], copy.deepcopy(row))
