# marmo_dataset.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
import re
import json
import traceback
import glob
import time as _time

try:
    import yaml  # pip install pyyaml
except ImportError as e:
    raise ImportError("PyYAML が必要です: pip install pyyaml") from e


# -----------------------------
# Spec loaders / helpers
# -----------------------------

def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _now_iso() -> str:
    return _time.strftime("%Y-%m-%dT%H:%M:%S%z")

def _expand_templates(s: str, *, project_root: Path) -> str:
    return s.replace("{PROJECT_ROOT}", str(project_root))

def _glob1(pattern: str) -> Optional[Path]:
    hits = [Path(p) for p in glob.glob(pattern)]
    if not hits:
        return None
    # deterministic
    return sorted(hits)[0]

def _glob_all(pattern: str) -> List[Path]:
    return sorted(Path(p) for p in glob.glob(pattern))

def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


# -----------------------------
# Data models
# -----------------------------

@dataclass(frozen=True)
class SkipRecord:
    timestamp: str
    session_id: str
    camid_missing: List[str] = field(default_factory=list)
    missing_files: List[str] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)
    exception: Optional[str] = None
    traceback: Optional[str] = None


@dataclass
class IntegrityReport:
    ok: bool
    session_id: str
    missing_camids: List[str] = field(default_factory=list)
    missing_files: List[str] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)

    def to_skip_record(self, exc: Optional[BaseException] = None) -> SkipRecord:
        return SkipRecord(
            timestamp=_now_iso(),
            session_id=self.session_id,
            camid_missing=self.missing_camids,
            missing_files=self.missing_files,
            failed_checks=self.failed_checks,
            exception=(repr(exc) if exc else None),
            traceback=("".join(traceback.format_exception(exc)) if exc else None),
        )


@dataclass(frozen=True)
class SessionIdParts:
    prefix: str
    animal: str
    date: str
    time: str


@dataclass
class HealthStatusDB:
    meta: Dict[str, Any]
    animals: Dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "HealthStatusDB":
        d = _read_yaml(path)
        return cls(meta=d.get("meta", {}), animals=d.get("animals", {}))

    def status_on(self, *, animal: str, date: str) -> str:
        """
        returns: healthy / diseased / unknown (or meta.default_status)
        """
        default_status = self.meta.get("default_status", "unknown")
        entry = self.animals.get(animal)
        if not entry:
            return default_status
        periods = entry.get("periods", [])
        # inclusive period match: from <= date <= to (to may be null)
        for p in periods:
            frm = p.get("from")
            to = p.get("to")
            if frm and date < frm:
                continue
            if to is not None and to != "" and date > to:
                continue
            return p.get("status", default_status)
        return default_status


@dataclass
class DatasetSpec:
    raw: Dict[str, Any]
    project_root: Path

    # compiled
    session_id_re: re.Pattern
    video_folder_id_re: re.Pattern

    @classmethod
    def load(cls, spec_path: Path, *, project_root: Path) -> "DatasetSpec":
        raw = _read_yaml(spec_path)
        ids = raw["dataset"]["ids"]
        session_id_re = re.compile(ids["session_id_regex"])
        video_folder_id_re = re.compile(ids["video_folder_id_regex"])
        return cls(raw=raw, project_root=project_root, session_id_re=session_id_re, video_folder_id_re=video_folder_id_re)

    # ---- Convenience accessors ----
    @property
    def dataset(self) -> Dict[str, Any]:
        return self.raw["dataset"]

    @property
    def cameras(self) -> Dict[str, Any]:
        return self.dataset["cameras"]

    @property
    def camera_allowed_ids(self) -> List[str]:
        return list(self.cameras.get("allowed_ids", []))

    @property
    def expected_cam_count(self) -> int:
        return int(self.cameras.get("expected_count", 8))

    @property
    def roots_video(self) -> List[Path]:
        roots = self.dataset["roots"]["video_roots"]
        return [Path(_expand_templates(r, project_root=self.project_root)) for r in roots]

    @property
    def roots_results(self) -> Path:
        return Path(_expand_templates(self.dataset["roots"]["results_root"], project_root=self.project_root))

    @property
    def results_subdirs_priority(self) -> List[str]:
        return list(self.dataset["discovery"]["results_subdirs"])

    @property
    def logging_path(self) -> Path:
        p = self.dataset["logging"]["path"]
        return Path(_expand_templates(p, project_root=self.project_root))

    @property
    def timebase(self) -> Dict[str, Any]:
        return self.dataset["timebase"]

    @property
    def artifacts(self) -> Dict[str, Any]:
        return self.dataset["artifacts"]

    @property
    def integrity(self) -> Dict[str, Any]:
        return self.dataset["integrity"]

    @property
    def partitioning(self) -> Dict[str, Any]:
        return self.dataset.get("partitioning", {})


# -----------------------------
# Dataset / Session objects
# -----------------------------

@dataclass(frozen=True)
class VideoCam:
    camid: str
    video_root: Path
    video_folder_id: str  # "{session_id}.{camid}"

    @property
    def folder(self) -> Path:
        return self.video_root / self.video_folder_id

    def metadata_path(self, spec: DatasetSpec) -> Path:
        # spec.layout.metadata.path_template
        tmpl = spec.dataset["layout"]["metadata"]["path_template"]
        tmpl = _expand_templates(tmpl, project_root=spec.project_root)
        return Path(tmpl.format(video_root=str(self.video_root), video_folder_id=self.video_folder_id))

    def mp4_chunks(self, spec: DatasetSpec) -> List[Path]:
        # {chunk_index:06d}.mp4
        pat = str(self.folder / "*.mp4")
        return sorted(Path(p) for p in glob.glob(pat))

    def npz_chunks(self, spec: DatasetSpec) -> List[Path]:
        pat = str(self.folder / "*.npz")
        return sorted(Path(p) for p in glob.glob(pat))


@dataclass
class Session:
    spec: DatasetSpec
    session_id: str
    parts: SessionIdParts
    video_root: Path
    results_root: Path
    results_subdir: str
    health_db: Optional[HealthStatusDB] = None

    def cam(self, camid: str) -> VideoCam:
        video_folder_id = f"{self.session_id}.{camid}"
        return VideoCam(camid=camid, video_root=self.video_root, video_folder_id=video_folder_id)

    @property
    def animal(self) -> str:
        return self.parts.animal

    @property
    def date(self) -> str:
        return self.parts.date

    @property
    def time(self) -> str:
        return self.parts.time

    def health_status(self) -> str:
        if not self.health_db:
            return "unknown"
        # health_status.yaml のキーは例で cj611 / cj425 なので、
        # dataset の animal が "611" の場合 "cj611" に寄せる（仕様に合わせて調整可）
        key = self.animal
        if not key.startswith("cj"):
            key = "cj" + key
        return self.health_db.status_on(animal=key, date=self.date)

    # ---- Artifacts ----
    def results_folder(self) -> Path:
        return self.results_root / self.results_subdir / self.session_id

    def _select_kp3d_path(self) -> Optional[Path]:
        kp3d_spec = self.spec.artifacts["kp3d"]
        templates = kp3d_spec["path_templates"]
        prefer_contains = kp3d_spec["selection"].get("prefer_contains", [])
        fallback = kp3d_spec["selection"].get("fallback", "first_match")

        candidates: List[Path] = []
        for t in templates:
            tt = _expand_templates(t, project_root=self.spec.project_root)
            pattern = tt.format(
                results_root=str(self.results_root),
                results_subdir=self.results_subdir,
                results_folder_id=self.session_id,
            )
            # allow glob in template (kp3d*.pickle)
            if "*" in pattern or "?" in pattern or "[" in pattern:
                candidates.extend(_glob_all(pattern))
            else:
                p = Path(pattern)
                if p.exists():
                    candidates.append(p)

        if not candidates:
            return None

        # prefer_contains
        for needle in prefer_contains:
            hit = [p for p in candidates if needle in p.name]
            if hit:
                return sorted(hit)[0]

        # fallback
        if fallback == "latest_mtime":
            return max(candidates, key=lambda p: p.stat().st_mtime)
        return sorted(candidates)[0]

    def kp3d_path(self) -> Optional[Path]:
        return self._select_kp3d_path()

    def kp2d_paths(self) -> List[Path]:
        kp2d_spec = self.spec.artifacts["kp2d"]
        hits: List[Path] = []
        for t in kp2d_spec.get("any_of", []):
            tt = _expand_templates(t, project_root=self.spec.project_root)
            pattern = tt.format(
                results_root=str(self.results_root),
                results_subdir=self.results_subdir,
                results_folder_id=self.session_id,
            )
            if "*" in pattern or "?" in pattern or "[" in pattern:
                hits.extend(_glob_all(pattern))
            else:
                p = Path(pattern)
                if p.exists():
                    hits.append(p)
        # unique + deterministic
        uniq = {str(p): p for p in hits}
        return sorted(uniq.values())

    # ---- Time mapping (24fps -> stored 25fps) ----
    def frame_24_to_25(self, frame_24: int) -> int:
        tb = self.spec.timebase
        fps25 = float(tb["video_fps_stored"])
        fps24 = float(tb["processing_fps"])
        # t25 = round(t24 * 25/24) with nearest
        return int(round(frame_24 * fps25 / fps24))

    # ---- Integrity ----
    def check_integrity(self) -> IntegrityReport:
        integ = self.spec.integrity
        camids = self.spec.camera_allowed_ids
        rep = IntegrityReport(ok=True, session_id=self.session_id)

        # required_video
        rv = integ["required_video"]
        if rv.get("must_have_all_cameras", True):
            missing = []
            for camid in camids:
                if not self.cam(camid).folder.exists():
                    missing.append(camid)
            if missing:
                rep.ok = False
                rep.missing_camids.extend(missing)
                rep.failed_checks.append("must_have_all_cameras")

        if rv.get("must_have_metadata", True):
            # check metadata for each existing camera folder (or all cameras if strict)
            for camid in camids:
                vc = self.cam(camid)
                md = vc.metadata_path(self.spec)
                if vc.folder.exists() and not md.exists():
                    rep.ok = False
                    rep.missing_files.append(str(md))
                    rep.failed_checks.append("must_have_metadata")

        if rv.get("must_have_mp4_at_least_one_chunk", True):
            for camid in camids:
                vc = self.cam(camid)
                if vc.folder.exists():
                    mp4s = vc.mp4_chunks(self.spec)
                    if len(mp4s) < 1:
                        rep.ok = False
                        rep.failed_checks.append("must_have_mp4_at_least_one_chunk")
                        rep.missing_files.append(f"{vc.folder}/*.mp4")

        if rv.get("must_have_npz_count_match_mp4", True):
            for camid in camids:
                vc = self.cam(camid)
                if vc.folder.exists():
                    mp4s = vc.mp4_chunks(self.spec)
                    npzs = vc.npz_chunks(self.spec)
                    if len(mp4s) != 0 and len(npzs) != len(mp4s):
                        rep.ok = False
                        rep.failed_checks.append("must_have_npz_count_match_mp4")
                        rep.missing_files.append(f"{vc.folder}/*.npz (expected {len(mp4s)})")

        # required_results
        rr = integ["required_results"]
        if rr.get("must_have_kp3d_at_least_one", True):
            if self.kp3d_path() is None:
                rep.ok = False
                rep.failed_checks.append("must_have_kp3d_at_least_one")
                rep.missing_files.append(f"{self.results_folder()}/kp3d*.pickle")

        # sync_check is defined in spec; implement a pragmatic check:
        # compare total frames per cam using mp4 count * nominal(10000), and ensure diff <= max_frame_diff
        sc = integ.get("sync_check", {})
        if sc.get("enabled", False):
            maxdiff = int(sc.get("max_frame_diff", 5000))
            totals: Dict[str, int] = {}
            nominal = int(self.spec.dataset["layout"]["video_chunks"]["chunk_frames_nominal"])
            for camid in camids:
                vc = self.cam(camid)
                if vc.folder.exists():
                    mp4_count = len(vc.mp4_chunks(self.spec))
                    totals[camid] = mp4_count * nominal
            if totals:
                mn, mx = min(totals.values()), max(totals.values())
                if (mx - mn) > maxdiff:
                    rep.ok = False
                    rep.failed_checks.append("sync_check_max_frame_diff")
                    rep.missing_files.append(f"frame_diff={mx-mn} > {maxdiff}")

        return rep


@dataclass
class MarmoDataset:
    spec: DatasetSpec
    video_root: Path
    results_root: Path
    results_subdir: str
    health_db: Optional[HealthStatusDB] = None
    ignored_sessions: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(
        cls,
        spec_path: Path,
        *,
        project_root: Path,
    ) -> "MarmoDataset":
        spec = DatasetSpec.load(spec_path, project_root=project_root)

        video_root = _first_existing(spec.roots_video)
        if not video_root:
            raise FileNotFoundError(f"video_roots が見つかりません: {spec.roots_video}")

        results_root = spec.roots_results
        if not results_root.exists():
            raise FileNotFoundError(f"results_root が見つかりません: {results_root}")

        # choose results_subdir by priority
        found_subdir = None
        for sub in spec.results_subdirs_priority:
            p = results_root / sub
            if p.exists():
                found_subdir = sub
                break
        if not found_subdir:
            raise FileNotFoundError(
                f"results_subdir が見つかりません: {[(results_root / s) for s in spec.results_subdirs_priority]}"
            )

        # health DB (optional)
        health_db = None
        hs_cfg = spec.partitioning.get("health_status", {})
        if hs_cfg.get("enabled", False):
            hs_path = Path(_expand_templates(hs_cfg["source"], project_root=project_root))
            if hs_path.exists():
                health_db = HealthStatusDB.load(hs_path)

        # ignored sessions (optional)
        ignored_sessions = []
        ignore_path = project_root / "config" / "ignore.yaml"
        if ignore_path.exists():
            ignore_data = _read_yaml(ignore_path)
            ignored_sessions = ignore_data.get("ignored_sessions", [])

        return cls(
            spec=spec,
            video_root=video_root,
            results_root=results_root,
            results_subdir=found_subdir,
            health_db=health_db,
            ignored_sessions=ignored_sessions,
        )

    # ---- ID parse/build ----
    def parse_session_id(self, session_id: str) -> SessionIdParts:
        m = self.spec.session_id_re.match(session_id)
        if not m:
            raise ValueError(f"session_id_regex にマッチしません: {session_id}")
        return SessionIdParts(
            prefix=m.group("prefix"),
            animal=m.group("animal"),
            date=m.group("date"),
            time=m.group("time"),
        )

    def session(self, session_id: str) -> Session:
        parts = self.parse_session_id(session_id)
        return Session(
            spec=self.spec,
            session_id=session_id,
            parts=parts,
            video_root=self.video_root,
            results_root=self.results_root,
            results_subdir=self.results_subdir,
            health_db=self.health_db,
        )

    # ---- Ignored sessions ----
    def is_session_ignored(self, session_id: str) -> bool:
        """セッションがignore.yamlで除外されているかチェック"""
        return session_id in self.ignored_sessions

    # ---- Discovery ----
    def discover_session_ids(self) -> List[str]:
        """
        基本は results 側から見つける（kp3d が必須なので）。
        results_root/results_subdir/* のディレクトリ名を session_id とみなす。
        """
        base = self.results_root / self.results_subdir
        if not base.exists():
            return []
        out: List[str] = []
        for p in sorted(base.iterdir()):
            if not p.is_dir():
                continue
            sid = p.name
            if self.spec.session_id_re.match(sid):
                out.append(sid)
        return out

    def iter_sessions(self, *, valid_only: bool = True, log_skips: bool = True) -> Iterator[Session]:
        logger = SkipLogger(self.spec.logging_path) if log_skips else None
        for sid in self.discover_session_ids():
            # Skip ignored sessions
            if self.is_session_ignored(sid):
                if logger:
                    skip_rec = SkipRecord(
                        timestamp=_now_iso(),
                        session_id=sid,
                        failed_checks=["ignored_session"],
                    )
                    logger.write(skip_rec)
                continue

            s = self.session(sid)
            try:
                rep = s.check_integrity()
                if valid_only and not rep.ok:
                    if logger:
                        logger.write(rep.to_skip_record())
                    continue
                yield s
            except Exception as e:
                if logger:
                    logger.write(IntegrityReport(ok=False, session_id=sid).to_skip_record(exc=e))
                if not valid_only:
                    # invalid session is still yielded if asked (but integrity threw)
                    yield s

    # ---- Filtering helpers ----
    def filter(
        self,
        sessions: Iterable[Session],
        *,
        animal: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
        health: Optional[str] = None,
    ) -> List[Session]:
        out = []
        for s in sessions:
            if animal is not None and s.animal != animal and ("cj" + s.animal) != animal:
                continue
            if date_range is not None:
                lo, hi = date_range
                if s.date < lo or s.date > hi:
                    continue
            if health is not None:
                if s.health_status() != health:
                    continue
            out.append(s)
        return out


@dataclass
class SkipLogger:
    path: Path

    def write(self, rec: SkipRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # 例: PROJECT_ROOT をこのファイルの親の親…などに合わせて調整
    PROJECT_ROOT = Path(".").resolve()

    spec_path = Path("dataset_spec.yaml")
    ds = MarmoDataset.from_yaml(spec_path, project_root=PROJECT_ROOT)

    # 全セッション（完全性OKのみ）
    sessions = list(ds.iter_sessions(valid_only=True, log_skips=True))
    print("valid sessions:", len(sessions))

    # 例: cj611 の diseased 期間だけ
    subset = ds.filter(sessions, animal="cj611", health="diseased")
    print("cj611 diseased sessions:", len(subset))

    # 例: 1つ見て kp3d パス確認
    if subset:
        s0 = subset[0]
        print("session:", s0.session_id, "kp3d:", s0.kp3d_path())
        # 24fps frame=240 -> stored 25fps frame?
        print("frame24=240 -> frame25=", s0.frame_24_to_25(240))
