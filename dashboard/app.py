"""Streamlit dashboard for the local SafeSight-FL prototype."""

from __future__ import annotations

import sqlite3
import json
import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from audit.hash_chain import verify_chain
from config import load_settings
from database import count_rows, init_db
from feedback.service import feedback_summary
from analytics.queries import grouped_counts, incident_counts, safety_analytics
from privacy.status import current_status
from training.artifacts import scan_run
from federated.verification import is_verified_ppe_federation_run
from ai_copilot.fallback import daily_summary, incident_report, supervisor_actions, incidents_for_day
from ai_copilot.provider import provider_status, generate_explanation
from feedback.service import record_feedback
from edge.offline_queue import pending as pending_sync_items
from sync.transport import sync_pending
from models.registry import (
    activate_version,
    get_active_version,
    list_versions,
    register_version,
    resolve_detection_model,
    rollback_version,
)
from models.imported import imported_model_details


st.set_page_config(page_title="SafeSight-FL | Safety Workspace", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
settings = load_settings()
init_db(settings.db_path)

st.markdown(
    """
    <style>
      :root { --ss-navy: #0b1730; --ss-blue: #2563eb; --ss-ink: #17243b; --ss-muted: #64748b; }
      .stApp { background: #f4f7fb; color: var(--ss-ink); }
      [data-testid="stHeader"] { background: rgba(244,247,251,.88); }
      [data-testid="stSidebar"] { background: linear-gradient(180deg,#0b1730 0%,#122448 100%); border-right: 1px solid #20345a; }
      [data-testid="stSidebar"] * { color: #e6edfb; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #aebedb; }
      [data-testid="stSidebar"] hr { border-color: #2b3c5c; margin: .6rem 0; }
      .ss-brand { display:flex; align-items:center; gap:12px; padding: 8px 2px 20px; }
      .ss-brand-mark { width:40px; height:40px; display:grid; place-items:center; border-radius:13px; background:#17386e; color:#72a7ff; font-size:22px; }
      .ss-brand-name { color:#f7faff; font-weight:750; font-size:18px; letter-spacing:-.3px; }
      .ss-brand-sub { color:#9eafd0; font-size:11px; margin-top:2px; }
      .ss-nav-heading { color:#8094b8; font-weight:700; font-size:10px; letter-spacing:1.35px; margin:12px 0 2px; }
      [data-testid="stSidebar"] .stButton { margin:0; }
      [data-testid="stSidebar"] .stButton button { justify-content:flex-start; text-align:left; border:0; border-radius:9px; padding:7px 10px; transition:background .15s ease; }
      [data-testid="stSidebar"] .stButton button > div { width:100%; justify-content:flex-start; }
      [data-testid="stSidebar"] .stButton button [data-testid="stMarkdownContainer"] p { text-align:left; margin:0; }
      [data-testid="stSidebar"] .stButton button[kind="secondary"] { background:transparent; color:#d7e2f5; }
      [data-testid="stSidebar"] .stButton button[kind="secondary"]:hover { background:#1b3157; color:#fff; }
      [data-testid="stSidebar"] .stButton button[kind="primary"] { background:#1d4ed8; border:0; color:#fff; }
      [data-testid="stMainBlockContainer"] { max-width:1500px; padding-top:2.1rem; padding-bottom:3rem; }
      h1,h2,h3 { color:#14223b; letter-spacing:-.45px; }
      .ss-page-head { display:flex; align-items:center; justify-content:space-between; gap:16px; margin:0 0 2px; }
      .ss-page-title { color:#12213a; font-size:29px; line-height:1.2; font-weight:760; letter-spacing:-.8px; }
      .ss-page-description { color:#66758d; font-size:14px; margin:5px 0 20px; }
      .ss-status { white-space:nowrap; border-radius:999px; padding:6px 11px; font-size:10px; line-height:1; font-weight:750; letter-spacing:.65px; background:#e8eef8; color:#40516d; border:1px solid #d9e2f0; }
      .ss-status-real { background:#e8f5ee; color:#19734a; border-color:#ccebd9; }
      .ss-status-sim { background:#fff4de; color:#a65c00; border-color:#f4dfb6; }
      .ss-status-exp { background:#f0ebff; color:#6845be; border-color:#dfd5ff; }
      [data-testid="stMetric"] { background:#fff; border:1px solid #e3eaf3; border-radius:15px; padding:15px 17px; box-shadow:0 3px 12px rgba(20,39,73,.035); }
      [data-testid="stMetricLabel"] { color:#63718a; font-weight:600; }
      [data-testid="stMetricValue"] { color:#162743; font-weight:750; }
      [data-testid="stAlert"] { border-radius:12px; }
      [data-testid="stDataFrame"] { border:1px solid #e3eaf3; border-radius:12px; overflow:hidden; }
      .stButton > button, [data-testid="stFormSubmitButton"] button { border-radius:9px; font-weight:650; }
      .stButton > button[kind="primary"] { background:#2563eb; border-color:#2563eb; }
      [data-testid="stJson"] { border:1px solid #e3eaf3; border-radius:12px; background:#fff; }
      @media (max-width: 760px) { [data-testid="stMainBlockContainer"] { padding:1.2rem 1rem 2rem; } .ss-page-title { font-size:24px; } .ss-status { font-size:9px; } }
    </style>
    """,
    unsafe_allow_html=True,
)


PAGE_DESCRIPTIONS = {
    "Overview": "Local system health and recorded activity across the simulated sites.",
    "Live Detection": "Run the configured PPE model locally on a webcam snapshot, image, or sampled video clip.",
    "Incidents": "Review only events produced by the detector and configured rules.",
    "Human Feedback": "Supervisor decisions are stored locally and linked to the audit trail.",
    "Dataset Explorer": "Inspect the configured dataset and its reproducible simulated-site partitions.",
    "Model Training": "View recorded training configuration, metrics, and artifacts.",
    "Model Evaluation": "Inspect held-out evaluation results; missing experiments are not filled with estimates.",
    "Federated Learning": "Federation integration evidence from simulated clients on this machine.",
    "Privacy Center": "Current privacy controls and recorded federation state, not aspirational settings.",
    "Model Versions": "Registered model artifacts and their recorded provenance.",
    "Audit Integrity": "Verify the local tamper-evident audit chain.",
    "AI Copilot": "Explain structured incidents; the model does not decide whether an incident occurred.",
    "Analytics": "Counts and groupings calculated from the local incident database.",
    "Settings": "Local paths and demo connectivity controls for this workspace.",
}


def title(text: str, status: str = "REAL") -> None:
    normalized = status.upper()
    css = "ss-status-real" if normalized.startswith("REAL") else "ss-status-sim" if "SIMULATED" in normalized else "ss-status-exp" if "EXPERIMENTAL" in normalized else ""
    st.markdown(
        f'<div class="ss-page-head"><div class="ss-page-title">{text}</div><span class="ss-status {css}">{status}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="ss-page-description">{PAGE_DESCRIPTIONS.get(text, "Local project workspace.")}</div>', unsafe_allow_html=True)


def load_artifact(name: str) -> dict[str, object] | None:
    path = settings.project_root / "artifacts" / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def workspace_path_label(path: Path) -> str:
    """Show a concise workspace-relative path without exposing machine locations."""
    try:
        return Path(path).resolve().relative_to(settings.project_root).as_posix()
    except ValueError:
        return "External local path (hidden)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotated_preview(root: Path, record: dict[str, object], classes: dict[str, str]):
    """Render a real YOLO-labelled sample from the optional local dataset root."""
    from PIL import Image, ImageDraw

    image = Image.open(root / str(record["image"])).convert("RGB")
    draw = ImageDraw.Draw(image)
    label_path = root / str(record["label"])
    width, height = image.size
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x, y, w, h = (float(value) for value in parts[1:])
            left, top = (x - w / 2) * width, (y - h / 2) * height
            right, bottom = (x + w / 2) * width, (y + h / 2) * height
            draw.rectangle((left, top, right, bottom), outline="#22c55e", width=2)
            draw.text((left, max(0, top - 14)), classes.get(str(class_id), str(class_id)), fill="#22c55e")
    return image


st.sidebar.markdown(
    '<div class="ss-brand"><div class="ss-brand-mark">🛡</div><div><div class="ss-brand-name">SafeSight-FL</div><div class="ss-brand-sub">WORKPLACE SAFETY · LOCAL DEMO</div></div></div>',
    unsafe_allow_html=True,
)
NAV_GROUPS = {
    "operations": ("OPERATIONS", ["Overview", "Live Detection", "Incidents", "Human Feedback"]),
    "model_lab": ("MODEL LAB", ["Dataset Explorer", "Model Training", "Model Evaluation"]),
    "federation": ("FEDERATION & PRIVACY", ["Federated Learning", "Privacy Center", "Model Versions"]),
    "system": ("SYSTEM", ["Audit Integrity", "AI Copilot", "Analytics", "Settings"]),
}


def select_navigation(selected_page: str) -> None:
    st.session_state.active_page = selected_page


if "active_page" not in st.session_state:
    st.session_state.active_page = "Overview"
for key, (heading, options) in NAV_GROUPS.items():
    st.sidebar.markdown(f'<div class="ss-nav-heading">{heading}</div>', unsafe_allow_html=True)
    for index, option in enumerate(options):
        st.sidebar.button(
            option,
            key=f"nav_{key}_{index}",
            type="primary" if st.session_state.active_page == option else "secondary",
            on_click=select_navigation,
            args=(option,),
            width="stretch",
        )
page = st.session_state.active_page
if "demo_federation_online" not in st.session_state:
    st.session_state.demo_federation_online = settings.federation_online
federation_online = st.sidebar.toggle(
    "Federation connection (demo)",
    key="demo_federation_online",
    help="Offline queues incident metadata locally; it does not upload CCTV or simulate a successful sync.",
)

if page == "Overview":
    title("Overview")
    cols = st.columns(4)
    for col, label, table in zip(cols, ["Sites", "Incidents", "Pending sync", "Audit entries"], ["sites", "incidents", "sync_queue", "audit_logs"]):
        col.metric(label, count_rows(settings.db_path, table))
    st.info("Factory A/B/C are simulated federated sites on this laptop.")
    with st.expander("Start the viva demo", expanded=True):
        st.write("1. Dataset Explorer → Training → Evaluation: inspect real local data and saved experiments.")
        st.write("2. Live Detection: use a permitted webcam still, uploaded video or image. Only save actual returned events.")
        st.write("3. Human Feedback → AI Copilot → Analytics: review, explain locally and inspect recorded alerts.")
        st.write("4. Federated Learning → Privacy Center → Model Versions: inspect separate recorded experiments and their scope.")
        st.write("5. Audit Integrity → Settings: verify the chain; demonstrate offline queuing and explicit sync only with a configured receiver.")
        st.warning("Weak experimental PPE model. No prerecorded legal clip is bundled; continuous video/RTSP is not implemented. Missing events stay empty—no fake demo incidents.")
elif page == "Live Detection":
    summary = load_artifact("ppe_demo_summary.json")
    fallback_model = settings.project_root / str(summary.get("best_artifact")) if summary else None
    model_path, active_model = resolve_detection_model(
        settings.db_path,
        fallback_model,
        project_root=settings.project_root,
    )
    status = "REAL model output — experimental safety workflow" if model_path and model_path.exists() else "DISABLED — no local PPE artifact"
    title("Live Detection", status)
    st.write("All inference runs locally. Video clips are sampled at one frame per second, for at most the first 30 seconds.")
    if active_model:
        st.caption(f"Active PPE model version: {active_model['version_id']} · selected locally in Model Versions")
        imported = imported_model_details(settings.project_root, active_model)
        if imported:
            st.write(f"Model: {imported['architecture']} · user-trained checkpoint")
            st.write("Supported classes: " + ", ".join(imported["classes"].values()))
            if not imported.get("missing_ppe_classes"):
                st.warning("This model detects PPE objects and people. It has no no-helmet/no-vest classes; absence of a detected item does not establish a violation. Tracked-person zone rules remain available for video.")
    if model_path and model_path.exists():
        model_identity = (str(model_path), model_path.stat().st_mtime_ns)
        if st.session_state.get("inference_model_identity") != model_identity:
            st.session_state.inference_model_identity = model_identity
            st.session_state.video_analysis = None
            st.session_state.video_input_digest = None
            st.session_state.video_recorded_frames = set()
        import io
        from PIL import Image, ImageDraw

        with sqlite3.connect(settings.db_path) as conn:
            sites = {row[0]: row[1] for row in conn.execute("SELECT site_id, name FROM sites ORDER BY site_id")}
        site_id = st.selectbox("Demo site (simulated)", list(sites), format_func=lambda item: sites[item])
        camera_id = f"{site_id}-camera-1"
        zone_enabled = st.checkbox("Experimental tracked-person restricted zone (sampled video only)")
        zone_polygon = None
        zone_id = "restricted-demo-zone"
        zone_config_valid = True
        if zone_enabled:
            from detector.geometry import parse_normalized_polygon

            zone_id = st.text_input("Zone ID", value=zone_id).strip()
            if not zone_id:
                zone_config_valid = False
                st.error("Zone ID cannot be empty")
            polygon_text = st.text_input(
                "Zone polygon (normalized x,y pairs separated by semicolons)",
                value="0.40,0.40;0.80,0.40;0.80,0.90;0.40,0.90",
                help="Coordinates are relative to the sampled image, from 0 to 1. Edit these for the demo camera view.",
            )
            try:
                zone_polygon = parse_normalized_polygon(polygon_text)
            except ValueError as exc:
                zone_config_valid = False
                st.error(str(exc))
            st.warning("Experimental low-rate tracking: only observed outside-to-inside person transitions are eligible. This does not establish production-grade tracking or safety performance.")
        input_mode = st.radio("Input source", ["Webcam snapshot", "Upload image", "Upload video"], horizontal=True)
        captured = st.camera_input("Take a local webcam snapshot") if input_mode == "Webcam snapshot" else None
        uploaded = st.file_uploader("Choose a local image", type=["jpg", "jpeg", "png", "webp"], key="ppe_image_upload") if input_mode == "Upload image" else None
        video_upload = st.file_uploader("Choose a local video (MP4, MOV, AVI, MKV; maximum 100 MB)", type=["mp4", "mov", "avi", "mkv"], key="ppe_video_upload") if input_mode == "Upload video" else None
        media = captured or uploaded
        if input_mode == "Upload video" and video_upload is not None:
            video_bytes = video_upload.getvalue()
            video_digest = hashlib.sha256(video_bytes).hexdigest()
            if len(video_bytes) > 100 * 1024 * 1024:
                st.error("This local demo accepts video files up to 100 MB.")
            else:
                if st.session_state.get("video_input_digest") != video_digest:
                    st.session_state.video_input_digest = video_digest
                    st.session_state.video_analysis = None
                    st.session_state.video_recorded_frames = set()
                if st.button("Analyze video locally", type="primary", disabled=not zone_config_valid):
                    temporary_path = None
                    try:
                        from detector.ultralytics_adapter import load_ppe_model
                        from edge.video_inference import analyze_local_video

                        with tempfile.NamedTemporaryFile(suffix=Path(video_upload.name).suffix, delete=False) as handle:
                            handle.write(video_bytes)
                            temporary_path = Path(handle.name)
                        with st.spinner("Sampling up to 30 seconds of video locally…"):
                            analysis = analyze_local_video(
                                temporary_path,
                                load_ppe_model(model_path),
                                zone_polygon=zone_polygon if zone_enabled else None,
                                zone_id=zone_id,
                            )
                        st.session_state.video_analysis = analysis
                    except Exception as exc:
                        st.session_state.video_analysis = None
                        st.error(f"Local video analysis unavailable: {exc}")
                    finally:
                        if temporary_path is not None:
                            temporary_path.unlink(missing_ok=True)

                analysis = st.session_state.get("video_analysis")
                if analysis is not None and st.session_state.get("video_input_digest") == video_digest:
                    st.caption(
                        f"Analyzed {len(analysis.frames)} sampled frames at {analysis.fps:.2f} FPS. "
                        f"Container duration: {analysis.duration_seconds:.1f}s. The decoder's temporary copy was deleted; the upload remains in this local Streamlit session until cleared or closed."
                    )
                    if analysis.truncated:
                        st.warning("Only the first 30 seconds were sampled in this demo; later frames were not analyzed.")
                    qualifying = []
                    qualifying_zone = []
                    for index, frame in enumerate(analysis.frames):
                        violations = [d for d in frame.detections if d.label.strip().lower() in {"no_helmet", "no_vest", "no_gloves", "no_boots", "no_goggle"}]
                        with st.expander(f"Sample at {frame.elapsed_seconds:.1f}s · {len(frame.detections)} detection(s)", expanded=bool(violations)):
                            st.image(frame.preview_jpeg, caption="Actual model output for this sampled frame", width="stretch")
                            st.json([d.__dict__ for d in frame.detections])
                            if not frame.detections:
                                st.caption("No detections returned for this frame; no boxes or events are fabricated.")
                        if violations and ("ppe", index) not in st.session_state.video_recorded_frames:
                            labels = ", ".join(sorted({d.label for d in violations}))
                            if st.checkbox(
                                f"Record actual {labels} result at {frame.elapsed_seconds:.1f}s",
                                key=f"video_record_{video_digest}_{index}",
                            ):
                                qualifying.append((index, frame, violations))
                        for entry_index, entry in enumerate(frame.zone_entries):
                            marker = ("zone", index, entry_index)
                            if st.checkbox(
                                f"Record tracked entry: {entry.zone_id} · person {entry.detection.track_id} at {frame.elapsed_seconds:.1f}s",
                                key=f"video_zone_{video_digest}_{index}_{entry_index}",
                                disabled=marker in st.session_state.video_recorded_frames,
                            ):
                                qualifying_zone.append((index, entry_index, frame, entry))
                    if zone_enabled:
                        st.info("Zone entry checks require actual person-class tracker IDs. The clip is sampled at only one frame per second, so missed transitions are possible; initial observations already inside the polygon are not treated as entry.")
                    if (qualifying or qualifying_zone) and st.button("Save selected video incident(s) locally"):
                        from edge.video_inference import record_selected_video_frame

                        created = 0
                        recorded = set(st.session_state.video_recorded_frames)
                        for index, frame, _violations in qualifying:
                            incidents = record_selected_video_frame(
                                db_path=settings.db_path,
                                frame=frame,
                                site_id=site_id,
                                camera_id=camera_id,
                                evidence_dir=settings.evidence_dir,
                                federation_online=federation_online,
                            )
                            created += len(incidents)
                            recorded.add(("ppe", index))
                        if qualifying_zone:
                            from edge.video_inference import record_selected_video_zone_entry

                            for index, entry_index, frame, entry in qualifying_zone:
                                incidents = record_selected_video_zone_entry(
                                    db_path=settings.db_path,
                                    frame=frame,
                                    entry=entry,
                                    site_id=site_id,
                                    camera_id=camera_id,
                                    evidence_dir=settings.evidence_dir,
                                    federation_online=federation_online,
                                )
                                created += len(incidents)
                                recorded.add(("zone", index, entry_index))
                        st.session_state.video_recorded_frames = recorded
                        if created:
                            st.success(f"Saved {created} incident record(s) from the selected sampled frame(s). Review them on Incidents.")
                            if not federation_online:
                                st.info("Only incident metadata was queued locally; source video and evidence were not queued.")
                        else:
                            st.info("No supported PPE violation was available to save.")
                    st.info("PPE results are frame-level and have no cross-frame deduplication. Zone entries are experimental and require sampled person-tracker IDs. Only explicitly selected, rule-qualified results are saved.")
        elif media is not None and st.button("Run local detection"):
            suffix = Path(getattr(media, "name", "capture.jpg")).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(media.getvalue())
                temporary_path = Path(handle.name)
            try:
                from detector.ultralytics_adapter import load_ppe_model, predict
                from edge.incident_pipeline import record_detection_events

                detections = predict(load_ppe_model(model_path), temporary_path, confidence=0.25)
                preview = Image.open(io.BytesIO(media.getvalue())).convert("RGB")
                draw = ImageDraw.Draw(preview)
                for detection in detections:
                    x1, y1, x2, y2 = detection.bbox_xyxy
                    draw.rectangle((x1, y1, x2, y2), outline="#2563eb", width=3)
                    draw.text((x1, max(0, y1 - 16)), f"{detection.label} {detection.confidence:.2f}", fill="#2563eb")
                st.image(preview, caption="Local inference output", width="stretch")
                st.json([d.__dict__ for d in detections])
                if not detections:
                    st.info("No detections were returned at this threshold; no boxes are fabricated.")
                incidents = record_detection_events(
                    db_path=settings.db_path,
                    detections=detections,
                    site_id=site_id,
                    camera_id=camera_id,
                    evidence_source=temporary_path,
                    evidence_dir=settings.evidence_dir,
                    federation_online=federation_online,
                )
                if incidents:
                    st.success(f"Recorded {len(incidents)} model-supported PPE incident(s). Review them on the Incidents page.")
                    if not federation_online:
                        st.info("Federation is offline. Only incident metadata is queued locally; no image or evidence is queued.")
                else:
                    st.caption("No supported PPE violation class was detected; no incident was created.")
            except Exception as exc:
                st.error(f"PPE inference unavailable: {exc}")
            finally:
                temporary_path.unlink(missing_ok=True)
    else:
        st.warning("No local demo artifact is available. A generic person model will not be labeled as helmet/vest detection.")
    st.caption("Raw images/videos are never stored in SQLite or federated. Video upload bytes stay in local session memory until cleared/closed; its decoder temp file is removed after analysis. Selected still evidence is saved locally; only incident metadata is queued while offline.")
elif page == "Incidents":
    title("Incidents")
    with sqlite3.connect(settings.db_path) as conn:
        rows = conn.execute("SELECT event_id, site_id, camera_id, event_type, timestamp, status, feedback FROM incidents ORDER BY created_at DESC").fetchall()
    st.dataframe(rows, width="stretch", column_config={"event_id": "Event ID"})
    if rows:
        with sqlite3.connect(settings.db_path) as conn:
            incidents = conn.execute("SELECT event_id, event_type, site_id, ppe_status, zone_id FROM incidents ORDER BY created_at DESC").fetchall()
        labels = {row[0]: f"{row[0][:8]} · {row[1]} · {row[2]}" for row in incidents}
        event_id = st.selectbox("Review an incident", list(labels), format_func=lambda key: labels[key])
        decision = st.selectbox("Human review", ["confirmed", "false_positive", "uncertain"])
        if st.button("Save review decision"):
            record_feedback(settings.db_path, event_id, decision)
            st.success("Review saved locally and added to the audit chain. Refresh to see the updated status.")
elif page == "Human Feedback":
    title("Human Feedback")
    st.json(feedback_summary(settings.db_path))
    st.caption("Available decisions: Confirmed, False Positive, Uncertain. Samples remain local.")
elif page in {"Dataset Explorer", "Model Training", "Model Evaluation"}:
    title(page, "REAL ARTIFACTS — demo experiment")
    if page == "Dataset Explorer":
        manifest = load_artifact("dataset_manifest.json")
        if manifest:
            st.caption("Dataset source and labels are real; Factory A/B/C are deterministic simulated federated sites.")
            split_rows = [{"split": split, "images": data["image_count"], "instances": data["instances"]} for split, data in manifest["splits"].items()]
            st.dataframe(split_rows, width="stretch")
            st.write("Classes")
            st.json(manifest["classes"])
            st.write("Simulated site partitions")
            st.dataframe([{"site": site, **values} for site, values in manifest["simulated_federated_sites"].items()], width="stretch")
            if settings.dataset_root and settings.dataset_root.exists():
                samples = manifest["splits"]["train"]["images"][:6]
                st.write("Real annotated samples from the configured local dataset")
                columns = st.columns(3)
                for index, record in enumerate(samples):
                    with columns[index % 3]:
                        st.image(annotated_preview(settings.dataset_root, record, manifest["classes"]), caption=str(record["image"]), width="stretch")
            else:
                st.caption("Set SAFESIGHT_DATASET_ROOT in .env to preview real local annotations; raw images are not bundled.")
        else:
            st.info("Experiment not run yet. Run scripts/inspect_dataset.py with a local dataset root.")
    elif page == "Model Training":
        imported = imported_model_details(settings.project_root, get_active_version(settings.db_path, project_root=settings.project_root))
        if imported:
            st.subheader("Active imported training artifacts")
            st.write(f"{imported['architecture']} · {imported['model_version_id']}")
            st.json(imported['training_config_from_checkpoint'])
            st.json(imported['artifact_roles'])
            st.caption("Training config is read from the supplied checkpoint. Historical training runs below belong to earlier models.")
        gpu_status = load_artifact("gpu_training_status.json")
        if gpu_status:
            st.subheader("Full-split GPU training")
            st.json(gpu_status)
            st.button("Refresh training status")
            st.caption("Saved progress updates after each epoch. Existing detector stays active; completion does not automatically activate the new model.")
        summary = load_artifact("ppe_demo_summary.json")
        if summary:
            st.json({key: summary[key] for key in ("dataset", "mode", "base_model", "epochs", "fraction", "image_size", "device", "train_run", "best_artifact")})
            st.write("Actual train metrics")
            st.json(summary.get("train_metrics", {}))
            st.write("Artifacts discovered")
            st.json(scan_run(settings.project_root / summary["train_run"]))
        else:
            st.info("Experiment not run yet.")
    else:
        imported = imported_model_details(settings.project_root, get_active_version(settings.db_path, project_root=settings.project_root))
        if imported:
            st.subheader("Active imported model — checkpoint metrics")
            st.write(f"{imported['architecture']} · {imported['model_version_id']}")
            st.warning(imported['metrics_scope'])
            st.json(imported.get('checkpoint_metrics') or {})
            st.info("Local held-out evaluation for the imported model: Experiment not run yet. Its 8-class label mapping differs from the earlier 11-class dataset; those test metrics cannot be reused.")
            st.caption("Earlier model evaluation records follow for reference only.")
        gpu_status = load_artifact("gpu_training_status.json")
        if gpu_status and gpu_status.get("status") == "completed":
            st.subheader("Full-split GPU model — held-out test results")
            st.json(gpu_status.get("test_metrics", {}))
            st.dataframe([{"class": label, **metrics} for label, metrics in gpu_status.get("per_class_test_metrics", {}).items()], width="stretch")
        summary = load_artifact("ppe_demo_summary.json")
        if summary:
            st.json({"held_out_split": "test", "test_metrics": summary.get("test_metrics"), "test_run": summary.get("test_run"), "classes": summary.get("classes")})
            if summary.get("per_class_test_metrics"):
                st.write("Actual per-class metrics")
                st.dataframe([{"class": label, **metrics} for label, metrics in summary["per_class_test_metrics"].items()], width="stretch")
            st.caption("Metrics are held-out test results from the one-epoch demo run, not a production benchmark.")
        else:
            st.info("Experiment not run yet.")
elif page == "Federated Learning":
    ppe_fl_run = load_artifact("federated/ppe_head_fl_latest.json")
    ppe_fl_verified = is_verified_ppe_federation_run(
        settings.project_root / "artifacts" / "federated", ppe_fl_run
    )
    title(
        "Federated Learning",
        "EXPERIMENTAL — PPE head-only round; simulated sites"
        if ppe_fl_verified
        else "EXPERIMENTAL — simulated sites; no verified PPE round",
    )
    with sqlite3.connect(settings.db_path) as conn:
        rounds = conn.execute("SELECT round_number, metrics_json, status, created_at FROM federation_rounds ORDER BY round_number DESC").fetchall()
        versions = conn.execute("SELECT version_id, round_number, validation_metric, artifact_path FROM model_versions ORDER BY created_at DESC").fetchall()
    st.info("The PPE head-only FedAvg and SecAgg+ runs, plus separate toy Flower integrations, are experimental software tests—not detector efficacy evidence. Factory A/B/C are simulated logical clients on one machine.")
    st.dataframe(rounds, width="stretch")
    st.write("Latest model versions")
    st.dataframe(versions, width="stretch")
    if ppe_fl_verified:
        st.subheader("PPE head-only federation run")
        st.caption(
            f"{ppe_fl_run['run_id']} · {ppe_fl_run['protocol']} · CPU · "
            "one local epoch per client · global checkpoint SHA-256 verified"
        )
        initial_metrics = ppe_fl_run["initial_test_metrics"]
        final_metrics = ppe_fl_run["post_federation_test_metrics"]
        metric_columns = st.columns(4)
        metric_columns[0].metric("Test mAP@0.5 before", f"{initial_metrics['mAP50']:.4f}")
        metric_columns[1].metric("Test mAP@0.5 after", f"{final_metrics['mAP50']:.4f}")
        metric_columns[2].metric("Test precision after", f"{final_metrics['precision']:.4f}")
        metric_columns[3].metric("Global head Δ (L2)", f"{float(ppe_fl_run['global_head_delta_l2']):.4f}")
        st.write("Per-client local training evidence")
        fit_metrics = ppe_fl_run.get("client_fit_metrics") or {}
        site_rows = []
        for site in ppe_fl_run["simulated_sites"]:
            row = {
                "simulated site": site,
                "train images": ppe_fl_run["site_train_images"][site],
                "epochs": ppe_fl_run["local_epochs"],
            }
            if ppe_fl_run.get("secure_aggregation") is True:
                row["head update"] = "concealed by Flower SecAgg+"
            else:
                row["head update Δ (L2)"] = fit_metrics[f"{site}_head_delta_l2"]
            site_rows.append(row)
        st.dataframe(site_rows, width="stretch")
        st.json({"held_out_test_metrics_before": initial_metrics, "held_out_test_metrics_after": final_metrics})
        secagg_state = "Flower SecAgg+ completed for this PPE round; Differential Privacy was not enabled." if ppe_fl_run.get("secure_aggregation") is True else "Secure Aggregation and Differential Privacy were not enabled for this PPE round."
        st.warning(
            "Experimental integration evidence only. This single, tiny round does not establish model quality; "
            f"precision remains weak. Factory A/B/C are simulated on one laptop. {secagg_state}"
        )
        if ppe_fl_run.get("server_visibility"):
            st.info(ppe_fl_run["server_visibility"])
        flower_completion = ppe_fl_run.get("flower_completion")
        if isinstance(flower_completion, dict) and flower_completion.get("launcher_exit_code") not in (None, 0):
            st.info(
                "The WSL Flower launcher returned a nonzero exit code after its server task finished. "
                "This run is shown only because the post-SecAgg aggregate file was produced and the "
                "checkpoint passed schema, change, hash, and held-out evaluation checks."
            )
        st.caption("Raw training images were not sent by this local Flower run. Metrics are real test-split measurements, not mock data.")
    elif ppe_fl_run and str(ppe_fl_run.get("status", "")).startswith("invalid"):
        st.warning("The latest PPE federation attempt was rejected because it did not produce verified changed client weights.")
    else:
        st.info("No verified PPE federation run is recorded. The existing lightweight Flower and SecAgg+ integrations use separate toy workloads.")

    adapter_check = load_artifact("federated/ppe_head_adapter_check.json")
    ppe_summary = load_artifact("ppe_demo_summary.json")
    ppe_checkpoint = settings.project_root / str(ppe_summary.get("best_artifact")) if ppe_summary else None
    adapter_matches_checkpoint = bool(
        adapter_check
        and ppe_checkpoint
        and ppe_checkpoint.exists()
        and adapter_check.get("checkpoint_sha256") == sha256_file(ppe_checkpoint)
    )
    st.subheader("PPE detector head compatibility")
    if adapter_matches_checkpoint:
        left, right, third = st.columns(3)
        left.metric("Checkpoint compatibility", "Verified")
        right.metric("Head parameters", f"{int(adapter_check['trainable_head_parameter_count']):,}")
        third.metric("Raw update / direction", f"{adapter_check['bytes_per_client_update'] / (1024 * 1024):.2f} MiB")
        st.info("The adapter compatibility check matches the local PPE checkpoint and reports the raw shareable parameter payload.")
        st.caption("This estimates one raw parameter payload only; it excludes protocol overhead and does not imply Secure Aggregation or Differential Privacy.")
    else:
        st.info("Experiment not run yet. The compatibility checker is local-only and does not train, federate, or send data.")
elif page == "Privacy Center":
    title("Privacy Center")
    privacy_state = current_status(settings.db_path, federation_online)
    st.caption("SIMULATED FEDERATED SITES — historical experiment evidence, not live security monitoring.")
    controls = st.columns(3)
    controls[0].metric("Raw CCTV sent to federation", "NO")
    controls[1].metric("Training images sent to federation", "NO")
    controls[2].metric("Recorded training rounds", privacy_state["completed_rounds"])
    st.info("Detection and evidence stay local. Optional Gemini Copilot sends structured incident text to its provider; this is separate from federation. The ONLINE/OFFLINE switch is a demo control, not a network health check.")
    st.subheader("Recorded federation experiments")
    if privacy_state["recorded_runs"]:
        st.dataframe(privacy_state["recorded_runs"], width="stretch")
        st.caption("Each row is a separate experiment. SecAgg from one run and DP from another do not constitute combined protection. Blank values mean not recorded. Bytes describe raw model arrays per direction, not measured network traffic.")
    else:
        st.info("Experiment not run yet. No completed federation evidence is available.")
    st.warning("Factory A/B/C share one laptop. Local administrators can inspect local data and artifacts; this is not an independent-site security boundary.")
    dp = privacy_state.get("federated_dp")
    if dp:
        st.warning("EXPERIMENTAL: DP-SGD on synthetic Flower clients only. Not PPE DP or combined SecAgg + DP. Secure RNG is disabled; separate runs are not composed.")
        st.caption(f"Recorded run: {dp['run_id']} — {dp['accountant']} accountant")
        st.dataframe([{"round": record["round"], **client}
                      for record in dp["accounting"] for client in record["clients"]], width="stretch")
    with st.expander("Detailed local evidence and scope"):
        st.json(privacy_state)
elif page == "Model Versions":
    title("Model Versions", "REAL — registry state")
    versions = list_versions(settings.db_path, project_root=settings.project_root)
    active = get_active_version(settings.db_path, project_root=settings.project_root)
    st.write("Registered versions")
    st.dataframe(
        [
            {
                "version": version["version_id"],
                "kind": version["model_kind"],
                "round": version["round_number"],
                "validation metric": version["validation_metric"],
                "clients": ", ".join(version["clients"]) or "—",
                "artifact": workspace_path_label(version["resolved_artifact"]) if version["resolved_artifact"] else "Unavailable / outside workspace",
                "available": version["artifact_available"],
                "notes": version["notes"],
                "created": version["created_at"],
            }
            for version in versions
        ],
        width="stretch",
    )

    training_summary = load_artifact("ppe_demo_summary.json")
    baseline_path = (
        settings.project_root / str(training_summary.get("best_artifact"))
        if training_summary and training_summary.get("best_artifact")
        else None
    )
    baseline_registered = bool(
        baseline_path
        and any(
            version["model_kind"] == "ppe_detector_full"
            and version["resolved_artifact"] == baseline_path.resolve()
            for version in versions
        )
    )
    if baseline_path and baseline_path.is_file() and not baseline_registered:
        if st.button("Register current local PPE checkpoint"):
            register_version(
                settings.db_path,
                baseline_path,
                notes="Existing one-epoch local Construction-PPE demo checkpoint; performance is weak and not safety-validated.",
                model_kind="ppe_detector_full",
                project_root=settings.project_root,
            )
            st.rerun()

    ppe_versions = [
        version
        for version in versions
        if version["model_kind"] in {"ppe_detector_full", "ppe_detector_head_only"}
        and version["artifact_available"]
    ]
    if ppe_versions:
        options = [version["version_id"] for version in ppe_versions]
        version_by_id = {version["version_id"]: version for version in ppe_versions}
        current_id = active["version_id"] if active else None
        selected_default = options.index(current_id) if current_id in options else 0
        selected_id = st.selectbox(
            "PPE model version",
            options,
            index=selected_default,
            format_func=lambda version_id: f"{version_id} · {version_by_id[version_id]['model_kind']} · round {version_by_id[version_id]['round_number'] or '—'}",
        )
        activate_col, rollback_col = st.columns(2)
        if activate_col.button("Activate selected PPE version", disabled=selected_id == current_id):
            try:
                activated = activate_version(
                    settings.db_path,
                    selected_id,
                    project_root=settings.project_root,
                )
                st.success(f"Activated local PPE model version {activated['version_id']}.")
                st.rerun()
            except (FileNotFoundError, KeyError, ValueError, RuntimeError):
                st.error("Activation failed. The selected artifact is missing, outside this workspace, or not a supported PPE checkpoint.")
        has_previous = bool(active and active.get("previous_version_id"))
        if rollback_col.button("Roll back to previous version", disabled=not has_previous):
            try:
                restored = rollback_version(settings.db_path, project_root=settings.project_root)
                st.success(f"Rolled back to local PPE model version {restored['version_id']}.")
                st.rerun()
            except (FileNotFoundError, KeyError, ValueError, RuntimeError):
                st.error("Rollback failed because the previous PPE checkpoint is no longer available or valid.")
    else:
        st.info("No available PPE checkpoint is registered yet. The local demo checkpoint can be registered above when present.")
    if active:
        st.success(f"Active PPE version: {active['version_id']} · {active['model_kind']}")
        st.caption("Live Detection uses this local checkpoint. Rollback switches to the previously active registered PPE checkpoint.")
    else:
        st.info("No version is active; Live Detection uses the configured demo checkpoint when available.")
elif page == "Audit Integrity":
    title("Audit Integrity", "REAL — local hash-chain verification")
    valid, message = verify_chain(settings.db_path)
    (st.success if valid else st.error)(message)
    st.caption("Tamper-evident hash chain; not a blockchain.")
elif page == "AI Copilot":
    copilot_status = provider_status()
    title("AI Incident Copilot", "EXPERIMENTAL — Gemini optional; deterministic fallback always available")
    st.info("The copilot summarizes structured CV/rules events only. It never determines whether an incident occurred.")
    st.json(copilot_status)
    remote_copilot = st.checkbox("Allow Gemini to receive allowlisted incident metadata", value=False)
    st.caption("Local-only by default. No images, evidence paths, person IDs or free-form notes are sent. AI prose is unverified assistance: review it against the recorded event.")
    summary_day = st.date_input("Summary date (UTC)")
    st.subheader("Daily safety summary — deterministic local counts")
    day_incidents = incidents_for_day(settings.db_path, summary_day.isoformat())
    st.write(daily_summary(day_incidents))
    st.caption(f"{len(day_incidents)} recorded alerts for the selected UTC date; not a count of confirmed violations.")
    with sqlite3.connect(settings.db_path) as conn:
        conn.row_factory = sqlite3.Row
        incidents = [dict(row) for row in conn.execute("SELECT * FROM incidents ORDER BY created_at DESC LIMIT 20")]
    if incidents:
        selected = st.selectbox("Structured incident", incidents, format_func=lambda item: f"{item['event_id']} · {item['event_type']} · {item['site_id']}")
        if st.button("Generate incident explanation"):
            explanation = generate_explanation(selected, allow_remote=remote_copilot)
            st.caption(f"Source: {explanation.provider}; outbound request attempted: {explanation.outbound_attempted}")
            st.text(explanation.text)
        else:
            st.write(incident_report(selected))
        st.write("Suggested supervisor actions")
        st.write(supervisor_actions(selected))
        st.download_button("Download local incident report", incident_report(selected), file_name="incident-report.txt", mime="text/plain")
    else:
        st.info("No structured incidents are recorded yet; run the local detection/demo workflow first.")
elif page == "Analytics":
    title("Analytics", "REAL — SQL-backed counts")
    from datetime import datetime, timezone, timedelta
    import pandas as pd
    today = datetime.now(timezone.utc).date()
    dates = st.columns(2)
    start = dates[0].date_input("Start date (UTC)", value=today - timedelta(days=6))
    end = dates[1].date_input("End date (UTC)", value=today)
    with sqlite3.connect(settings.db_path) as conn:
        sites = [r[0] for r in conn.execute("SELECT DISTINCT site_id FROM incidents ORDER BY site_id")]
    site = st.selectbox("Site", [None, *sites], format_func=lambda v: "All sites" if v is None else f"{v} — SIMULATED SITE")
    with sqlite3.connect(settings.db_path) as conn:
        cameras = [r[0] for r in conn.execute("SELECT DISTINCT camera_id FROM incidents WHERE (? IS NULL OR site_id = ?) ORDER BY camera_id", (site, site))]
    camera = st.selectbox("Camera", [None, *cameras], format_func=lambda v: "All cameras" if v is None else v)
    st.caption("Recorded alerts, not confirmed violations. Review totals use each incident's current decision; repeated reviews are not counted twice. Dates refer to event time, not review time.")
    try:
        report = safety_analytics(settings.db_path, start.isoformat(), end.isoformat(), site_id=site, camera_id=camera)
    except ValueError as exc:
        st.error(str(exc))
    else:
        columns = st.columns(3)
        for index, (key, value) in enumerate(report["counts"].items()):
            columns[index % 3].metric(key.replace("_", " ").title(), value)
        if not report["counts"]["total"]:
            st.info("No recorded alerts match these filters.")
        st.write("Most common recorded alert type(s): " + (", ".join(report["most_common_alert_types"]) or "None"))
        st.line_chart(pd.DataFrame(report["daily"]).set_index("date"))
        for field, rows in report["groups"].items():
            st.subheader("Alerts by " + field.replace("_", " "))
            if rows:
                st.bar_chart(pd.DataFrame(rows).set_index(field))
                st.dataframe(rows, width="stretch")
        if report["invalid_timestamps_in_database"]:
            st.warning(f"{report['invalid_timestamps_in_database']} records across the database have invalid timestamps and cannot be included in date-filtered analytics.")
else:
    title("Settings", "REAL — local configuration")
    cards = st.columns(3)
    cards[0].metric("Environment", settings.environment.title())
    cards[1].metric("Demo mode", "Enabled" if settings.demo_mode else "Disabled")
    cards[2].metric("Federation demo", "Online" if federation_online else "Offline")
    st.markdown("#### Local storage")
    st.dataframe(
        [
            {"Resource": "Metadata database", "Workspace path": workspace_path_label(settings.db_path), "Status": "Local metadata only"},
            {"Resource": "Incident evidence", "Workspace path": workspace_path_label(settings.evidence_dir), "Status": "Local files"},
            {"Resource": "Dataset preview", "Workspace path": workspace_path_label(settings.dataset_root) if settings.dataset_root else "—", "Status": "Configured locally" if settings.dataset_root else "Not configured"},
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption("Camera images and training data remain on this machine. Workspace-relative locations are shown; external machine paths are hidden.")
    st.markdown("#### Offline incident metadata sync")
    pending_items = pending_sync_items(settings.db_path)
    transport_ready = bool(settings.sync_endpoint and settings.sync_token)
    sync_status = "Configured; manual action only" if transport_ready else "Disabled — endpoint/token not configured"
    st.write(f"Transport: **{sync_status}** · Pending queue items: **{len(pending_items)}**")
    st.caption("Only allowlisted incident metadata is sent. HTTPS is required for remote destinations; plain HTTP is accepted only for a loopback demo receiver. Images and evidence are never sent. No automatic delivery occurs.")
    if st.button("Send pending metadata now", disabled=not (transport_ready and federation_online and pending_items)):
        try:
            result = sync_pending(
                settings.db_path,
                endpoint=settings.sync_endpoint,
                token=settings.sync_token,
                online=federation_online,
            )
            st.success(f"Acknowledged {result.synced} of {result.attempted} attempted items. Pending: {result.pending_after}; unsupported: {result.unsupported}; failed: {result.failed}.")
        except ValueError as exc:
            st.error(str(exc))
