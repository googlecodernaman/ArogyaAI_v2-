"use client";
import type { ReactNode } from "react";
import { useState, useRef, useCallback } from "react";
import {
    Send, Shield, Activity, Brain, CheckCircle,
    Loader2, ThumbsUp, ThumbsDown, AlertTriangle,
    ChevronDown, ChevronUp, Bot, Cpu, Zap, GitMerge,
    Phone, Eye, Sparkles, Upload, FileText, Trash2,
    Database, Search, Stethoscope, X
} from "lucide-react";
import { buildSanitizedPrompt } from "@/lib/sanitizer";
import { submitFederatedUpdate } from "@/lib/local_learning";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8080";

/* ── Types ─────────────────────────────────────────────────────────────────── */
interface StageStatus {
    divergence: "idle" | "running" | "complete";
    convergence: "idle" | "running" | "complete";
    synthesis: "idle" | "running" | "complete";
}
interface MemberData {
    differentials?: string[];
    next_steps?: string[];
    confidence?: number;
    red_flag?: boolean;
    raw?: string;
}
interface CouncilResult {
    divergence: Record<string, unknown>;
    convergence: unknown;
    synthesis: {
        final_differentials?: string[];
        recommended_next_steps?: string[];
        confidence?: number;
        red_flag?: boolean;
        summary?: string;
    };
}
interface Emergency { active: boolean; reason: string; message: string; }

interface Classification {
    category: string;
    label: string;
    severity: string;
    confidence: number;
    description: string;
    action: string;
    probabilities: Record<string, number>;
}

interface RAGInfo {
    documents_found: number;
    topics: string[];
}

interface UploadedReport {
    id: string;
    filename: string;
    file_type: string;
    uploaded_at: string;
    word_count: number;
}

/* ── Member meta ───────────────────────────────────────────────────────────── */
const MEMBER_META: Record<string, { label: string; model: string; provider: string; icon: ReactNode; color: string; bg: string; border: string }> = {
    member_a: { label: "Member A", model: "Llama 3.3 70B", provider: "Groq", icon: <Brain style={{ width: 14, height: 14 }} />, color: "#93c5fd", bg: "rgba(59,130,246,0.08)", border: "rgba(59,130,246,0.2)" },
    member_b: { label: "Member B", model: "Gemini 2.0 Flash", provider: "Google", icon: <Zap style={{ width: 14, height: 14 }} />, color: "#6ee7b7", bg: "rgba(16,185,129,0.08)", border: "rgba(16,185,129,0.2)" },
    member_c: { label: "Member C", model: "Mistral Small", provider: "Mistral AI", icon: <Cpu style={{ width: 14, height: 14 }} />, color: "#c4b5fd", bg: "rgba(139,92,246,0.08)", border: "rgba(139,92,246,0.2)" },
};

const STAGES = [
    { key: "divergence" as const, label: "Divergence", icon: <Brain style={{ width: 13, height: 13 }} />, desc: "3 models reasoning" },
    { key: "convergence" as const, label: "Convergence", icon: <Eye style={{ width: 13, height: 13 }} />, desc: "Peer review" },
    { key: "synthesis" as const, label: "Synthesis", icon: <GitMerge style={{ width: 13, height: 13 }} />, desc: "Chairman decides" },
];

const SEVERITY_COLORS: Record<string, { color: string; bg: string; border: string }> = {
    critical: { color: "#f87171", bg: "rgba(239,68,68,0.08)", border: "rgba(239,68,68,0.25)" },
    moderate: { color: "#fbbf24", bg: "rgba(245,158,11,0.08)", border: "rgba(245,158,11,0.25)" },
    "low-moderate": { color: "#93c5fd", bg: "rgba(59,130,246,0.08)", border: "rgba(59,130,246,0.25)" },
    low: { color: "#6ee7b7", bg: "rgba(16,185,129,0.08)", border: "rgba(16,185,129,0.25)" },
};

/* ── Emergency overlay ─────────────────────────────────────────────────────── */
function EmergencyOverlay({ reason, message, onDismiss }: { reason: string; message: string; onDismiss: () => void }) {
    return (
        <div className="emergency-overlay">
            <div className="emergency-card">
                <div className="emergency-ring" />
                <div className="emergency-icon">
                    <AlertTriangle />
                </div>
                <h2 className="emergency-title">Emergency Detected</h2>
                <p className="emergency-reason"><strong>Reason:</strong> {reason}</p>
                <p className="emergency-msg">{message}</p>
                <a href="tel:112" className="emergency-call">
                    <Phone /> Call Emergency Services (112)
                </a>
                <button className="emergency-dismiss" onClick={onDismiss}>
                    I understand — dismiss alert
                </button>
            </div>
        </div>
    );
}

/* ── Member tab ────────────────────────────────────────────────────────────── */
function MemberTab({ memberKey, data }: { memberKey: string; data: unknown }) {
    const [open, setOpen] = useState(false);
    const meta = MEMBER_META[memberKey] ?? {
        label: memberKey, model: "",
        icon: <Bot style={{ width: 14, height: 14 }} />,
        color: "#94a3b8", bg: "rgba(148,163,184,0.08)", border: "rgba(148,163,184,0.2)",
    };
    const parsed = data as MemberData;

    return (
        <div className="member-tab" style={{ background: meta.bg, border: `1px solid ${meta.border}` }}>
            <button className="member-tab__header" onClick={() => setOpen(!open)}>
                <div className="member-tab__left">
                    <span className="member-tab__icon" style={{ color: meta.color }}>{meta.icon}</span>
                    <span className="member-tab__name">{meta.label}</span>
                    <span className="member-tab__model">{meta.model}</span>
                    <span className="member-tab__model" style={{ opacity: 0.5 }}>· {meta.provider}</span>
                </div>
                <div className="member-tab__right">
                    {parsed?.confidence !== undefined && (
                        <span className="member-tab__conf" style={{ color: meta.color }}>
                            {Math.round(parsed.confidence * 100)}% conf
                        </span>
                    )}
                    <span className="member-tab__chevron">
                        {open ? <ChevronUp style={{ width: 14, height: 14, color: "#475569" }} />
                            : <ChevronDown style={{ width: 14, height: 14, color: "#475569" }} />}
                    </span>
                </div>
            </button>

            {open && (
                <div className="member-tab__body animate-fade-in">
                    {parsed?.differentials && parsed.differentials.length > 0 && (
                        <>
                            <div className="member-tab__body-label">Differentials</div>
                            <ul className="result-list">
                                {parsed.differentials.map((d, i) => (
                                    <li key={i} className="result-list__item">
                                        <span className="result-list__dot" style={{ background: meta.color }} />
                                        {d}
                                    </li>
                                ))}
                            </ul>
                        </>
                    )}
                    {parsed?.next_steps && parsed.next_steps.length > 0 && (
                        <>
                            <div className="member-tab__body-label">Next Steps</div>
                            <ul className="result-list">
                                {parsed.next_steps.map((s, i) => (
                                    <li key={i} className="result-list__item">
                                        <span className="result-list__dot" style={{ background: "#10b981" }} />
                                        {s}
                                    </li>
                                ))}
                            </ul>
                        </>
                    )}
                </div>
            )}
        </div>
    );
}

/* ── Main ──────────────────────────────────────────────────────────────────── */
export default function ChatInterface() {
    const [symptoms, setSymptoms] = useState("");
    const [age, setAge] = useState("");
    const [sex, setSex] = useState("");
    const [vitals, setVitals] = useState({ heart_rate: "", spo2: "", systolic_bp: "" });
    const [loading, setLoading] = useState(false);
    const [stages, setStages] = useState<StageStatus>({ divergence: "idle", convergence: "idle", synthesis: "idle" });
    const [result, setResult] = useState<CouncilResult | null>(null);
    const [emergency, setEmergency] = useState<Emergency>({ active: false, reason: "", message: "" });
    const [feedbackSent, setFeedbackSent] = useState(false);
    const [showMembers, setShowMembers] = useState(false);
    const clientId = useRef(`client_${Math.random().toString(36).slice(2)}`);

    // New state for RAG/ML features
    const [classification, setClassification] = useState<Classification | null>(null);
    const [ragInfo, setRagInfo] = useState<RAGInfo | null>(null);
    const [reports, setReports] = useState<UploadedReport[]>([]);
    const [uploading, setUploading] = useState(false);
    const [showReports, setShowReports] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Fetch existing reports on mount
    const fetchReports = useCallback(async () => {
        try {
            const res = await fetch(`${BACKEND_URL}/api/reports`);
            const data = await res.json();
            setReports(data.reports || []);
        } catch { /* ignore */ }
    }, []);

    // Fetch on first render
    useState(() => { fetchReports(); });

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setUploading(true);
        try {
            const formData = new FormData();
            formData.append("file", file);
            const res = await fetch(`${BACKEND_URL}/api/reports/upload`, {
                method: "POST",
                body: formData,
            });
            if (res.ok) {
                await fetchReports();
            }
        } catch (err) {
            console.error("Upload failed:", err);
        } finally {
            setUploading(false);
            if (fileInputRef.current) fileInputRef.current.value = "";
        }
    };

    const handleDeleteReport = async (reportId: string) => {
        try {
            await fetch(`${BACKEND_URL}/api/reports/${reportId}`, { method: "DELETE" });
            await fetchReports();
        } catch { /* ignore */ }
    };

    const handleSubmit = useCallback(async () => {
        if (!symptoms.trim()) return;
        setLoading(true);
        setResult(null);
        setShowMembers(false);
        setClassification(null);
        setRagInfo(null);
        setStages({ divergence: "idle", convergence: "idle", synthesis: "idle" });
        setFeedbackSent(false);

        const sanitizedPrompt = buildSanitizedPrompt(symptoms, age ? parseInt(age) : undefined, sex || undefined);
        const vitalsPayload: Record<string, number> = {};
        if (vitals.heart_rate) vitalsPayload.heart_rate = parseInt(vitals.heart_rate);
        if (vitals.spo2) vitalsPayload.spo2 = parseInt(vitals.spo2);
        if (vitals.systolic_bp) vitalsPayload.systolic_bp = parseInt(vitals.systolic_bp);

        // Triage
        try {
            const triageRes = await fetch(`${BACKEND_URL}/api/triage`, {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sanitized_prompt: sanitizedPrompt, vitals: vitalsPayload }),
            });
            const triage = await triageRes.json();
            if (triage.is_emergency) {
                setEmergency({ active: true, reason: triage.reason, message: triage.message });
                setLoading(false); return;
            }
        } catch (e) { console.error("Triage failed:", e); }

        // Council SSE (now includes classification + RAG stages)
        try {
            const response = await fetch(`${BACKEND_URL}/api/council`, {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sanitized_prompt: sanitizedPrompt, vitals: vitalsPayload }),
            });
            const reader = response.body!.getReader();
            const decoder = new TextDecoder();
            const councilResult: Partial<CouncilResult> = {};
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const parts = buffer.split("\n\n");
                buffer = parts.pop() ?? "";
                for (const part of parts) {
                    const dataLine = part.split("\n").find(l => l.startsWith("data: "));
                    if (!dataLine) continue;
                    try {
                        const event = JSON.parse(dataLine.slice(6));
                        if (event.stage === "classification" && event.data) {
                            setClassification(event.data);
                        } else if (event.stage === "rag_retrieval" && event.data) {
                            setRagInfo(event.data);
                        } else if (event.stage === "divergence") {
                            setStages(s => ({ ...s, divergence: event.status === "complete" ? "complete" : "running" }));
                            if (event.data) councilResult.divergence = event.data;
                        } else if (event.stage === "convergence") {
                            setStages(s => ({ ...s, convergence: event.status === "complete" ? "complete" : "running" }));
                            if (event.data) councilResult.convergence = event.data;
                        } else if (event.stage === "synthesis") {
                            setStages(s => ({ ...s, synthesis: event.status === "complete" ? "complete" : "running" }));
                            if (event.data) councilResult.synthesis = event.data;
                        }
                    } catch { /* skip */ }
                }
            }
            setResult(councilResult as CouncilResult);
        } catch (e) { console.error("Council error:", e); }
        finally { setLoading(false); }
    }, [symptoms, age, sex, vitals]);

    const handleFeedback = async (positive: boolean) => {
        if (!result?.synthesis?.summary) return;
        await submitFederatedUpdate(clientId.current, result.synthesis.summary, positive ? "Accurate." : "Needs improvement.", BACKEND_URL);
        setFeedbackSent(true);
    };

    const allComplete = stages.divergence === "complete" && stages.convergence === "complete" && stages.synthesis === "complete";

    return (
        <div>
            {emergency.active && (
                <EmergencyOverlay reason={emergency.reason} message={emergency.message}
                    onDismiss={() => setEmergency({ active: false, reason: "", message: "" })} />
            )}

            {/* Privacy badge */}
            <div className="privacy-badge">
                <Shield style={{ width: 16, height: 16, color: "#10b981" }} />
                <span className="privacy-badge__text">Privacy-Protected Input</span>
                <span className="privacy-badge__pill">PII stripped before sending</span>
            </div>

            {/* Medical Reports Upload */}
            <div className="report-section">
                <button className="report-toggle" onClick={() => setShowReports(!showReports)}>
                    <FileText style={{ width: 15, height: 15, color: "#8b5cf6" }} />
                    <span>Medical Reports</span>
                    {reports.length > 0 && (
                        <span className="report-count">{reports.length}</span>
                    )}
                    <span className="members-toggle__chevron">
                        {showReports
                            ? <ChevronUp style={{ width: 15, height: 15 }} />
                            : <ChevronDown style={{ width: 15, height: 15 }} />}
                    </span>
                </button>

                {showReports && (
                    <div className="report-panel animate-fade-in">
                        <div className="report-upload-area">
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept=".pdf,.docx,.doc,.txt"
                                onChange={handleFileUpload}
                                style={{ display: "none" }}
                                id="report-upload"
                            />
                            <button
                                className="report-upload-btn"
                                onClick={() => fileInputRef.current?.click()}
                                disabled={uploading}
                            >
                                {uploading
                                    ? <Loader2 className="animate-spin" style={{ width: 16, height: 16 }} />
                                    : <Upload style={{ width: 16, height: 16 }} />}
                                {uploading ? "Uploading…" : "Upload Report (PDF, DOCX, TXT)"}
                            </button>
                            <span className="report-upload-hint">
                                Reports are stored locally and used for RAG context. Max 10MB.
                            </span>
                        </div>

                        {reports.length > 0 && (
                            <div className="report-list">
                                {reports.map(r => (
                                    <div key={r.id} className="report-item">
                                        <FileText style={{ width: 14, height: 14, color: "#8b5cf6", flexShrink: 0 }} />
                                        <div className="report-item__info">
                                            <span className="report-item__name">{r.filename}</span>
                                            <span className="report-item__meta">
                                                {r.word_count} words · {new Date(r.uploaded_at).toLocaleDateString()}
                                            </span>
                                        </div>
                                        <button
                                            className="report-item__delete"
                                            onClick={() => handleDeleteReport(r.id)}
                                            title="Delete report"
                                        >
                                            <Trash2 style={{ width: 13, height: 13 }} />
                                        </button>
                                    </div>
                                ))}
                            </div>
                        )}

                        {reports.length === 0 && (
                            <div className="report-empty">
                                No reports uploaded yet. Upload your medical reports for personalized insights.
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Demographics */}
            <div className="form-row form-row--2">
                <input type="number" placeholder="Age (optional)" value={age}
                    onChange={e => setAge(e.target.value)} className="input-field" />
                <select value={sex} onChange={e => setSex(e.target.value)} className="input-field">
                    <option value="">Biological sex (optional)</option>
                    <option value="M">Male</option>
                    <option value="F">Female</option>
                    <option value="Other">Other / Prefer not to say</option>
                </select>
            </div>

            {/* Vitals */}
            <div className="form-row form-row--3">
                <input type="number" placeholder="Heart Rate (bpm)" value={vitals.heart_rate}
                    onChange={e => setVitals(v => ({ ...v, heart_rate: e.target.value }))} className="input-field" />
                <input type="number" placeholder="SpO₂ (%)" value={vitals.spo2}
                    onChange={e => setVitals(v => ({ ...v, spo2: e.target.value }))} className="input-field" />
                <input type="number" placeholder="Systolic BP (mmHg)" value={vitals.systolic_bp}
                    onChange={e => setVitals(v => ({ ...v, systolic_bp: e.target.value }))} className="input-field" />
            </div>

            {/* Symptoms */}
            <div className="textarea-wrap" style={{ marginBottom: 8 }}>
                <textarea
                    value={symptoms}
                    onChange={e => setSymptoms(e.target.value)}
                    onKeyDown={e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleSubmit(); }}
                    placeholder="Describe your symptoms in detail... (e.g., persistent dry cough for 3 days, mild fever 38°C, fatigue, no chest pain)"
                    rows={4}
                    className="input-field"
                />
                <button onClick={handleSubmit} disabled={loading || !symptoms.trim()} className="submit-btn">
                    {loading ? <Loader2 className="animate-spin" style={{ width: 16, height: 16 }} /> : <Send style={{ width: 16, height: 16 }} />}
                </button>
            </div>
            <p className="form-hint">
                Tip: Press <kbd>⌘ Enter</kbd> to submit
            </p>

            {/* Classification + RAG pre-stage info */}
            {(classification || ragInfo) && (
                <div className="prestage-cards" style={{ marginTop: 16 }}>
                    {/* Classification Card */}
                    {classification && (
                        <div className="classification-card animate-fade-in"
                            style={{
                                background: SEVERITY_COLORS[classification.severity]?.bg || "rgba(148,163,184,0.08)",
                                border: `1px solid ${SEVERITY_COLORS[classification.severity]?.border || "rgba(148,163,184,0.2)"}`,
                            }}>
                            <div className="classification-card__header">
                                <Stethoscope style={{
                                    width: 16, height: 16,
                                    color: SEVERITY_COLORS[classification.severity]?.color || "#94a3b8"
                                }} />
                                <span className="classification-card__title">Local ML Classification</span>
                                <span className="classification-card__badge"
                                    style={{
                                        color: SEVERITY_COLORS[classification.severity]?.color || "#94a3b8",
                                        background: SEVERITY_COLORS[classification.severity]?.bg || "rgba(148,163,184,0.08)",
                                        border: `1px solid ${SEVERITY_COLORS[classification.severity]?.border || "rgba(148,163,184,0.2)"}`,
                                    }}>
                                    {classification.severity}
                                </span>
                            </div>
                            <div className="classification-card__body">
                                <div className="classification-card__label">{classification.label}</div>
                                <div className="classification-card__desc">{classification.description}</div>
                                <div className="classification-card__conf">
                                    Confidence: {Math.round(classification.confidence * 100)}%
                                </div>
                                <div className="classification-card__action">
                                    <strong>Recommended:</strong> {classification.action}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* RAG Info Card */}
                    {ragInfo && ragInfo.documents_found > 0 && (
                        <div className="rag-card animate-fade-in">
                            <div className="rag-card__header">
                                <Search style={{ width: 15, height: 15, color: "#8b5cf6" }} />
                                <span className="rag-card__title">Knowledge Base Context</span>
                                <span className="rag-card__count">{ragInfo.documents_found} sources</span>
                            </div>
                            <div className="rag-card__topics">
                                {ragInfo.topics.map((topic, i) => (
                                    <span key={i} className="rag-card__topic">{topic}</span>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Stage progress */}
            {(loading || result) && (
                <div className="stage-bar" style={{ marginTop: 20 }}>
                    <div className="stage-bar__header">
                        <Activity style={{ width: 14, height: 14, color: "#3b82f6" }} />
                        <span className="stage-bar__header-text">Council Progress</span>
                        {allComplete && (
                            <span className="stage-bar__complete">
                                <CheckCircle /> Complete
                            </span>
                        )}
                    </div>
                    <div className="stages">
                        {STAGES.map(({ key, label, icon, desc }) => {
                            const status = stages[key];
                            return (
                                <div key={key} className={`stage-item stage-item--${status}`}>
                                    <div className="stage-item__top">
                                        {status === "running" ? <Loader2 className="animate-spin" style={{ width: 12, height: 12 }} /> :
                                            status === "complete" ? <CheckCircle style={{ width: 12, height: 12 }} /> :
                                                icon}
                                        <span className="stage-item__label">{label}</span>
                                    </div>
                                    <div className="stage-item__desc">{desc}</div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Results */}
            {result?.synthesis && (
                <div style={{ marginTop: 20, display: "flex", flexDirection: "column", gap: 12 }}>
                    {/* Consensus card */}
                    <div className="result-card">
                        <div className="result-card__header">
                            <div className="result-card__icon">
                                <Sparkles />
                            </div>
                            <span className="result-card__title">Council Consensus</span>
                            {result.synthesis.confidence !== undefined && (
                                <span className="result-card__conf">
                                    {Math.round(result.synthesis.confidence * 100)}% confidence
                                </span>
                            )}
                        </div>

                        {result.synthesis.summary && (
                            <p className="result-card__summary">{result.synthesis.summary}</p>
                        )}

                        <div className="result-cols">
                            {result.synthesis.final_differentials && (
                                <div>
                                    <div className="result-col__label">Differential Diagnoses</div>
                                    <ul className="result-list">
                                        {result.synthesis.final_differentials.map((d, i) => (
                                            <li key={i} className="result-list__item">
                                                <span className="result-list__dot" style={{ background: "#3b82f6" }} />
                                                {d}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                            {result.synthesis.recommended_next_steps && (
                                <div>
                                    <div className="result-col__label">Recommended Next Steps</div>
                                    <ul className="result-list">
                                        {result.synthesis.recommended_next_steps.map((s, i) => (
                                            <li key={i} className="result-list__item">
                                                <span className="result-list__dot" style={{ background: "#10b981" }} />
                                                {s}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </div>

                        {result.synthesis.confidence !== undefined && (
                            <div className="conf-bar-wrap">
                                <div className="conf-bar-track">
                                    <div className="conf-bar-fill" style={{ width: `${result.synthesis.confidence * 100}%` }} />
                                </div>
                            </div>
                        )}

                        <hr className="result-divider" />

                        <div className="feedback-row">
                            {!feedbackSent ? (
                                <>
                                    <span className="feedback-text">Was this helpful? Your feedback trains the model privately.</span>
                                    <div className="feedback-btns">
                                        <button onClick={() => handleFeedback(true)} className="feedback-btn feedback-btn--yes">
                                            <ThumbsUp /> Helpful
                                        </button>
                                        <button onClick={() => handleFeedback(false)} className="feedback-btn feedback-btn--no">
                                            <ThumbsDown /> Not helpful
                                        </button>
                                    </div>
                                </>
                            ) : (
                                <span className="feedback-sent">
                                    <CheckCircle /> Feedback submitted privately via federated learning. Thank you.
                                </span>
                            )}
                        </div>
                    </div>

                    {/* Individual member responses */}
                    {result.divergence && (
                        <div>
                            <button className="members-toggle" onClick={() => setShowMembers(!showMembers)}>
                                <Bot style={{ width: 15, height: 15 }} />
                                Individual Council Responses
                                <span className="members-toggle__chevron">
                                    {showMembers
                                        ? <ChevronUp style={{ width: 15, height: 15 }} />
                                        : <ChevronDown style={{ width: 15, height: 15 }} />}
                                </span>
                            </button>
                            {showMembers && (
                                <div className="animate-fade-in" style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 8 }}>
                                    {Object.entries(result.divergence).map(([key, val]) => (
                                        <MemberTab key={key} memberKey={key} data={val} />
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
