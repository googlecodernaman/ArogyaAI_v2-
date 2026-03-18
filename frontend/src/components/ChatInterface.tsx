"use client";
import type { ReactNode } from "react";
import { useState, useEffect, useRef, useCallback } from "react";
import {
    Send, Shield, Activity, Brain, CheckCircle,
    Loader2, ThumbsUp, ThumbsDown, AlertTriangle,
    ChevronDown, ChevronUp, Bot, Cpu, Zap, GitMerge,
    Phone, Eye, Sparkles, Upload, FileText, Trash2,
    Database, Search, Stethoscope, X, Layers
} from "lucide-react";
import { buildSanitizedPrompt } from "@/lib/sanitizer";
import { submitFederatedUpdate } from "@/lib/local_learning";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

/* ── Types ─────────────────────────────────────────────────────────────────── */
interface StageStatus {
    divergence:    "idle" | "running" | "complete";
    convergence:   "idle" | "running" | "complete";
    synthesis:     "idle" | "running" | "complete";
    hybrid_fusion: "idle" | "running" | "complete";
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

interface StructuredPrediction {
    prediction: {
        category: string;
        label: string;
        severity: string;
        confidence: number;
        probabilities: Record<string, number>;
        model: string;
    } | null;
    vital_interpretation: Record<string, string>;
}

interface HybridDecision {
    final_category: string;
    label: string;
    severity: string;
    confidence: number;
    is_emergency: boolean;
    recommended_action: string;
    score_breakdown: Record<string, number>;
    sources: {
        text_classifier: { category: string | null; confidence: number | null };
        federated_nn:    { category: string | null; confidence: number | null };
        llm_council:     { category: string | null; confidence: number | null };
    };
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
    member_b: { label: "Member B", model: "DeepSeek Chat", provider: "OpenRouter", icon: <Zap style={{ width: 14, height: 14 }} />, color: "#6ee7b7", bg: "rgba(16,185,129,0.08)", border: "rgba(16,185,129,0.2)" },
    member_c: { label: "Member C", model: "Mistral Small", provider: "Mistral AI", icon: <Cpu style={{ width: 14, height: 14 }} />, color: "#c4b5fd", bg: "rgba(139,92,246,0.08)", border: "rgba(139,92,246,0.2)" },
};

const STAGES = [
    { key: "divergence"    as const, label: "Divergence",  icon: <Brain style={{ width: 13, height: 13 }} />, desc: "3 models reasoning" },
    { key: "convergence"   as const, label: "Convergence", icon: <Eye style={{ width: 13, height: 13 }} />,   desc: "Peer review" },
    { key: "synthesis"     as const, label: "Synthesis",   icon: <GitMerge style={{ width: 13, height: 13 }} />, desc: "Chairman decides" },
    { key: "hybrid_fusion" as const, label: "Fusion",      icon: <Layers style={{ width: 13, height: 13 }} />,  desc: "Hybrid decision" },
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

/* ── Structured (FNN) prediction card ─────────────────────────────────────── */
const VITAL_STATUS_COLORS: Record<string, string> = {
    normal: "#6ee7b7", elevated: "#fbbf24",
    stage1_hypertension: "#f97316", stage2_hypertension: "#ef4444",
    hypertensive_crisis: "#b91c1c",
    tachycardia: "#f97316", severe_tachycardia: "#ef4444",
    bradycardia: "#93c5fd", severe_bradycardia: "#3b82f6",
    mild_hypoxia: "#fbbf24", severe_hypoxia: "#f97316", critical_hypoxia: "#ef4444",
    overweight: "#fbbf24", obesity_class1: "#f97316",
    obesity_class2: "#ef4444", morbid_obesity: "#b91c1c", underweight: "#93c5fd",
};

function StructuredPredictionCard({ data }: { data: StructuredPrediction }) {
    const { prediction, vital_interpretation } = data;
    if (!prediction) return null;
    const sev = SEVERITY_COLORS[prediction.severity];
    return (
        <div className="classification-card animate-fade-in" style={{
            background: sev?.bg || "rgba(148,163,184,0.08)",
            border: `1px solid ${sev?.border || "rgba(148,163,184,0.2)"}`,
        }}>
            <div className="classification-card__header">
                <Activity style={{ width: 16, height: 16, color: sev?.color || "#94a3b8" }} />
                <span className="classification-card__title">Federated NN (Vitals)</span>
                <span className="classification-card__badge" style={{
                    color: sev?.color, background: sev?.bg,
                    border: `1px solid ${sev?.border}`,
                }}>
                    {prediction.severity}
                </span>
            </div>
            <div className="classification-card__body">
                <div className="classification-card__label">{prediction.label}</div>
                <div className="classification-card__conf">
                    Confidence: {Math.round(prediction.confidence * 100)}%
                </div>
                {Object.keys(vital_interpretation).length > 0 && (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginTop: 6 }}>
                        {Object.entries(vital_interpretation).map(([k, v]) => (
                            <span key={k} style={{
                                fontSize: 11, padding: "2px 7px", borderRadius: 10,
                                background: "rgba(0,0,0,0.15)",
                                color: VITAL_STATUS_COLORS[v] || "#94a3b8",
                                border: `1px solid ${VITAL_STATUS_COLORS[v] || "#94a3b8"}40`,
                            }}>
                                {k.replace(/_/g, " ")}: {v.replace(/_/g, " ")}
                            </span>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}

/* ── Hybrid decision fusion card ───────────────────────────────────────────── */
function HybridDecisionCard({ hybrid }: { hybrid: HybridDecision }) {
    const [showBreakdown, setShowBreakdown] = useState(false);
    const sev = SEVERITY_COLORS[hybrid.severity] ?? SEVERITY_COLORS.moderate;

    const sourceRows = [
        { key: "Federated NN",      data: hybrid.sources.federated_nn },
        { key: "Text Classifier",   data: hybrid.sources.text_classifier },
        { key: "LLM Council",       data: hybrid.sources.llm_council },
    ];
    const topScore = Math.max(...Object.values(hybrid.score_breakdown));

    return (
        <div className="result-card animate-fade-in" style={{
            border: `1px solid ${hybrid.is_emergency ? "rgba(239,68,68,0.4)" : sev.border}`,
            background: hybrid.is_emergency ? "rgba(239,68,68,0.06)" : sev.bg,
        }}>
            {/* Header */}
            <div className="result-card__header">
                <div className="result-card__icon">
                    <Layers />
                </div>
                <span className="result-card__title">Hybrid Decision Fusion</span>
                <span className="result-card__conf" style={{ color: sev.color }}>
                    {Math.round(hybrid.confidence * 100)}% confidence
                </span>
            </div>

            {/* Primary verdict */}
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                <span style={{
                    fontSize: 15, fontWeight: 600, color: sev.color,
                }}>
                    {hybrid.label}
                </span>
                <span style={{
                    fontSize: 11, padding: "2px 8px", borderRadius: 10,
                    background: sev.bg, color: sev.color,
                    border: `1px solid ${sev.border}`,
                }}>
                    {hybrid.severity}
                </span>
                {hybrid.is_emergency && (
                    <span style={{
                        fontSize: 11, padding: "2px 8px", borderRadius: 10,
                        background: "rgba(239,68,68,0.12)", color: "#f87171",
                        border: "1px solid rgba(239,68,68,0.3)", fontWeight: 600,
                    }}>
                        EMERGENCY
                    </span>
                )}
            </div>

            {/* Recommended action */}
            <div style={{
                padding: "8px 12px", borderRadius: 6, marginBottom: 12,
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
                fontSize: 13, color: "#e2e8f0",
            }}>
                <strong style={{ color: sev.color }}>Action: </strong>
                {hybrid.recommended_action}
            </div>

            {/* Score bar chart */}
            <div style={{ marginBottom: 10 }}>
                {Object.entries(hybrid.score_breakdown).map(([cat, score]) => (
                    <div key={cat} style={{ marginBottom: 4 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#94a3b8", marginBottom: 2 }}>
                            <span>{cat.replace(/_/g, " ")}</span>
                            <span>{Math.round(score * 100)}%</span>
                        </div>
                        <div style={{ height: 4, borderRadius: 2, background: "rgba(255,255,255,0.07)" }}>
                            <div style={{
                                height: "100%", borderRadius: 2,
                                width: `${(score / (topScore || 1)) * 100}%`,
                                background: cat === hybrid.final_category ? sev.color : "rgba(148,163,184,0.3)",
                                transition: "width 0.4s ease",
                            }} />
                        </div>
                    </div>
                ))}
            </div>

            {/* Source breakdown toggle */}
            <button
                onClick={() => setShowBreakdown(!showBreakdown)}
                style={{
                    background: "none", border: "none", cursor: "pointer",
                    display: "flex", alignItems: "center", gap: 5,
                    fontSize: 11, color: "#64748b", padding: 0,
                }}
            >
                {showBreakdown ? <ChevronUp style={{ width: 12, height: 12 }} /> : <ChevronDown style={{ width: 12, height: 12 }} />}
                Source contributions
            </button>
            {showBreakdown && (
                <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 4 }}>
                    {sourceRows.map(({ key, data: src }) => (
                        <div key={key} style={{
                            display: "flex", justifyContent: "space-between",
                            fontSize: 11, padding: "4px 8px", borderRadius: 4,
                            background: "rgba(255,255,255,0.04)",
                        }}>
                            <span style={{ color: "#94a3b8" }}>{key}</span>
                            <span style={{ color: "#e2e8f0" }}>
                                {src.category ?? "—"}
                                {src.confidence != null ? ` · ${Math.round(src.confidence * 100)}%` : ""}
                            </span>
                        </div>
                    ))}
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
    const [stages, setStages] = useState<StageStatus>({ divergence: "idle", convergence: "idle", synthesis: "idle", hybrid_fusion: "idle" });
    const [result, setResult] = useState<CouncilResult | null>(null);
    const [emergency, setEmergency] = useState<Emergency>({ active: false, reason: "", message: "" });
    const [feedbackSent, setFeedbackSent] = useState(false);
    const [showMembers, setShowMembers] = useState(false);
    const clientId = useRef(`client_${Math.random().toString(36).slice(2)}`);

    // New state for RAG/ML features
    const [classification, setClassification] = useState<Classification | null>(null);
    const [ragInfo, setRagInfo] = useState<RAGInfo | null>(null);
    const [structuredPrediction, setStructuredPrediction] = useState<StructuredPrediction | null>(null);
    const [hybridDecision, setHybridDecision] = useState<HybridDecision | null>(null);
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
    useEffect(() => { fetchReports(); }, [fetchReports]);

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
        setStructuredPrediction(null);
        setHybridDecision(null);
        setStages({ divergence: "idle", convergence: "idle", synthesis: "idle", hybrid_fusion: "idle" });
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
                        } else if (event.stage === "structured_prediction" && event.data) {
                            setStructuredPrediction(event.data);
                        } else if (event.stage === "rag_retrieval" && event.data) {
                            setRagInfo(event.data);
                        } else if (event.stage === "divergence") {
                            setStages(s => ({ ...s, divergence: event.status === "complete" ? "complete" : "running" }));
                            if (event.data) setResult(prev => ({ ...(prev || {}), divergence: event.data } as CouncilResult));
                        } else if (event.stage === "convergence") {
                            setStages(s => ({ ...s, convergence: event.status === "complete" ? "complete" : "running" }));
                            if (event.data) setResult(prev => ({ ...(prev || {}), convergence: event.data } as CouncilResult));
                        } else if (event.stage === "synthesis") {
                            setStages(s => ({ ...s, synthesis: event.status === "complete" ? "complete" : "running" }));
                            if (event.data) setResult(prev => ({ ...(prev || {}), synthesis: event.data } as CouncilResult));
                        } else if (event.stage === "hybrid_fusion" && event.data) {
                            setStages(s => ({ ...s, hybrid_fusion: "complete" }));
                            setHybridDecision(event.data);
                        } else if (event.stage === "error") {
                            console.error("Stream error event:", event.message);
                            alert("Council processing error: " + event.message);
                            break;
                        }
                    } catch (err) {
                        console.error("Parse error for part:", part, err);
                    }
                }
            }
            
            // Flush remaining buffer if any
            if (buffer.trim()) {
                const dataLine = buffer.split("\n").find(l => l.startsWith("data: "));
                if (dataLine) {
                    try {
                        const event = JSON.parse(dataLine.slice(6));
                        if (event.stage === "hybrid_fusion" && event.data) {
                            setStages(s => ({ ...s, hybrid_fusion: "complete" }));
                            setHybridDecision(event.data);
                        } else if (event.stage === "synthesis") {
                            setStages(s => ({ ...s, synthesis: event.status === "complete" ? "complete" : "running" }));
                            if (event.data) setResult(prev => ({ ...(prev || {}), synthesis: event.data } as CouncilResult));
                        }
                    } catch (e) {}
                }
            }
        } catch (e) {
            console.error("Council network error:", e);
        } finally {
            setLoading(false);
        }
    }, [symptoms, age, sex, vitals]);

    const handleFeedback = async (positive: boolean) => {
        if (!result?.synthesis?.summary) return;
        await submitFederatedUpdate(clientId.current, result.synthesis.summary, positive ? "Accurate." : "Needs improvement.", BACKEND_URL);
        setFeedbackSent(true);
    };

    const allComplete = stages.divergence === "complete" && stages.convergence === "complete" && stages.synthesis === "complete" && stages.hybrid_fusion === "complete";

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
            {(classification || ragInfo || structuredPrediction) && (
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

                    {/* Structured (FNN) Prediction Card */}
                    {structuredPrediction && (
                        <StructuredPredictionCard data={structuredPrediction} />
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
            {(loading || result || hybridDecision) && (
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
            {(result?.synthesis || hybridDecision) && (
                <div style={{ marginTop: 20, display: "flex", flexDirection: "column", gap: 12 }}>
                    {/* Hybrid Decision Card — primary output */}
                    {hybridDecision && (
                        <HybridDecisionCard hybrid={hybridDecision} />
                    )}

                    {/* Consensus card */}
                    {result?.synthesis && (
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
                    )}

                    {/* Individual member responses */}
                    {result?.divergence && (
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
                                    {Object.entries(result?.divergence ?? {}).map(([key, val]) => (
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
