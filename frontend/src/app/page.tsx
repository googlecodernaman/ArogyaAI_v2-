"use client";
import { useEffect } from "react";
import ChatInterface from "@/components/ChatInterface";
import {
  Activity, Shield, Brain, Zap, Lock, GitMerge,
  Cpu, ChevronDown, Server, Eye, Database, Sparkles,
  Search, FileText, Stethoscope, HardDrive
} from "lucide-react";

const FEATURES = [
  { icon: Brain, color: "#93c5fd", bg: "rgba(59,130,246,0.1)", border: "rgba(59,130,246,0.2)", title: "LLM Council", desc: "Three independent AI models deliberate in parallel — Divergence, Convergence, Synthesis — before reaching consensus." },
  { icon: Shield, color: "#6ee7b7", bg: "rgba(16,185,129,0.1)", border: "rgba(16,185,129,0.2)", title: "Zero PII Exposure", desc: "All personally identifiable information is stripped client-side before any network call. Your raw data never leaves the browser." },
  { icon: Search, color: "#c4b5fd", bg: "rgba(139,92,246,0.1)", border: "rgba(139,92,246,0.2)", title: "RAG Knowledge Base", desc: "FAISS-indexed medical knowledge from AHA/ACC guidelines. Retrieval-Augmented Generation gives LLMs clinical context." },
  { icon: Stethoscope, color: "#f472b6", bg: "rgba(236,72,153,0.1)", border: "rgba(236,72,153,0.2)", title: "Local ML Classifier", desc: "On-device symptom classifier predicts cardiac category and severity using TF-IDF + Logistic Regression — no cloud calls." },
  { icon: FileText, color: "#fbbf24", bg: "rgba(245,158,11,0.1)", border: "rgba(245,158,11,0.2)", title: "Medical Report Upload", desc: "Upload your lab reports and records (PDF/DOCX). They're indexed locally and used for personalised RAG context." },
  { icon: HardDrive, color: "#93c5fd", bg: "rgba(59,130,246,0.1)", border: "rgba(59,130,246,0.2)", title: "Hospital DB (Local)", desc: "SQLite-based edge storage for anonymised consultations and federated contributions. Your medical timeline, on your device." },
  { icon: Lock, color: "#6ee7b7", bg: "rgba(16,185,129,0.1)", border: "rgba(16,185,129,0.2)", title: "Differential Privacy", desc: "Federated learning updates are protected by gradient clipping and Gaussian noise before leaving your device." },
  { icon: GitMerge, color: "#c4b5fd", bg: "rgba(139,92,246,0.1)", border: "rgba(139,92,246,0.2)", title: "Federated Learning", desc: "Your anonymised corrections improve the global model via FedAvg — without ever sharing your raw medical data." },
  { icon: Database, color: "#fbbf24", bg: "rgba(245,158,11,0.1)", border: "rgba(245,158,11,0.2)", title: "Encrypted Storage", desc: "Medical history stored locally in IndexedDB with AES-256-GCM encryption. Only you hold the key." },
];

const PIPELINE = [
  { step: "01", icon: Shield, color: "#6ee7b7", bg: "rgba(16,185,129,0.1)", border: "rgba(16,185,129,0.2)", title: "PII Sanitize", desc: "Names, dates, contacts stripped client-side before any network call." },
  { step: "02", icon: Zap, color: "#93c5fd", bg: "rgba(59,130,246,0.1)", border: "rgba(59,130,246,0.2)", title: "Red-Flag Triage", desc: "Deterministic vital-sign check. Emergencies flagged instantly — no LLM needed." },
  { step: "03", icon: Stethoscope, color: "#f472b6", bg: "rgba(236,72,153,0.1)", border: "rgba(236,72,153,0.2)", title: "ML Classify + RAG", desc: "Local model classifies symptoms. FAISS retrieves relevant medical knowledge." },
  { step: "04", icon: Brain, color: "#c4b5fd", bg: "rgba(139,92,246,0.1)", border: "rgba(139,92,246,0.2)", title: "Council Deliberation", desc: "3 models reason in parallel. Reviewer ranks. Chairman synthesises." },
  { step: "05", icon: HardDrive, color: "#fbbf24", bg: "rgba(245,158,11,0.1)", border: "rgba(245,158,11,0.2)", title: "Local DB + FL", desc: "Anonymised results stored locally. Feedback improves global model via federated learning." },
];

const COUNCIL = [
  { label: "Member A", model: "Llama 3.3 70B", provider: "Groq", role: "Divergence", color: "#3b82f6", icon: Brain },
  { label: "Member B", model: "DeepSeek Chat", provider: "OpenRouter", role: "Divergence", color: "#10b981", icon: Zap },
  { label: "Member C", model: "Mistral Small", provider: "Mistral AI", role: "Divergence", color: "#8b5cf6", icon: Cpu },
  { label: "Reviewer", model: "Llama 3.1 8B", provider: "Groq", role: "Convergence", color: "#06b6d4", icon: Eye },
  { label: "Chairman", model: "Llama 3.3 70B", provider: "Groq", role: "Synthesis", color: "#f59e0b", icon: GitMerge },
];

export default function Home() {
  /* Parallax orbs */
  useEffect(() => {
    const orbs = document.querySelectorAll<HTMLElement>(".orb");
    const onMove = (e: MouseEvent) => {
      const cx = window.innerWidth / 2, cy = window.innerHeight / 2;
      const dx = (e.clientX - cx) / cx, dy = (e.clientY - cy) / cy;
      orbs.forEach((orb, i) => {
        const f = (i + 1) * 14;
        orb.style.transform = `translate(${dx * f}px, ${dy * f}px)`;
      });
    };
    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, []);

  return (
    <>
      {/* Background */}
      <div className="bg-grid" />
      <div className="orbs">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
      </div>

      {/* Navbar */}
      <nav className="navbar">
        <div className="navbar__logo">
          <div className="navbar__logo-icon"><Activity /></div>
          <span className="navbar__logo-name">MEDORBY</span>
          <span className="navbar__badge">v1.0 Beta</span>
        </div>
        <div className="navbar__links">
          <a href="#how-it-works" className="navbar__link">How it works</a>
          <a href="#council" className="navbar__link">Council</a>
          <a href="#chat" className="navbar__cta">
            <Zap style={{ width: 14, height: 14 }} /> Try Now
          </a>
        </div>
      </nav>

      {/* Hero */}
      <section className="hero">
        <div className="hero__badge">
          <span className="hero__badge-dot">
            <span className="hero__badge-dot-ping" />
            <span className="hero__badge-dot-inner" />
          </span>
          PII Never Leaves Your Device · Zero-Trust Architecture
        </div>

        <h1 className="hero__title">
          <span className="hero__title-grad">Medical AI</span>
          <br />
          <span>Built on </span>
          <span className="hero__title-grad">Council Consensus</span>
        </h1>

        <p className="hero__sub">
          Three independent LLMs deliberate your case in parallel. A reviewer ranks them.
          A chairman synthesises the final answer — all in{" "}
          <strong>under 2.5 seconds</strong>.
        </p>

        <div className="hero__ctas">
          <a href="#chat" className="btn-primary">
            <Zap /> Start Consultation
          </a>
          <a href="#how-it-works" className="btn-secondary">
            <Brain /> How it works
          </a>
        </div>

        <div className="hero__stats">
          {[
            { value: "~2.2s", label: "Full Council" },
            { value: "3+1+1", label: "AI Models" },
            { value: "AES-256", label: "Encryption" },
          ].map(({ value, label }) => (
            <div key={label} style={{ textAlign: "center" }}>
              <div className="hero__stat-value">{value}</div>
              <div className="hero__stat-label">{label}</div>
            </div>
          ))}
        </div>

        <a href="#how-it-works" className="hero__scroll">
          <span>Scroll to explore</span>
          <ChevronDown />
        </a>
      </section>

      {/* How it works */}
      <section id="how-it-works" className="section">
        <div className="container">
          <div className="section__header">
            <div className="section__label section__label--blue">Architecture</div>
            <h2 className="section__title">How MEDORBY Works</h2>
            <p className="section__sub">
              A privacy-first pipeline that keeps your data local while leveraging the world's fastest LLM inference.
            </p>
          </div>

          {/* Pipeline */}
          <div className="pipeline">
            {PIPELINE.map(({ step, icon: Icon, color, bg, border, title, desc }) => (
              <div key={step} className="pipeline__step"
                style={{ background: bg, border: `1px solid ${border}` }}
                onMouseEnter={e => (e.currentTarget.style.boxShadow = `0 0 28px ${color}22`)}
                onMouseLeave={e => (e.currentTarget.style.boxShadow = "none")}>
                <div className="pipeline__num" style={{ color }}>{step}</div>
                <div className="pipeline__icon" style={{ background: `${color}18`, border: `1px solid ${color}33` }}>
                  <Icon style={{ color }} />
                </div>
                <div className="pipeline__title">{title}</div>
                <div className="pipeline__desc">{desc}</div>
              </div>
            ))}
          </div>

          {/* Features */}
          <div className="features">
            {FEATURES.map(({ icon: Icon, color, bg, border, title, desc }) => (
              <div key={title} className="feature-card"
                onMouseEnter={e => { e.currentTarget.style.borderColor = border; e.currentTarget.style.boxShadow = `0 0 24px ${color}22`; e.currentTarget.style.transform = "translateY(-4px)"; }}
                onMouseLeave={e => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.07)"; e.currentTarget.style.boxShadow = "none"; e.currentTarget.style.transform = "translateY(0)"; }}>
                <div className="feature-card__icon" style={{ background: bg, border: `1px solid ${border}` }}>
                  <Icon style={{ color }} />
                </div>
                <div className="feature-card__title">{title}</div>
                <div className="feature-card__desc">{desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Council */}
      <section id="council" className="section" style={{ paddingTop: 0 }}>
        <div className="container">
          <div className="section__header">
            <div className="section__label section__label--violet">The Council</div>
            <h2 className="section__title">Five AI Minds, One Answer</h2>
            <p className="section__sub">
              Each council member brings a different perspective. The chairman synthesises the consensus.
            </p>
          </div>

          <div className="council-grid">
            {COUNCIL.map(({ label, model, provider, role, color, icon: Icon }) => (
              <div key={label} className="council-card"
                onMouseEnter={e => { e.currentTarget.style.borderColor = `${color}44`; e.currentTarget.style.boxShadow = `0 0 24px ${color}22`; e.currentTarget.style.transform = "translateY(-5px)"; }}
                onMouseLeave={e => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.07)"; e.currentTarget.style.boxShadow = "none"; e.currentTarget.style.transform = "translateY(0)"; }}>
                <div className="council-card__icon" style={{ background: `${color}18`, border: `1px solid ${color}33` }}>
                  <Icon style={{ color }} />
                </div>
                <div className="council-card__label">{label}</div>
                <div className="council-card__model">{model}</div>
                <div className="council-card__model" style={{ opacity: 0.5, fontSize: "0.7rem" }}>{provider}</div>
                <span className="council-card__badge" style={{ background: `${color}18`, color, border: `1px solid ${color}33` }}>
                  {role}
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Chat */}
      <section id="chat" className="chat-section">
        <div className="container container--narrow">
          <div className="section__header">
            <div className="section__label section__label--blue" style={{ display: "inline-flex" }}>
              <span style={{ position: "relative", width: 8, height: 8, display: "inline-block", marginRight: 4 }}>
                <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: "#3b82f6", animation: "ping 1.5s ease-out infinite" }} />
                <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: "#3b82f6" }} />
              </span>
              Live · Backend Connected
            </div>
            <h2 className="section__title">Start Your Consultation</h2>
            <p className="section__sub">
              Your input is sanitised before leaving the browser. No raw data is ever transmitted.
            </p>
          </div>

          {/* Disclaimer */}
          <div className="disclaimer">
            <span className="disclaimer__icon">⚠️</span>
            <span>
              <strong>Medical Disclaimer:</strong> MEDORBY is an AI research assistant only.
              It does not replace professional medical advice. Always consult a qualified healthcare provider.
            </span>
          </div>

          {/* Chat card */}
          <div className="chat-card">
            <ChatInterface />
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="footer__inner">
          <div className="footer__logo">
            <div className="footer__logo-icon"><Activity /></div>
            <span className="footer__logo-name">MEDORBY</span>
            <span style={{ fontSize: 12, color: "#475569" }}>v1.0</span>
          </div>
          <div className="footer__links">
            <span>Local-First</span>
            <span>·</span>
            <span>Privacy-Preserving</span>
            <span>·</span>
            <span>LLM Council Architecture</span>
          </div>
          <div className="footer__powered">
            <Server style={{ width: 13, height: 13 }} />
            <span>Powered by Groq</span>
          </div>
        </div>
      </footer>
    </>
  );
}
