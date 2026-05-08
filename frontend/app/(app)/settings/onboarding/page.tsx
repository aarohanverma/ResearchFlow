"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { CheckIcon, ChevronRightIcon } from "lucide-react";

const STEPS = ["Subjects", "Topics", "Depth", "Notifications"];

const SUBJECTS: string[] = [
  "Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative Biology", "Economics",
];

interface TopicEntry { key: string; label: string }

const TOPICS: Record<string, TopicEntry[]> = {
  "Computer Science": [
    { key: "cs.AI",  label: "Artificial Intelligence" },
    { key: "cs.LG",  label: "Machine Learning" },
    { key: "cs.CL",  label: "NLP" },
    { key: "cs.CV",  label: "Computer Vision" },
    { key: "cs.RO",  label: "Robotics" },
    { key: "cs.MA",  label: "Multi-Agent Systems" },
    { key: "cs.CR",  label: "Security" },
    { key: "cs.IR",  label: "Information Retrieval" },
    { key: "cs.HC",  label: "HCI" },
    { key: "cs.NE",  label: "Neural & Evolutionary" },
    { key: "cs.DC",  label: "Distributed Computing" },
    { key: "cs.SE",  label: "Software Engineering" },
  ],
  "Physics": [
    { key: "quant-ph",          label: "Quantum Physics" },
    { key: "hep-th",            label: "High Energy Theory" },
    { key: "hep-ph",            label: "High Energy Phenomenology" },
    { key: "cond-mat.stat-mech",label: "Statistical Mechanics" },
    { key: "cond-mat.str-el",   label: "Strongly Correlated Systems" },
    { key: "astro-ph.CO",       label: "Cosmology" },
    { key: "astro-ph.HE",       label: "High Energy Astrophysics" },
  ],
  "Mathematics": [
    { key: "math.OC", label: "Optimization & Control" },
    { key: "math.PR", label: "Probability" },
    { key: "math.ST", label: "Statistics" },
    { key: "math.CO", label: "Combinatorics" },
    { key: "math.AP", label: "Analysis of PDEs" },
  ],
  "Statistics": [
    { key: "stat.ML", label: "Statistical Machine Learning" },
    { key: "stat.ME", label: "Methodology" },
    { key: "stat.TH", label: "Theory" },
  ],
  "Quantitative Biology": [
    { key: "q-bio.NC", label: "Computational Neuroscience" },
    { key: "q-bio.GN", label: "Genomics" },
    { key: "q-bio.QM", label: "Quantitative Methods" },
  ],
  "Economics": [
    { key: "econ.TH", label: "Economic Theory" },
    { key: "econ.EM", label: "Econometrics" },
  ],
};

export default function OnboardingPage() {
  const router = useRouter();
  const [step, setStep] = useState(0);
  const [selectedSubjects, setSelectedSubjects] = useState<string[]>([]);
  const [selectedNamespaces, setSelectedNamespaces] = useState<string[]>([]);
  const [expertise, setExpertise] = useState("practitioner");
  const [orientation, setOrientation] = useState("both");
  const [notifications, setNotifications] = useState({
    notify_potd: true,
    notify_digest: true,
  });
  const [submitting, setSubmitting] = useState(false);

  function toggleSubject(s: string) {
    setSelectedSubjects((prev) =>
      prev.includes(s) ? prev.filter((x) => x !== s) : [...prev, s]
    );
  }

  function toggleNamespace(key: string) {
    setSelectedNamespaces((prev) =>
      prev.includes(key) ? prev.filter((x) => x !== key) : [...prev, key]
    );
  }

  async function finish() {
    setSubmitting(true);
    try {
      await api.post("/settings/onboarding", {
        subjects: selectedSubjects,
        topics: selectedNamespaces,
        expertise_level: expertise,
        orientation,
        ...notifications,
      });
      router.push("/feed");
    } catch (err) {
      console.error(err);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center p-6">
      <div className="w-full max-w-xl">
        {/* Progress */}
        <div className="flex gap-2 mb-8">
          {STEPS.map((s, i) => (
            <div
              key={s}
              className={`flex-1 h-1.5 rounded-full transition-colors ${
                i <= step ? "bg-brand" : "bg-gray-800"
              }`}
            />
          ))}
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-8">
          {step === 0 && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-bold text-white mb-1">Pick your research areas</h2>
                <p className="text-sm text-gray-400">Select at least one subject.</p>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {SUBJECTS.map((s) => (
                  <button
                    key={s}
                    onClick={() => toggleSubject(s)}
                    className={`flex items-center justify-between px-4 py-3 rounded-xl border text-sm font-medium transition-colors ${
                      selectedSubjects.includes(s)
                        ? "border-brand bg-indigo-950/40 text-indigo-200"
                        : "border-gray-700 bg-gray-800 text-gray-300 hover:border-gray-600"
                    }`}
                  >
                    {s}
                    {selectedSubjects.includes(s) && <CheckIcon size={14} />}
                  </button>
                ))}
              </div>
            </div>
          )}

          {step === 1 && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-bold text-white mb-1">Choose specific topics</h2>
                <p className="text-sm text-gray-400">Select topics within your chosen subjects.</p>
              </div>
              <div className="space-y-5 max-h-80 overflow-y-auto pr-1">
                {selectedSubjects.map((subject) => (
                  <div key={subject}>
                    <p className="text-xs text-gray-500 uppercase tracking-wide mb-2">{subject}</p>
                    <div className="flex flex-wrap gap-2">
                      {(TOPICS[subject] || []).map(({ key, label }) => (
                        <button
                          key={key}
                          onClick={() => toggleNamespace(key)}
                          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                            selectedNamespaces.includes(key)
                              ? "bg-brand text-white"
                              : "bg-gray-800 text-gray-300 hover:bg-gray-700"
                          }`}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-bold text-white mb-1">Your depth profile</h2>
                <p className="text-sm text-gray-400">Tailors explanations to your background.</p>
              </div>

              <div className="space-y-4">
                <div>
                  <p className="text-sm text-gray-400 mb-2">Expertise level</p>
                  <div className="grid grid-cols-3 gap-2">
                    {[
                      { key: "newcomer", label: "Newcomer", desc: "New to the field" },
                      { key: "practitioner", label: "Practitioner", desc: "Industry/research background" },
                      { key: "expert", label: "Expert", desc: "Deep domain expertise" },
                    ].map(({ key, label, desc }) => (
                      <button
                        key={key}
                        onClick={() => setExpertise(key)}
                        className={`p-3 rounded-xl border text-left transition-colors ${
                          expertise === key
                            ? "border-brand bg-indigo-950/40"
                            : "border-gray-700 bg-gray-800 hover:border-gray-600"
                        }`}
                      >
                        <p className="text-sm font-medium text-white">{label}</p>
                        <p className="text-xs text-gray-500 mt-0.5">{desc}</p>
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <p className="text-sm text-gray-400 mb-2">Orientation</p>
                  <div className="grid grid-cols-3 gap-2">
                    {[
                      { key: "research", label: "Research" },
                      { key: "production", label: "Production" },
                      { key: "both", label: "Both" },
                    ].map(({ key, label }) => (
                      <button
                        key={key}
                        onClick={() => setOrientation(key)}
                        className={`py-2 rounded-xl border text-sm font-medium transition-colors ${
                          orientation === key
                            ? "border-brand bg-indigo-950/40 text-indigo-200"
                            : "border-gray-700 bg-gray-800 text-gray-300 hover:border-gray-600"
                        }`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {step === 3 && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-bold text-white mb-1">Email notifications</h2>
                <p className="text-sm text-gray-400">Changeable any time from Settings.</p>
              </div>
              <div className="space-y-3">
                {[
                  { key: "notify_potd", label: "Paper of the Day", desc: "One hand-picked paper every morning" },
                  { key: "notify_digest", label: "Weekly Digest", desc: "Top papers from your week" },
                ].map(({ key, label, desc }) => (
                  <label key={key} className="flex items-center justify-between p-4 bg-gray-800 rounded-xl cursor-pointer hover:bg-gray-750">
                    <div>
                      <p className="text-sm font-medium text-white">{label}</p>
                      <p className="text-xs text-gray-500 mt-0.5">{desc}</p>
                    </div>
                    <input
                      type="checkbox"
                      checked={(notifications as any)[key]}
                      onChange={(e) =>
                        setNotifications((n) => ({ ...n, [key]: e.target.checked }))
                      }
                      className="w-5 h-5 accent-brand"
                    />
                  </label>
                ))}
              </div>
            </div>
          )}

          {/* Navigation */}
          <div className="flex gap-3 mt-8">
            {step > 0 && (
              <button
                onClick={() => setStep((s) => s - 1)}
                className="flex-1 py-2.5 rounded-xl border border-gray-700 text-gray-300 text-sm font-medium hover:bg-gray-800 transition-colors"
              >
                Back
              </button>
            )}
            <button
              onClick={() => {
                if (step < STEPS.length - 1) setStep((s) => s + 1);
                else finish();
              }}
              disabled={
                (step === 0 && selectedSubjects.length === 0) ||
                (step === 1 && selectedNamespaces.length === 0) ||
                submitting
              }
              className="flex-1 flex items-center justify-center gap-2 bg-brand hover:bg-indigo-600 disabled:opacity-40 text-white font-semibold py-2.5 rounded-xl transition-colors"
            >
              {step === STEPS.length - 1 ? (submitting ? "Setting up…" : "Go to Feed") : "Continue"}
              {step < STEPS.length - 1 && <ChevronRightIcon size={16} />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
