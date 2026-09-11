const analyzeBtn = document.getElementById("analyze-btn");
const analyzeBtnText = document.getElementById("analyze-btn-text");
const inputText = document.getElementById("input-text");
const resultBox = document.getElementById("result");
const wordCounter = document.getElementById("word-counter");

const toClassName = (label) => label.toLowerCase().replace(/\s+/g, "-");

// Live word counter — rough guide only (the backend measures actual model
// tokens, which don't map 1:1 to words), but gives a sense of length before
// submitting.
function updateWordCounter() {
  const words = inputText.value.trim().split(/\s+/).filter(Boolean);
  const count = words.length;
  wordCounter.textContent = count === 1 ? "1 word" : `${count} words`;
}
inputText.addEventListener("input", updateWordCounter);
updateWordCounter();

// Dark mode toggle — initial theme is already set by the inline script in
// <head> (before first paint); this just handles switching + remembering.
const themeToggleBtn = document.getElementById("theme-toggle");
themeToggleBtn.setAttribute("aria-pressed", document.documentElement.getAttribute("data-theme") === "dark");
themeToggleBtn.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  const next = current === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("psythread-theme", next);
  themeToggleBtn.setAttribute("aria-pressed", next === "dark");
});

// Rotating loading messages — LIME explainability can take a few seconds,
// so a single static "Reading..." can feel stuck. Cycling messages signal
// progress even though the actual work is one request.
const LOADING_MESSAGES = [
  "Reading...",
  "Checking word choices...",
  "Weighing the signal...",
  "Almost done...",
];
let loadingInterval = null;

function startLoadingMessages() {
  let i = 0;
  analyzeBtnText.textContent = LOADING_MESSAGES[0];
  loadingInterval = setInterval(() => {
    i = (i + 1) % LOADING_MESSAGES.length;
    analyzeBtnText.textContent = LOADING_MESSAGES[i];
  }, 1800);
}

function stopLoadingMessages() {
  clearInterval(loadingInterval);
  loadingInterval = null;
}

let currentPrediction = null;

const feedbackBlock = document.getElementById("feedback-block");
const feedbackButtons = document.getElementById("feedback-buttons");
const feedbackCorrection = document.getElementById("feedback-correction");
const feedbackThanks = document.getElementById("feedback-thanks");

async function sendFeedback(wasAccurate, correctedLabel) {
  if (!currentPrediction) return;
  const consentCheckbox = document.getElementById("feedback-consent-checkbox");
  const consentGiven = consentCheckbox ? consentCheckbox.checked : false;
  try {
    await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        predicted_label: currentPrediction.label,
        was_accurate: wasAccurate,
        corrected_label: correctedLabel || null,
        consent_save_text: consentGiven,
        text: consentGiven ? currentPrediction.text : null,
      }),
    });
  } catch (err) {
    console.error("Feedback submission failed:", err);
  }
  feedbackButtons.classList.add("hidden");
  feedbackCorrection.classList.add("hidden");
  feedbackThanks.classList.remove("hidden");
}

document.getElementById("feedback-yes").addEventListener("click", () => sendFeedback(true));
document.getElementById("feedback-no").addEventListener("click", () => {
  feedbackButtons.classList.add("hidden");
  feedbackCorrection.classList.remove("hidden");
});
document.getElementById("feedback-submit-correction").addEventListener("click", () => {
  const correctedLabel = document.getElementById("feedback-correct-label").value;
  sendFeedback(false, correctedLabel);
});

analyzeBtn.addEventListener("click", async () => {
  const text = inputText.value.trim();
  if (!text) return;

  analyzeBtn.disabled = true;
  startLoadingMessages();

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();

    if (data.error) {
      alert(data.error);
      return;
    }

    resultBox.classList.remove("hidden");

    // Reset feedback UI for this new result
    currentPrediction = { label: data.sentiment.label, text: text };
    feedbackButtons.classList.remove("hidden");
    feedbackCorrection.classList.add("hidden");
    feedbackThanks.classList.add("hidden");
    document.getElementById("feedback-correct-label").value = "";
    document.getElementById("feedback-consent-checkbox").checked = false;

    // Segments note — shown when long text was split into multiple chunks
    const segmentsNote = document.getElementById("segments-note");
    if (data.num_segments && data.num_segments > 1) {
      segmentsNote.textContent = `· analyzed as ${data.num_segments} segments`;
      segmentsNote.classList.remove("hidden");
    } else {
      segmentsNote.classList.add("hidden");
    }

    // Primary signal + confidence
    const sentimentBadge = document.getElementById("result-sentiment");
    sentimentBadge.textContent = data.sentiment.label;
    sentimentBadge.className = "badge " + toClassName(data.sentiment.label);
    document.getElementById("result-confidence").textContent =
      `(${(data.sentiment.confidence * 100).toFixed(1)}% confidence)`;

    // Severity / care level — uses its own tier-* color scheme (distinct
    // from the mood-category colors used for the primary signal badge)
    const severityBadge = document.getElementById("result-severity");
    severityBadge.textContent = data.severity.tier;
    severityBadge.className = "badge tier-" + toClassName(data.severity.tier);

    // Translation note — shown when non-English input was auto-translated
    const translationNote = document.getElementById("translation-note");
    if (data.was_translated) {
      translationNote.innerHTML = `Detected <strong>${data.detected_language}</strong> — translated to English for analysis: <em>"${data.translated_text}"</em>`;
      translationNote.classList.remove("hidden");
    } else {
      translationNote.classList.add("hidden");
    }

    // Uncertainty note — only shown when the model's top prediction was
    // below the confidence threshold set server-side.
    const uncertainNote = document.getElementById("uncertain-note");
    uncertainNote.classList.toggle("hidden", !data.sentiment.is_uncertain);

    // Explanation words (LIME) — weighted chips, plum for words that pushed
    // toward the predicted label, sage for words that pushed away from it.
    const explanationBlock = document.getElementById("explanation-block");
    const explanationWords = document.getElementById("explanation-words");
    explanationWords.innerHTML = "";
    if (data.explanation && data.explanation.length) {
      explanationBlock.classList.remove("hidden");
      const maxWeight = Math.max(...data.explanation.map((w) => Math.abs(w.weight)), 0.01);
      data.explanation.forEach(({ word, weight }) => {
        const chip = document.createElement("span");
        chip.className = "explanation-word";
        chip.textContent = word;
        const intensity = Math.min(Math.abs(weight) / maxWeight, 1);
        if (weight >= 0) {
          // plum, intensity-scaled — themed via CSS variables so it adapts to dark mode
          chip.style.background = `rgba(var(--lime-positive-rgb), ${0.12 + intensity * 0.35})`;
          chip.style.color = "var(--lime-positive-text)";
        } else {
          // sage, intensity-scaled — themed via CSS variables so it adapts to dark mode
          chip.style.background = `rgba(var(--lime-negative-rgb), ${0.12 + intensity * 0.35})`;
          chip.style.color = "var(--lime-negative-text)";
        }
        explanationWords.appendChild(chip);
      });
    } else {
      explanationBlock.classList.add("hidden");
    }

    // Possible themes noticed (rule-based keyword tagger)
    const themesBlock = document.getElementById("themes-block");
    const themesList = document.getElementById("themes-list");
    themesList.innerHTML = "";
    if (data.themes && data.themes.length) {
      themesBlock.classList.remove("hidden");
      data.themes.forEach((theme) => {
        const chip = document.createElement("span");
        chip.className = "theme-chip";
        chip.textContent = theme;
        themesList.appendChild(chip);
      });
    } else {
      themesBlock.classList.add("hidden");
    }

    // Crisis resources — only present in the response when severity is
    // Crisis or Red.
    const crisisBox = document.getElementById("crisis-resources");
    const crisisList = document.getElementById("crisis-resources-list");
    crisisList.innerHTML = "";
    if (data.severity.resources && data.severity.resources.length) {
      crisisBox.classList.remove("hidden");
      data.severity.resources.forEach((r) => {
        const li = document.createElement("li");
        li.innerHTML = `<strong>${r.name}</strong> — ${r.contact} <span>(${r.note})</span>`;
        crisisList.appendChild(li);
      });
    } else {
      crisisBox.classList.add("hidden");
    }

    // Matched crisis keywords (secondary safety-net signal)
    const keywordsDiv = document.getElementById("result-keywords");
    keywordsDiv.textContent = data.severity.matched_keywords && data.severity.matched_keywords.length
      ? `Matched keywords: ${data.severity.matched_keywords.join(", ")}`
      : "";
  } catch (err) {
    alert("Something went wrong: " + err.message);
  } finally {
    stopLoadingMessages();
    analyzeBtn.disabled = false;
    analyzeBtnText.textContent = "Read the signal";
  }
});

const batchFileInput = document.getElementById("batch-file-input");
const batchAnalyzeBtn = document.getElementById("batch-analyze-btn");
const batchBtnText = document.getElementById("batch-btn-text");
const batchError = document.getElementById("batch-error");
const batchResults = document.getElementById("batch-results");
const batchSummary = document.getElementById("batch-summary");
const batchTableBody = document.getElementById("batch-table-body");

batchAnalyzeBtn.addEventListener("click", async () => {
  const file = batchFileInput.files[0];
  batchError.classList.add("hidden");
  batchResults.classList.add("hidden");

  if (!file) {
    batchError.textContent = "Choose a CSV file first.";
    batchError.classList.remove("hidden");
    return;
  }

  batchAnalyzeBtn.disabled = true;
  batchBtnText.textContent = "Analyzing...";

  try {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("/api/batch-analyze", { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) {
      batchError.textContent = data.error;
      batchError.classList.remove("hidden");
      return;
    }

    batchSummary.textContent = `${data.total} entries analyzed`
      + (data.truncated ? " (file had more rows — capped at 150)" : "");

    batchTableBody.innerHTML = "";
    data.results.forEach((r) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${r.text_preview.replace(/</g, "&lt;")}</td>
        <td><span class="badge ${toClassName(r.label)}">${r.label}</span></td>
        <td>${(r.confidence * 100).toFixed(1)}%</td>
        <td><span class="badge tier-${toClassName(r.severity)}">${r.severity}</span></td>
      `;
      batchTableBody.appendChild(row);
    });

    batchResults.classList.remove("hidden");
  } catch (err) {
    batchError.textContent = "Something went wrong: " + err.message;
    batchError.classList.remove("hidden");
  } finally {
    batchAnalyzeBtn.disabled = false;
    batchBtnText.textContent = "Analyze batch";
  }
});

async function loadDashboard() {
  try {
    const res = await fetch("/api/dashboard-data");
    const data = await res.json();
    if (data.error) {
      document.getElementById("dashboard-summary").textContent = data.error;
      return;
    }

    document.getElementById("dashboard-summary").textContent =
      `${data.total_posts.toLocaleString()} posts in the research archive`;

    // Read the current theme's muted text color so chart legends/ticks stay
    // readable in both light and dark mode instead of a hardcoded hex.
    const mutedColor = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim();

    const dist = data.sentiment_distribution;
    const palette = ["#2ec4b6", "#ff6f61", "#8a6fd6", "#ffd166", "#f4a261", "#6b4fa0", "#4ecdc4"];
    new Chart(document.getElementById("sentimentChart"), {
      type: "pie",
      data: {
        labels: Object.keys(dist),
        datasets: [{
          data: Object.values(dist),
          backgroundColor: palette.slice(0, Object.keys(dist).length),
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { position: "bottom", labels: { color: mutedColor, font: { family: "DM Sans", size: 11 }, padding: 14, usePointStyle: true } } },
      },
    });

    const topicLabel = data.topics && (data.topics.Depression ? "Depression" : Object.keys(data.topics)[0]);
    if (topicLabel && data.topics[topicLabel]) {
      const topic = data.topics[topicLabel][0];
      // Use LDA's real per-word probability weights when available; falls
      // back to a rank-based value for older topics.json files that don't
      // have "weights" yet (from before topic_model.py was updated).
      const chartValues = (topic.weights && topic.weights.length === topic.keywords.length)
        ? topic.weights
        : topic.keywords.map((_, i) => topic.keywords.length - i);
      new Chart(document.getElementById("topicChart"), {
        type: "bar",
        data: {
          labels: topic.keywords,
          datasets: [{ label: `Top '${topicLabel}' topic keywords`, data: chartValues, backgroundColor: "#ff6f61" }],
        },
        options: {
          indexAxis: "y",
          responsive: true,
          scales: { x: { display: false }, y: { grid: { display: false }, ticks: { color: mutedColor, font: { family: "DM Sans", size: 11 } } } },
          plugins: { legend: { display: false } },
        },
      });
    }
  } catch (err) {
    console.error(err);
  }
}

loadDashboard();