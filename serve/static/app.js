/**
 * Lulogooli Translate — Interactive Frontend Client
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const sourceInput = document.getElementById('sourceInput');
  const targetOutput = document.getElementById('targetOutput');
  const sourceLangSelect = document.getElementById('sourceLangSelect');
  const translateBtn = document.getElementById('translateBtn');
  const clearSourceBtn = document.getElementById('clearSourceBtn');
  const pasteBtn = document.getElementById('pasteBtn');
  const copyTargetBtn = document.getElementById('copyTargetBtn');
  const sourceCharCount = document.getElementById('sourceCharCount');
  const targetWordCount = document.getElementById('targetWordCount');
  const latencyDisplay = document.getElementById('latencyDisplay');
  const confidenceTag = document.getElementById('confidenceTag');
  const statusBadge = document.getElementById('statusBadge');
  const statusText = document.getElementById('statusText');
  const presetChips = document.querySelectorAll('.chip-btn');
  const toastContainer = document.getElementById('toastContainer');

  // Modals
  const suggestCorrectionBtn = document.getElementById('suggestCorrectionBtn');
  const feedbackModal = document.getElementById('feedbackModal');
  const closeFeedbackModal = document.getElementById('closeFeedbackModal');
  const cancelFeedbackBtn = document.getElementById('cancelFeedbackBtn');
  const feedbackForm = document.getElementById('feedbackForm');
  const feedbackSource = document.getElementById('feedbackSource');
  const feedbackOriginal = document.getElementById('feedbackOriginal');
  const feedbackCorrection = document.getElementById('feedbackCorrection');
  const feedbackNotes = document.getElementById('feedbackNotes');

  const metricsModalBtn = document.getElementById('metricsModalBtn');
  const benchmarksModal = document.getElementById('benchmarksModal');
  const closeBenchmarksModal = document.getElementById('closeBenchmarksModal');
  const closeBenchmarksBtn = document.getElementById('closeBenchmarksBtn');

  let debounceTimer = null;

  // -------------------------------------------------------------------------
  // Health Check / Status
  // -------------------------------------------------------------------------
  async function checkHealth() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        if (data.model_loaded && !data.mock_mode) {
          statusText.textContent = `Neural Model (${data.device.toUpperCase()})`;
          statusBadge.style.color = 'var(--accent-emerald)';
        } else {
          statusText.textContent = 'Smart Demo Mode';
          statusBadge.style.color = 'var(--accent-cyan)';
        }
      }
    } catch (e) {
      statusText.textContent = 'Offline';
      statusBadge.style.color = 'var(--accent-amber)';
    }
  }
  checkHealth();

  // -------------------------------------------------------------------------
  // Translation Core
  // -------------------------------------------------------------------------
  async function performTranslation() {
    const text = sourceInput.value.trim();
    if (!text) {
      targetOutput.value = '';
      latencyDisplay.textContent = '— ms';
      targetWordCount.textContent = '0 words';
      confidenceTag.style.display = 'none';
      return;
    }

    targetOutput.classList.add('shimmer');
    targetOutput.placeholder = 'Translating into Maragoli...';

    try {
      const payload = {
        text: text,
        src_lang: sourceLangSelect.value,
        tgt_lang: 'rag_Latn',
        num_beams: 4,
        no_repeat_ngram_size: 3,
        repetition_penalty: 1.2,
        max_new_tokens: 128
      };

      const response = await fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`API error ${response.status}`);
      }

      const result = await response.json();
      targetOutput.value = result.translated_text;
      latencyDisplay.textContent = `${result.latency_ms} ms`;
      
      const words = result.translated_text ? result.translated_text.trim().split(/\s+/).length : 0;
      targetWordCount.textContent = `${words} words`;

      confidenceTag.style.display = 'inline-block';
      confidenceTag.textContent = result.confidence_label || 'Predicted';

    } catch (err) {
      console.error(err);
      targetOutput.value = 'Translation service encountered an error. Please try again.';
      showToast('Translation request failed', 'error');
    } finally {
      targetOutput.classList.remove('shimmer');
    }
  }

  // Auto-translate on typing with debounce
  sourceInput.addEventListener('input', () => {
    const len = sourceInput.value.length;
    sourceCharCount.textContent = `${len} / 500 characters`;

    clearTimeout(debounceTimer);
    if (len > 3) {
      debounceTimer = setTimeout(performTranslation, 450);
    } else if (len === 0) {
      targetOutput.value = '';
      targetWordCount.textContent = '0 words';
      latencyDisplay.textContent = '— ms';
    }
  });

  translateBtn.addEventListener('click', performTranslation);

  // -------------------------------------------------------------------------
  // Preset Chips & Language Switching
  // -------------------------------------------------------------------------
  const PRESETS_DATA = {
    eng_Latn: [
      { tag: "Health", text: "Report suspected health cases to the nearest facility." },
      { tag: "Sanitation", text: "Wash hands with soap and clean running water." },
      { tag: "Road Safety", text: "Observe traffic rules to prevent road accidents." },
      { tag: "Children", text: "Children must be immunized against diseases at six months." },
      { tag: "Civil", text: "All citizens have a right to clean drinking water." }
    ],
    swh_Latn: [
      { tag: "Afya", text: "Ripoti wagonjwa wanaoshukiwa katika kituo cha afya kilicho karibu." },
      { tag: "Usafi", text: "Osha mikono yako kwa sabuni na maji safi yanayotiririka." },
      { tag: "Usalama", text: "Fuata sheria za barabarani kuzuia ajali." },
      { tag: "Watoto", text: "Watoto lazima wapewe chanjo ya kuzuia magonjwa wakiwa na miezi sita." },
      { tag: "Haki za Raia", text: "Wananchi wote wana haki ya kupata maji safi ya kunywa." },
      { tag: "Simulizi", text: "Ulikoenda ulikulia nini?" }
    ]
  };

  const presetChipsList = document.getElementById('presetChipsList');

  function renderPresets(lang) {
    const items = PRESETS_DATA[lang] || PRESETS_DATA.eng_Latn;
    presetChipsList.innerHTML = '';
    items.forEach(item => {
      const btn = document.createElement('button');
      btn.className = 'chip-btn';
      btn.setAttribute('data-text', item.text);
      btn.innerHTML = `<span class="chip-tag">${item.tag}</span> ${item.text}`;
      btn.addEventListener('click', () => {
        sourceInput.value = item.text;
        sourceCharCount.textContent = `${item.text.length} / 500 characters`;
        performTranslation();
      });
      presetChipsList.appendChild(btn);
    });
  }

  sourceLangSelect.addEventListener('change', () => {
    const selected = sourceLangSelect.value;
    renderPresets(selected);
    if (sourceInput.value.trim().length > 0) {
      performTranslation();
    }
  });

  renderPresets(sourceLangSelect.value);

  // -------------------------------------------------------------------------
  // Tools & Actions
  // -------------------------------------------------------------------------
  clearSourceBtn.addEventListener('click', () => {
    sourceInput.value = '';
    targetOutput.value = '';
    sourceCharCount.textContent = '0 / 500 characters';
    targetWordCount.textContent = '0 words';
    latencyDisplay.textContent = '— ms';
    confidenceTag.style.display = 'none';
    sourceInput.focus();
  });

  pasteBtn.addEventListener('click', async () => {
    try {
      const clip = await navigator.clipboard.readText();
      if (clip) {
        sourceInput.value = clip;
        sourceCharCount.textContent = `${clip.length} / 500 characters`;
        performTranslation();
        showToast('Pasted from clipboard');
      }
    } catch (e) {
      showToast('Clipboard access denied or unavailable', 'error');
    }
  });

  copyTargetBtn.addEventListener('click', async () => {
    const text = targetOutput.value.trim();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      copyTargetBtn.classList.add('copied');
      showToast('Copied Maragoli translation!');
      setTimeout(() => copyTargetBtn.classList.remove('copied'), 2000);
    } catch (e) {
      showToast('Could not copy to clipboard', 'error');
    }
  });

  // -------------------------------------------------------------------------
  // Community Feedback Modal
  // -------------------------------------------------------------------------
  suggestCorrectionBtn.addEventListener('click', () => {
    const src = sourceInput.value.trim();
    const tgt = targetOutput.value.trim();
    if (!src) {
      showToast('Enter a sentence to translate first', 'error');
      return;
    }
    feedbackSource.value = src;
    feedbackOriginal.value = tgt;
    feedbackCorrection.value = tgt;
    feedbackModal.classList.add('active');
    feedbackCorrection.focus();
  });

  function closeFeedback() {
    feedbackModal.classList.remove('active');
    feedbackCorrection.value = '';
    feedbackNotes.value = '';
  }

  closeFeedbackModal.addEventListener('click', closeFeedback);
  cancelFeedbackBtn.addEventListener('click', closeFeedback);

  feedbackForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const correction = feedbackCorrection.value.trim();
    if (!correction) return;

    try {
      const payload = {
        source_text: feedbackSource.value,
        translated_text: feedbackOriginal.value,
        corrected_text: correction,
        notes: feedbackNotes.value.trim() || null
      };

      const res = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        showToast('Thank you! Correction logged for model retraining.');
        closeFeedback();
      } else {
        showToast('Failed to save correction', 'error');
      }
    } catch (err) {
      showToast('Error recording correction', 'error');
    }
  });

  // -------------------------------------------------------------------------
  // Benchmarks Modal
  // -------------------------------------------------------------------------
  metricsModalBtn.addEventListener('click', () => {
    benchmarksModal.classList.add('active');
  });

  function closeBenchmarks() {
    benchmarksModal.classList.remove('active');
  }

  closeBenchmarksModal.addEventListener('click', closeBenchmarks);
  closeBenchmarksBtn.addEventListener('click', closeBenchmarks);

  // Close modals on escape or backdrop click
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeFeedback();
      closeBenchmarks();
    }
  });

  [feedbackModal, benchmarksModal].forEach(modal => {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        modal.classList.remove('active');
      }
    });
  });

  // -------------------------------------------------------------------------
  // Toast Helper
  // -------------------------------------------------------------------------
  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = 'toast';
    const icon = type === 'error' ? '⚠️' : '✓';
    toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.25s ease';
      setTimeout(() => toast.remove(), 250);
    }, 2800);
  }
});
