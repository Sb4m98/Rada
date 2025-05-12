// script.js

// Avatar URLs
const botAvatarUrl  = "https://i.ibb.co/dPQxgfx/robo.png";
const userAvatarUrl = "https://i.ibb.co/NW2NRnb/dxc-logo.png";

let lastEventId = null;
let selectedProfile = null;

// Bootstrap Toast utility
function showToast(message, variant = 'info') {
  const container = document.getElementById('toast-container') || createToastContainer();
  const toastEl = document.createElement('div');
  toastEl.className = `toast align-items-center text-bg-${variant} border-0 mb-2`;
  toastEl.setAttribute('role', 'alert');
  toastEl.setAttribute('aria-live', 'assertive');
  toastEl.setAttribute('aria-atomic', 'true');
  toastEl.innerHTML = `
    <div class="d-flex">
      <div class="toast-body">${message}</div>
      <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
    </div>`;
  container.appendChild(toastEl);
  new bootstrap.Toast(toastEl, { delay: 5000 }).show();
}

function createToastContainer() {
  const container = document.createElement('div');
  container.id = 'toast-container';
  container.style.position = 'fixed';
  container.style.top = '1rem';
  container.style.right = '1rem';
  container.style.zIndex = 1080;
  document.body.appendChild(container);
  return container;
}

function setButtonLoading(btn, loading=true, label='') {
  if (!btn) return;
  if (loading) {
    btn._origLabel = btn.textContent;
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> ${label}`;
  } else {
    btn.disabled = false;
    btn.textContent = btn._origLabel;
  }
}

const processStyles = [
  {
    selector: 'node',
    style: {
      'label': 'data(label)',
      'text-valign': 'center',
      'text-halign': 'center',
      'font-size': '14px',
      'width': '180px',
      'height': '60px',
      'shape': 'round-rectangle',
      'border-width': 2,
      'padding': '10px',
      'background-color': '#F0F0F0',    // grigio molto chiaro di base
      'border-color': '#888'
    }
  },
  {
    selector: '.beginEnd',
    style: {
      'background-color': '#8DD3C7',    // verde acqua pastello
      'border-color': '#4DAF4A'
    }
  },
  {
    selector: '.mainProcess',
    style: {
      'background-color': '#FFFFB3',    // giallo pastello
      'border-color': '#FFED6F'
    }
  },
  {
    selector: '.decision',
    style: {
      'background-color': '#BEBADA',    // lilla pastello
      'border-color': '#6B4C9A',
      'shape': 'diamond'
    }
  },
  {
    selector: '.criticalNode',
    style: {
      'background-color': '#FB8072',    // corallo pastello
      'border-color': '#D23F31'
    }
  },
  {
    selector: 'edge',
    style: {
      'label': 'data(label)',
      'curve-style': 'bezier',
      'target-arrow-shape': 'triangle',
      'arrow-scale': 1,
      'font-size': '12px',
      'line-color': '#AAA',
      'target-arrow-color': '#AAA'
    }
  }
];

let cy;

document.addEventListener('DOMContentLoaded', () => {
  let sessionId = null;
  cytoscape.use(cytoscapeDagre);

  // Sidebar toggle
  document.getElementById('toggle-sidebar')?.addEventListener('click', () => {
    document.getElementById('sidebar').classList.toggle('d-none');
  });

  // Show section utility
  function showSection(id) {
    document.querySelectorAll('.content-section').forEach(sec => sec.classList.toggle('active', sec.id === id));
    document.querySelectorAll('.nav-link').forEach(link => link.classList.toggle('active', link.id === id.replace('-content','-tab')));
  }

  // Nav links including new profile tab
  ['home','profile','upload','assistant','process'].forEach(name => {
    document.getElementById(`${name}-tab`)?.addEventListener('click', e => {
      e.preventDefault();
      showSection(`${name}-content`);
      if (name==='policy') document.getElementById('clause-text')?.focus();
    });
  });
  
// Profile selection buttons
document.getElementById('profile-content')?.addEventListener('click', async e => {
  const btn = e.target.closest('button[data-user]');
  if (!btn) return;

  const userId = btn.dataset.user;
  try {
    const res = await fetch(`/api/user/${userId}`);
    const profile = await res.json();
    if (!res.ok) throw new Error(profile.error || 'Error in loading the profile');

    // 1. Salva il profilo
    selectedProfile = profile;
    createSession();
    const lines = [];
    lines.push(`<strong>${profile.name} ${profile.surname}</strong> (ID: ${profile.user_id})`);
    lines.push(`Age: ${profile.age}`);
    lines.push(`Marital status: ${profile.marital_status}`);
    lines.push(`Children: ${profile.children}`);
    lines.push(`Income: €${profile.income.toLocaleString()}`);
    lines.push(`Profession: ${profile.profession}`);
    lines.push(`Product: ${profile.product.name} (${profile.product.premium_type} – €${profile.product.premium_amount})`);
    lines.push(`Duration: ${profile.product.duration_years} anni`);
    lines.push(`Objectives: ${profile.objectives.join(', ')}`);
    lines.push(`App usage: ${profile.engagement.app_usage_freq}`);
    lines.push(`Preference chat: ${profile.engagement.prefers_chat ? 'Sì' : 'No'}`);
    lines.push(`Risk profile: ${profile.experience.risk_profile}`);
    lines.push(`Preferred style: ${profile.preferred_style}`);
    lines.push(`Last request: ${profile.recent_inquiry_type}`);

    // 3. Inietta in pagina
    document.getElementById('selected-profile').innerHTML =
      `<ul class="list-unstyled mb-0"><li>${lines.join('</li><li>')}</li></ul>`;
    

  } catch (err) {
    console.error(err);
    showToast('Error in loading the profile: ' + err.message, 'danger');
  }
});

function appendAssistantMessage(text, isUser=false) {
  const container = document.getElementById('assistant-messages');
  const wrapper = document.createElement('div');
  wrapper.className = 'chat-message ' + (isUser ? 'user' : 'assistant');
  wrapper.innerText = text;
  container.appendChild(wrapper);
  container.scrollTop = container.scrollHeight;
}

document.getElementById('assistant-reset')?.addEventListener('click', () => {
  createSession();
  document.getElementById('assistant-messages').innerHTML = '';
  showToast('Session ressetted', 'info');
});
  function createSession() {
    fetch('/api/create_session', { method: 'POST' })
      .then(res => res.json())
      .then(data => {
        sessionId = data.session_id;
        // Aggiorna il badge in pagina (se ne hai uno)
        const badge = document.getElementById('session-id');
        if (badge) badge.textContent = sessionId;
  
        // Pulisci le UI delle varie sezioni
        const filesTable = document.getElementById('files-table-body');
        if (filesTable) filesTable.innerHTML = '';
  
        const chatContainer = document.getElementById('messages');
        if (chatContainer) chatContainer.innerHTML = '';
  
        const processJson = document.getElementById('process-json');
        if (processJson) processJson.value = '';
  
        const policyExplanation = document.getElementById('policy-explanation');
        if (policyExplanation) policyExplanation.innerHTML = '';
  
        const feedbackSection = document.getElementById('feedback-section');
        if (feedbackSection) feedbackSection.style.display = 'none';
  
        showToast('New session created: ' + sessionId, 'success');
      })
      .catch(err => {
        console.error('createSession error', err);
        showToast('Error in creating the session.', 'danger');
      });
  }
  document.getElementById('new-session-btn')?.addEventListener('click', createSession);

  // Upload PDF
  document.getElementById('upload-btn')?.addEventListener('click', async () => {
    const btn = document.getElementById('upload-btn');
    const file = document.getElementById('pdf-input').files[0];
    if (!file) return showToast('Swelect a PDF.', 'warning');
    if (!sessionId) return showToast('Create a session first...', 'warning');
  
    const fd = new FormData();
    fd.append('session_id', sessionId);
    fd.append('file', file);
  
    setButtonLoading(btn, true, 'Loading...');
  
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: fd });
      const data = await res.json();
      if (data.success) {
        updateFileList();
      } else {
        showToast(data.error || 'Loading error.', 'danger');
      }
    } catch (err) {
      showToast('Error network loading.', 'danger');
    } finally {
      setButtonLoading(btn, false);
    }
  });
  
  // Update file list
  function updateFileList() {
    fetch(`/api/files?session_id=${encodeURIComponent(sessionId)}`)
      .then(res => res.json())
      .then(data => {
        const tbody = document.getElementById('files-table-body');
        tbody.innerHTML = '';
        if (data.files?.length) {
          data.files.forEach(f => {
            const a = f.analysis || {};
            const topWords = a.word_freq ? a.word_freq.map(([w,c]) => `${w} (${c})`).join(', ') : '';
            tbody.innerHTML += `
              <tr>
                <td>${f.name}</td>
                <td>${a.word_count||0}</td>
                <td>${a.unique_words||0}</td>
                <td>${a.avg_word_length ? a.avg_word_length.toFixed(2) : 0}</td>
                <td>${a.vocabulary_density ? a.vocabulary_density.toFixed(2) : 0}</td>
                <td>${topWords}</td>
              </tr>`;
          });
        } else {
          tbody.innerHTML = '<tr><td colspan="6" class="text-center">Nessun file caricato.</td></tr>';
        }
      })
      .catch(() => showToast('Error in getting the file list.', 'danger'));
  }

  // Chat send
  function appendMessage(text, isUser, isSource=false) {
    const container = document.getElementById('messages');
    const msg = document.createElement('div');
    msg.className = 'chat-message ' + (isUser ? 'user' : 'assistant');
    const avatar = document.createElement('div'); avatar.className = 'avatar';
    const img = document.createElement('img'); img.src = isUser ? userAvatarUrl : botAvatarUrl;
    avatar.appendChild(img);

    const bubble = document.createElement('div'); bubble.className = 'message';
    if (isSource) bubble.classList.add('fst-italic', 'small');
    bubble.textContent = text;

    msg.appendChild(avatar);
    msg.appendChild(bubble);
    container.appendChild(msg);
    container.scrollTop = container.scrollHeight;
  }

  // Process graph rendering
  function renderProcessGraph(jsonData) {
    if (!jsonData?.fasi) return;
    cy?.destroy();
  
    const elements = [];
    let desc = '';
  
    jsonData.fasi.forEach(f => {
      elements.push({
        data: {
          id: f.nome,
          label: f.nome
        },
        classes: f.node_type || 'mainProcess'  
      });
    });
    
    jsonData.relazioni.forEach(r => {
      elements.push({ data: { id: `${r.da}-${r.a}`, source: r.da, target: r.a, label: r.condizione } });
    });
  
    document.getElementById('process-description').value = desc.trim();
  
    cy = cytoscape({
      container: document.getElementById('cy'),
      elements,
      style: processStyles,
      layout: { name: 'dagre', rankDir: 'TB', nodeDimensionsIncludeLabels: true }
    });
  // Zoom in
document.getElementById('zoom-in')?.addEventListener('click', () => {
  cy.zoom({ level: cy.zoom() * 1.2 });
});

// Zoom out
document.getElementById('zoom-out')?.addEventListener('click', () => {
  cy.zoom({ level: cy.zoom() * 0.8 });
});

// Reset view (fit all)
document.getElementById('reset-zoom')?.addEventListener('click', () => {
  cy.fit(50);       // 50px padding
  cy.center();      // centra il layout
});
    cy.fit(50);
  }

  document.getElementById('process-btn')?.addEventListener('click', async () => {
    const btn       = document.getElementById('process-btn');
    const descInput = document.getElementById('process-input');
    const query     = descInput.value.trim();
    const textarea  = document.getElementById('process-description');
  
    if (!query) {
      return showToast('Digit the process description.', 'warning');
    }
    if (!sessionId) {
      return showToast('Please, create a session.', 'warning');
    }
  
    setButtonLoading(btn, true, 'Generating…');
  
    try {
      const res = await fetch('/api/extract_process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, query })
      });
  
      // Verifica JSON e stato
      const json = await res.json();
      if (!res.ok) {
        throw new Error(json.error || JSON.stringify(json));
      }
  
      // 1. Renderizza il grafo
      if (!json.fasi || !json.relazioni) {
        throw new Error('Data proces format not vaild');
      }
      renderProcessGraph(json);
  
      // 2. Mostra la sezione dei processi
      showSection('process-content');
  
      // 3. Popola la descrizione degli step
      const lines = json.fasi.map(f =>
        `• ${f.nome}:\n` +
        `    - Description: ${f.descrizione || 'N/A'}\n` +
        `    - Prerequisites: ${f.prerequisiti || 'No'}\n` +
        `    - Effects: ${f.effetti || 'No'}\n` +
        `    - Timings: ${f.tempistiche || 'N/A'}`
      );
      textarea.value = lines.join('\n\n');
  
    } catch (err) {
      console.error(err);
      showToast('Error during the process extraction: ' + err.message, 'danger');
    } finally {
      setButtonLoading(btn, false);
    }
  });


// Ask Assistant
document.getElementById('assistant-send')?.addEventListener('click', async () => {
  const btn = document.getElementById('assistant-send');
  const q = document.getElementById('assistant-query').value.trim();

  if (!selectedProfile) return showToast('Pleases, select a profile.', 'warning');
  if (!q) return showToast('Enter a question.', 'warning');

  // 1) Append messaggio utente
  appendAssistantMessage(q, true);
  document.getElementById('assistant-query').value = '';
  setButtonLoading(btn, true, 'Loading...');

  try {
    const res = await fetch('/api/assist', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        query: q,
        profile: selectedProfile
      })
    });

    const text = await res.text();

    if (!res.ok) {
      console.error('Server error:', text);
      return showToast('Server error: ' + text, 'danger');
    }

    let j;
    try {
      j = JSON.parse(text);
    } catch {
      console.error('Invalid JSON:', text);
      return showToast('Response not valid.', 'danger');
    }

    // 2) Append risposta AI
    appendAssistantMessage(j.answer, false);

    // 3) Automatismo Process-driven
    const PROCESS_INTENTS = new Set(['AskClaimProcedure']);
    const THRESHOLD = 0.6;

    const tops = Array.isArray(j.topIntents) ? j.topIntents : [];
    const candidate = tops
      .filter(i => PROCESS_INTENTS.has(i.category) && i.confidenceScore >= THRESHOLD)
      .sort((a, b) => b.confidenceScore - a.confidenceScore)[0];

    if (candidate) {
      const prev = document.getElementById('assistant-next-step');
      if (prev) prev.remove();

      const wrapper = document.createElement('div');
      wrapper.id = 'assistant-next-step';
      wrapper.className = 'mt-2 p-2 bg-info text-white rounded';
      wrapper.innerHTML = `
        I detected the intent "<strong>${candidate.category}</strong>" (confidence ${(candidate.confidenceScore * 100).toFixed(0)}%).<br>
        Want to see the process diagram?
        <button id="show-claim-process" class="btn btn-sm btn-light ms-2">Yes</button>
         <button id="dismiss-process" class="btn btn-sm btn-outline-light ms-2">No</button>
      `;

      document.getElementById('assistant-content').appendChild(wrapper);

      // Animazione ingresso
      wrapper.style.opacity = 0;
      setTimeout(() => wrapper.style.opacity = 1, 10);

      // Gestione click Sì
      document.getElementById('show-claim-process').addEventListener('click', () => {
        wrapper.remove();
        showSection('process-content');
        document.getElementById('process-input').value = `Process: ${candidate.category}`;
        document.getElementById('process-btn').click();
      });

      // Gestione click No
      document.getElementById('dismiss-process').addEventListener('click', () => {
        wrapper.style.opacity = 0;
        setTimeout(() => wrapper.remove(), 300);
        showToast('Process suggestion ignored', 'info');
      });

      // Auto-rimozione dopo 30 secondi
      const timeoutId = setTimeout(() => {
        if (document.body.contains(wrapper)) {
          wrapper.style.opacity = 0;
          setTimeout(() => wrapper.remove(), 300);
          showToast('Process suggestion ignored', 'info');
        }
      }, 30000);

      // Pulisci timeout se componente rimosso
      wrapper.addEventListener('dismiss', () => clearTimeout(timeoutId));
    }

  } catch (err) {
    console.error(err);
    showToast('Network error.', 'danger');
  } finally {
    setButtonLoading(btn, false);
  }
});
  
  // Feedback reward function
  window.sendReward = function(score) {
    if (!lastEventId) {
      return showToast('No explanation to evaluate.', 'warning');
    }
    fetch('/api/personalizer/reward', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ eventId: lastEventId, reward: score })
    })
    .then(r => r.json())
    .then(d => {
      if (d.success) {
        showToast('Thanks for the feedback!', 'success');
        document.getElementById('feedback-section').style.display = 'none';
      } else {
        showToast('Error in sending the feedback.', 'danger');
      }
    })
    .catch(() => showToast('Network error during feedback.', 'danger'));
  };

  // Initialize session
  createSession();
  // Default show home
  showSection('home-content');
});
