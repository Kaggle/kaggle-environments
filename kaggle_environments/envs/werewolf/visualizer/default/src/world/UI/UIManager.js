import { marked } from 'marked';
import { applyTranscriptOverrides } from '../../utils/transcriptUtils.js';

export class UIManager {
  constructor(CSS2DObject, width, height, camera, parent) {
    this.CSS2DObject = CSS2DObject;
    this.width = width;
    this.height = height;
    this.camera = camera;
    this.parent = parent;

    // Cinematic Subtitle Container
    this.subtitleContainer = document.createElement('div');
    this.subtitleContainer.className = 'cinematic-subtitle-container';
    // Ensure parent exists, otherwise fallback to body
    (this.parent || document.body).appendChild(this.subtitleContainer);

    this.subtitleTimeout = null;
    this.isExpanded = true;
    this.subtitleContainer.classList.add('expanded');

    // Add click listener for expansion toggle
    this.subtitleContainer.addEventListener('click', () => this.toggleExpansion());

    // Add global key listener for shortcuts
    window.addEventListener('keydown', (e) => {
      const key = e.key.toLowerCase();
      if (key === 'e') {
        this.toggleExpansion();
      } else if (key === 'r') {
        const reasoningToggle = document.getElementById('global-reasoning-toggle');
        if (reasoningToggle) {
          reasoningToggle.click();
        }
      }
    });
  }

  toggleExpansion() {
    this.isExpanded = !this.isExpanded;
    if (this.isExpanded) {
      this.subtitleContainer.classList.add('expanded');
    } else {
      this.subtitleContainer.classList.remove('expanded');
    }
    console.debug(`[Werewolf] Subtitle expanded: ${this.isExpanded}`);
  }

  createPlayerUI(name, displayName, imageUrl, focusCallback) {
    const container = document.createElement('div');
    container.className = 'player-ui-container';

    const playerInfoCard = document.createElement('div');
    playerInfoCard.className = 'player-info-card centered-component';

    const textDetails = document.createElement('div');
    textDetails.className = 'player-text-details';

    const nameRow = document.createElement('div');
    nameRow.className = 'player-name-row';

    const img = document.createElement('img');
    img.className = 'player-avatar-3d';
    img.src = imageUrl;
    img.onerror = function () {
      if (window.handleThumbnailError) {
        window.handleThumbnailError(this);
      }
    };
    nameRow.appendChild(img);

    const nameText = document.createElement('div');
    nameText.className = 'player-name-3d';
    nameText.textContent = displayName || name;
    nameRow.appendChild(nameText);

    textDetails.appendChild(nameRow);

    const roleText = document.createElement('div');
    roleText.className = 'player-role-3d';
    roleText.textContent = 'Role: Unknown';
    textDetails.appendChild(roleText);

    playerInfoCard.appendChild(textDetails);

    // Removed legacy chatMessageCard creation

    container.appendChild(playerInfoCard);

    playerInfoCard.onclick = (e) => {
      // Simplified click handler - just focus
      if (focusCallback) {
        focusCallback(name);
      }
    };

    const label = new this.CSS2DObject(container);
    return label;
  }

  displayPlayerBubble(playerUI, message, reasoning, timestamp, isAction = false) {
    let speakerName = '';
    if (playerUI && playerUI.element) {
      const nameEl = playerUI.element.querySelector('.player-name-3d');
      if (nameEl) speakerName = nameEl.textContent;
    }

    // Redirect to cinematic subtitle
    this.displaySubtitle(message, reasoning, speakerName, isAction);

    // Visual feedback on the player's UI (only for chat, not actions)
    if (!isAction && playerUI && playerUI.element) {
      playerUI.element.classList.add('chat-active');
    }
  }

  displaySubtitle(message, reasoning, speakerName = '', isAction = false) {
    // No timeout - subtitle persists until cleared manually (or replaced)

    // Construct HTML structure
    let innerContent = '';
    if (speakerName) {
      innerContent += `<span class="subtitle-speaker">${speakerName}:</span> `;
    }
    // 0. Apply Transcript Overrides
    const textToProcess = applyTranscriptOverrides(message || '');

    // 1. Get Replacer
    const replacer = window.werewolfGamePlayer && (window.werewolfGamePlayer.htmlPlayerIdReplacer || window.werewolfGamePlayer.playerIdReplacer);
    let finalHtml = '';

    if (replacer && replacer.replaceToPlaceholders && replacer.expandPlaceholders) {
      // Phase 1: Text -> Placeholders
      const textWithPlaceholders = replacer.replaceToPlaceholders(textToProcess);

      // Phase 2: Markdown Parse
      const htmlWithPlaceholders = marked.parse(textWithPlaceholders);

      // Phase 3: Placeholders -> HTML Capsules
      finalHtml = replacer.expandPlaceholders(htmlWithPlaceholders);
    } else {
      // Fallback (unsafe) behavior
      const parsedMessage = marked.parse(textToProcess);
      finalHtml = replacer ? replacer(parsedMessage) : parsedMessage;
    }

    innerContent += `<div class="cinematic-subtitle-text">${finalHtml}</div>`;

    if (reasoning) {
      let finalReasoningHtml = '';
      const replacer = window.werewolfGamePlayer && (window.werewolfGamePlayer.htmlPlayerIdReplacer || window.werewolfGamePlayer.playerIdReplacer);

      if (replacer && replacer.replaceToPlaceholders && replacer.expandPlaceholders) {
        const rText = replacer.replaceToPlaceholders(reasoning);
        const rHtml = marked.parse(rText);
        finalReasoningHtml = replacer.expandPlaceholders(rHtml);
      } else {
        const pR = marked.parse(reasoning);
        finalReasoningHtml = replacer ? replacer(pR) : pR;
      }
      innerContent += `<div class="cinematic-subtitle-reasoning">${finalReasoningHtml}</div>`;
    }

    // 1. Check if we already have the wrapper, if so preserve it and just update content
    let wrapper = this.subtitleContainer.querySelector('.cinematic-subtitle-content-wrapper');
    if (!wrapper) {
      // First time initialization of layout
      this.subtitleContainer.innerHTML = `
            <div class="cinematic-subtitle-content-wrapper"></div>
            <div class="subtitle-controls">
                <button class="subtitle-scroll-btn up">▲</button>
                <div class="scroll-track">
                    <div class="scroll-thumb"></div>
                </div>
                <button class="subtitle-scroll-btn down">▼</button>
            </div>
        `;
      wrapper = this.subtitleContainer.querySelector('.cinematic-subtitle-content-wrapper');

      // Setup Button Listeners
      const upBtn = this.subtitleContainer.querySelector('.subtitle-scroll-btn.up');
      const downBtn = this.subtitleContainer.querySelector('.subtitle-scroll-btn.down');

      const scrollStep = 20;

      if (upBtn) {
        upBtn.onclick = (e) => {
          e.stopPropagation();
          if (wrapper) wrapper.scrollTop -= scrollStep;
        };
      }

      if (downBtn) {
        downBtn.onclick = (e) => {
          e.stopPropagation();
          if (wrapper) wrapper.scrollTop += scrollStep;
        };
      }
    }

    // Update Content
    if (wrapper) {
      wrapper.innerHTML = innerContent;
      wrapper.scrollTop = 0; // Reset scroll on new message
      this.lastMessageTime = performance.now(); // Record time for scroll delay
    }

    // Show Container
    this.subtitleContainer.classList.add('visible');

    // Sync visibility state
    if (window.werewolfGamePlayer && window.werewolfGamePlayer.isReasoningMode) {
      this.subtitleContainer.classList.add('show-reasoning');
    } else {
      this.subtitleContainer.classList.remove('show-reasoning');
    }

    // Ensure we aren't in moderator mode
    this.subtitleContainer.classList.remove('moderator-mode');

    // Add action mode if applicable (centers text)
    if (isAction) {
      this.subtitleContainer.classList.add('action-mode');
    } else {
      this.subtitleContainer.classList.remove('action-mode');
    }

    // Show/Hide controls based on content overflow (checked in update loop or next frame)
    // For now we set them visible if we expect overflow, but dynamic check is better.
    // We'll rely on the update loop or just show them always if content is long?
    // We'll check in update()
  }

  displayModeratorAnnouncement(message, isAction = false) {
    // Reuse displaySubtitle logic but add moderator class? 
    // Or reimplement structure since moderator mode has specific styling.
    // Let's reimplement structure for consistent scrolling.

    // Parse Markdown
    let innerContent = '';

    if (isAction) {
      // Actions are pre-formatted HTML (capsules). Do NOT markdown parse.
      // Also assume player IDs are already replaced by caller (legacy-renderer).
      const textToDisplay = applyTranscriptOverrides(message || '');
      innerContent = `<span class="subtitle-speaker">Moderator</span><div class="cinematic-subtitle-text">${textToDisplay}</div>`;
    } else {
      // System / Narrator messages.
      const textToProcess = applyTranscriptOverrides(message || '');
      // Use Phased Replacement
      const replacer = window.werewolfGamePlayer && (window.werewolfGamePlayer.htmlPlayerIdReplacer || window.werewolfGamePlayer.playerIdReplacer);
      let finalHtml = '';

      if (replacer && replacer.replaceToPlaceholders && replacer.expandPlaceholders) {
        const t = replacer.replaceToPlaceholders(textToProcess);
        const h = marked.parse(t);
        finalHtml = replacer.expandPlaceholders(h);
      } else {
        const p = marked.parse(textToProcess);
        finalHtml = replacer ? replacer(p) : p;
      }

      innerContent = `<span class="subtitle-speaker">Moderator</span><div class="cinematic-subtitle-text">${finalHtml}</div>`;
    }

    let wrapper = this.subtitleContainer.querySelector('.cinematic-subtitle-content-wrapper');
    if (!wrapper) {
      this.subtitleContainer.innerHTML = `
            <div class="cinematic-subtitle-content-wrapper"></div>
            <div class="subtitle-controls">
                <button class="subtitle-scroll-btn up">▲</button>
                <div class="scroll-track">
                    <div class="scroll-thumb"></div>
                </div>
                <button class="subtitle-scroll-btn down">▼</button>
            </div>
        `;
      wrapper = this.subtitleContainer.querySelector('.cinematic-subtitle-content-wrapper');

      // Setup Button Listeners (duplicated logic, should refactor if strict DRY needed but fine here)
      const upBtn = this.subtitleContainer.querySelector('.subtitle-scroll-btn.up');
      const downBtn = this.subtitleContainer.querySelector('.subtitle-scroll-btn.down');
      const scrollStep = 20;
      if (upBtn) upBtn.onclick = (e) => { e.stopPropagation(); if (wrapper) wrapper.scrollTop -= scrollStep; };
      if (downBtn) downBtn.onclick = (e) => { e.stopPropagation(); if (wrapper) wrapper.scrollTop += scrollStep; };
    }

    if (wrapper) {
      wrapper.innerHTML = innerContent;
      wrapper.scrollTop = 0;
      this.lastMessageTime = performance.now(); // Record time for scroll delay
    }

    this.subtitleContainer.classList.add('visible');
    this.subtitleContainer.classList.add('moderator-mode'); // Adds transparency/blur/shape

    // Toggle action mode for alignment
    if (isAction) {
      this.subtitleContainer.classList.add('action-mode');
    } else {
      this.subtitleContainer.classList.remove('action-mode');
    }

    // Sync visibility state
    if (window.werewolfGamePlayer && window.werewolfGamePlayer.isReasoningMode) {
      this.subtitleContainer.classList.add('show-reasoning');
    } else {
      this.subtitleContainer.classList.remove('show-reasoning');
    }
  }

  clearSubtitle() {
    this.subtitleContainer.classList.remove('visible');
  }

  update(delta) {
    if (!this.subtitleContainer.classList.contains('visible')) return;

    const wrapper = this.subtitleContainer.querySelector('.cinematic-subtitle-content-wrapper');
    const controls = this.subtitleContainer.querySelector('.subtitle-controls');

    if (wrapper && controls) {
      // Check if scrollable
      if (wrapper.scrollHeight > wrapper.clientHeight) {
        controls.classList.add('visible');

        // Auto-Scroll Logic
        const audioState = window.kaggleWerewolf;
        const isAudioPlaying = audioState && audioState.isAudioEnabled && !audioState.isPaused;

        // Check if steps are updating
        const lastUpdate = (window.werewolfThreeJs && window.werewolfThreeJs.lastStepUpdateTime) || 0;
        const isStepping = (performance.now() - lastUpdate) < 1000;

        const timeSinceMessage = performance.now() - (this.lastMessageTime || 0);

        if (timeSinceMessage > 1500 && (isAudioPlaying || isStepping)) {
          // Slow auto scroll
          const rate = (audioState && audioState.playbackRate) || 1.0;
          const speed = 10 * rate; // Pixels per second

          // Only scroll if not at bottom?
          if (Math.ceil(wrapper.scrollTop + wrapper.clientHeight) < wrapper.scrollHeight) {
            wrapper.scrollTop += speed * delta;
          }
        }

        // Update Scroll Thumb Position/Size
        const track = this.subtitleContainer.querySelector('.scroll-track');
        const thumb = this.subtitleContainer.querySelector('.scroll-thumb');
        if (track && thumb) {
          const trackHeight = track.clientHeight;
          const thumbHeight = 6; // Fixed dot size
          const maxScroll = wrapper.scrollHeight - wrapper.clientHeight;
          const scrollRatio = maxScroll > 0 ? wrapper.scrollTop / maxScroll : 0;
          const thumbMaxTop = trackHeight - thumbHeight;
          const thumbTop = scrollRatio * thumbMaxTop;

          thumb.style.top = `${thumbTop}px`;
        }
      } else {
        controls.classList.remove('visible');
      }
    }
  }

  updateDynamicUI(playerObjects) {
    // Logic for arrows was commented out in legacy code, preserving stub
  }
}
