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
  }

  createPlayerUI(name, displayName, imageUrl, focusCallback) {
    const container = document.createElement('div');
    container.className = 'player-ui-container';

    const playerInfoCard = document.createElement('div');
    playerInfoCard.className = 'player-info-card centered-component';

    const img = document.createElement('img');
    img.className = 'player-avatar-3d';
    img.src = imageUrl;
    playerInfoCard.appendChild(img);

    const textDetails = document.createElement('div');
    textDetails.className = 'player-text-details';

    const nameText = document.createElement('div');
    nameText.className = 'player-name-3d';
    nameText.textContent = displayName || name;
    textDetails.appendChild(nameText);

    // Removed player ID display
    // const playerIdText = document.createElement('div');
    // playerIdText.className = 'player-id-3d';
    // playerIdText.textContent = name;
    // textDetails.appendChild(playerIdText);

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

  displayPlayerBubble(playerUI, message, reasoning, timestamp) {
    let speakerName = '';
    if (playerUI && playerUI.element) {
        const nameEl = playerUI.element.querySelector('.player-name-3d');
        if (nameEl) speakerName = nameEl.textContent;
    }

    // Redirect to cinematic subtitle
    this.displaySubtitle(message, reasoning, speakerName);

    // Visual feedback on the player's UI (optional, e.g., highlight the nameplate)
    if (playerUI && playerUI.element) {
        playerUI.element.classList.add('chat-active');
    }
  }

  displaySubtitle(message, reasoning, speakerName = '') {
    // No timeout - subtitle persists until cleared manually (or replaced)
    
    let content = '';
    if (speakerName) {
        content += `<span class="subtitle-speaker">${speakerName}:</span> `;
    }
    content += `<div class="cinematic-subtitle-text">${message}</div>`;

    if (reasoning) {
        content += `<div class="cinematic-subtitle-reasoning">${reasoning}</div>`;
    }

    this.subtitleContainer.innerHTML = content;
    this.subtitleContainer.classList.add('visible');
    
    // Sync visibility state
    if (window.werewolfGamePlayer && window.werewolfGamePlayer.isReasoningMode) {
        this.subtitleContainer.classList.add('show-reasoning');
    } else {
        this.subtitleContainer.classList.remove('show-reasoning');
    }
    
    // Ensure we aren't in moderator mode
    this.subtitleContainer.classList.remove('moderator-mode');
  }

  displayModeratorAnnouncement(message) {
      let content = `<span class="subtitle-speaker">Moderator</span><div class="cinematic-subtitle-text">${message}</div>`;
      
      this.subtitleContainer.innerHTML = content;
      this.subtitleContainer.classList.add('visible');
      this.subtitleContainer.classList.add('moderator-mode');

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

  updateDynamicUI(playerObjects) {
      // Logic for arrows was commented out in legacy code, preserving stub
  }
}
