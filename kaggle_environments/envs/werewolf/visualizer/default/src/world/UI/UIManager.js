export class UIManager {
  constructor(CSS2DObject, width, height, camera, parent) {
    this.CSS2DObject = CSS2DObject;
    this.width = width;
    this.height = height;
    this.camera = camera;
    this.parent = parent;
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

    const playerIdText = document.createElement('div');
    playerIdText.className = 'player-id-3d';
    playerIdText.textContent = name;
    textDetails.appendChild(playerIdText);

    const roleText = document.createElement('div');
    roleText.className = 'player-role-3d';
    roleText.textContent = 'Role: Unknown';
    textDetails.appendChild(roleText);

    playerInfoCard.appendChild(textDetails);

    const chatMessageCard = document.createElement('div');
    chatMessageCard.className = 'chat-message-card';
    chatMessageCard.innerHTML = `
              <div class="bubble-message"></div>
              <div class="bubble-reasoning"></div>
          `;

    playerInfoCard.appendChild(chatMessageCard);
    container.appendChild(playerInfoCard);

    playerInfoCard.onclick = (e) => {
      const playerContainer = e.target.closest('.player-ui-container');
      if (e.target.closest('.chat-message-card') && playerContainer) {
        const reasoningEl = playerContainer.querySelector('.bubble-reasoning');
        if (reasoningEl && (reasoningEl.innerHTML || reasoningEl.textContent)) {
          e.stopPropagation();
          playerContainer.classList.toggle('show-reasoning');
        } else {
          e.stopPropagation();
        }
      } else {
        if (focusCallback) {
            focusCallback(name);
        }
      }
    };

    const label = new this.CSS2DObject(container);
    return label;
  }

  displayPlayerBubble(playerUI, message, reasoning, timestamp) {
    if (!playerUI || !playerUI.element) return;

    const uiElement = playerUI.element;
    const messageEl = uiElement.querySelector('.bubble-message');
    const reasoningEl = uiElement.querySelector('.bubble-reasoning');

    if (messageEl) messageEl.innerHTML = message;
    if (reasoningEl) {
        reasoningEl.innerHTML = reasoning ? reasoning : '';
    }

    if (window.werewolfGamePlayer.isReasoningMode === undefined) {
      window.werewolfGamePlayer.isReasoningMode = false;
    }
    const isGlobalReasoningOn = window.werewolfGamePlayer.isReasoningMode;

    if (isGlobalReasoningOn && reasoning) {
      uiElement.classList.add('show-reasoning');
    } else {
      uiElement.classList.remove('show-reasoning');
    }

    uiElement.classList.add('chat-active');
  }

  updateDynamicUI(playerObjects) {
      // Logic for arrows was commented out in legacy code, preserving stub
  }
}
