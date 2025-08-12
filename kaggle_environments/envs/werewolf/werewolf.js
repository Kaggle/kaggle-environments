function renderer({
  environment,
  step,
  parent,
  height = 700, // Default height
  width = 1100, // Default width
}) {
  // --- THREE.js Scene Setup (Singleton Pattern) ---
  if (!window.werewolfThreeJs) {
    window.werewolfThreeJs = {
      initialized: false,
      scene: null,
      camera: null,
      renderer: null,
      model: null,
      ambientLight: null,
      directionalLight: null,
    };
  }
  const threeState = window.werewolfThreeJs;

  function initThreeJs() {
    if (threeState.initialized) {
        if (threeState.renderer && parent && !parent.contains(threeState.renderer.domElement)) {
            parent.appendChild(threeState.renderer.domElement);
        }
        return;
    }

    if (typeof THREE === 'undefined' || (typeof THREE.GLTFLoader === 'undefined')) {
      const scriptId = 'threejs-script';
      if (document.getElementById(scriptId)) return; // Script already requested

      const script = document.createElement('script');
      script.id = scriptId;
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
      script.onload = () => {
        const loaderScript = document.createElement('script');
        loaderScript.id = 'gltf-loader-script';
        loaderScript.src = 'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js';
        loaderScript.onload = setupScene;
        document.head.appendChild(loaderScript);
      };
      document.head.appendChild(script);
    } else {
      setupScene();
    }
  }

  function setupScene() {
    if (threeState.initialized) return;

    threeState.scene = new THREE.Scene();
    threeState.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    threeState.renderer = new THREE.WebGLRenderer({ antialias: true }); // No alpha needed
    threeState.renderer.setSize(width, height);

    const threeCanvas = threeState.renderer.domElement;
    threeCanvas.style.position = 'absolute';
    threeCanvas.style.top = '0';
    threeCanvas.style.left = '0';
    threeCanvas.style.zIndex = '0'; // Behind UI
    parent.appendChild(threeCanvas);

    threeState.ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
    threeState.scene.add(threeState.ambientLight);
    threeState.directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
    threeState.directionalLight.position.set(5, 10, 7.5);
    threeState.scene.add(threeState.directionalLight);

    const loader = new THREE.GLTFLoader();
    loader.load(
      'assets/low_poly_medieval_windmill/scene.gltf',
      function (gltf) {
        threeState.model = gltf.scene;
        threeState.model.scale.set(0.5, 0.5, 0.5); // Adjust scale for the new model
        threeState.model.position.set(0, -2, 0); // Adjust position
        threeState.scene.add(threeState.model);
      },
      undefined,
      function(error) {
        console.error('An error occurred loading the model:', error);
      }
    );

    threeState.camera.position.z = 5;
    threeState.initialized = true;
    animate();
  }

  function animate() {
    if (!threeState.initialized) return;
    requestAnimationFrame(animate);
    if (threeState.model) {
      threeState.model.rotation.y += 0.005;
    }
    threeState.renderer.render(threeState.scene, threeState.camera);
  }

  function updateBackground(isNight) {
      if (!threeState.initialized) return;
      if (isNight) {
          threeState.scene.background = new THREE.Color(0x2c3e50); // Night color from CSS var
          threeState.ambientLight.intensity = 0.5;
          threeState.directionalLight.intensity = 0.5;
      } else {
          threeState.scene.background = new THREE.Color(0x3498db); // Day color from CSS var
          threeState.ambientLight.intensity = 1.0;
          threeState.directionalLight.intensity = 1.0;
      }
  }

  // --- Enhanced Color Scheme ---
  const colorScheme = {
    day: {
      primary: '#3498db',
      secondary: '#5dade2',
      accent: '#f39c12',
      background: 'rgba(52, 152, 219, 0.1)',
      text: '#2c3e50',
      textLight: '#34495e',
      cardBg: 'rgba(255, 255, 255, 0.15)',
      shadow: 'rgba(0, 0, 0, 0.1)'
    },
    night: {
      primary: '#2c3e50',
      secondary: '#34495e',
      accent: '#e74c3c',
      background: 'rgba(44, 62, 80, 0.1)',
      text: '#ecf0f1',
      textLight: '#bdc3c7',
      cardBg: 'rgba(0, 0, 0, 0.25)',
      shadow: 'rgba(0, 0, 0, 0.3)'
    },
    roles: {
      werewolf: { color: '#e74c3c', glow: 'rgba(231, 76, 60, 0.5)' },
      doctor: { color: '#2ecc71', glow: 'rgba(46, 204, 113, 0.5)' },
      seer: { color: '#9b59b6', glow: 'rgba(155, 89, 182, 0.5)' },
      villager: { color: '#3498db', glow: 'rgba(52, 152, 219, 0.5)' }
    }
  };

  // --- Typography System ---
  const typography = {
    fontFamily: {
      primary: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
      mono: 'SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace'
    },
    scale: {
      h1: '2rem',
      h2: '1.5rem',
      h3: '1.25rem',
      body: '1rem',
      small: '0.875rem',
      tiny: '0.75rem'
    },
    weight: {
      light: 300,
      regular: 400,
      medium: 500,
      bold: 700
    },
    lineHeight: {
      tight: 1.2,
      normal: 1.5,
      relaxed: 1.75
    }
  };

  // --- CSS Factory Functions ---
  function generateCSSVariables() {
    return `
      :root {
        /* Day Theme Colors */
        --day-primary: ${colorScheme.day.primary};
        --day-secondary: ${colorScheme.day.secondary};
        --day-accent: ${colorScheme.day.accent};
        --day-background: ${colorScheme.day.background};
        --day-text: ${colorScheme.day.text};
        --day-text-light: ${colorScheme.day.textLight};
        --day-card-bg: ${colorScheme.day.cardBg};
        --day-shadow: ${colorScheme.day.shadow};
        
        /* Night Theme Colors */
        --night-primary: ${colorScheme.night.primary};
        --night-secondary: ${colorScheme.night.secondary};
        --night-accent: ${colorScheme.night.accent};
        --night-background: ${colorScheme.night.background};
        --night-text: ${colorScheme.night.text};
        --night-text-light: ${colorScheme.night.textLight};
        --night-card-bg: ${colorScheme.night.cardBg};
        --night-shadow: ${colorScheme.night.shadow};
        
        /* Role Colors */
        --werewolf-color: ${colorScheme.roles.werewolf.color};
        --werewolf-glow: ${colorScheme.roles.werewolf.glow};
        --doctor-color: ${colorScheme.roles.doctor.color};
        --doctor-glow: ${colorScheme.roles.doctor.glow};
        --seer-color: ${colorScheme.roles.seer.color};
        --seer-glow: ${colorScheme.roles.seer.glow};
        --villager-color: ${colorScheme.roles.villager.color};
        --villager-glow: ${colorScheme.roles.villager.glow};
        
        /* Typography */
        --font-primary: ${typography.fontFamily.primary};
        --font-mono: ${typography.fontFamily.mono};
        --font-size-h1: ${typography.scale.h1};
        --font-size-h2: ${typography.scale.h2};
        --font-size-h3: ${typography.scale.h3};
        --font-size-body: ${typography.scale.body};
        --font-size-small: ${typography.scale.small};
        --font-size-tiny: ${typography.scale.tiny};
        --font-weight-light: ${typography.weight.light};
        --font-weight-regular: ${typography.weight.regular};
        --font-weight-medium: ${typography.weight.medium};
        --font-weight-bold: ${typography.weight.bold};
        --line-height-tight: ${typography.lineHeight.tight};
        --line-height-normal: ${typography.lineHeight.normal};
        --line-height-relaxed: ${typography.lineHeight.relaxed};
        
        /* Animations */
        --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-base: 300ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-slow: 500ms cubic-bezier(0.4, 0, 0.2, 1);
        
        /* Spacing */
        --spacing-xs: 0.25rem;
        --spacing-sm: 0.5rem;
        --spacing-md: 1rem;
        --spacing-lg: 1.5rem;
        --spacing-xl: 2rem;
        
        /* Border Radius */
        --radius-sm: 0.25rem;
        --radius-md: 0.5rem;
        --radius-lg: 0.75rem;
        --radius-xl: 1rem;
        --radius-full: 9999px;
        
        /* Legacy variables for compatibility */
        --night-bg: #2c3e50;
        --day-bg: #3498db;
        --night-text: #ecf0f1;
        --day-text: #2c3e50;
        --dead-filter: grayscale(100%) brightness(50%);
        --active-border: #f1c40f;
      }
    `;
  }

  function generateBaseStyles() {
    return `
      * {
        box-sizing: border-box;
        margin: 0;
        padding: 0;
      }
      
      .werewolf-parent {
        position: relative;
        overflow: hidden;
        width: 100%;
        height: 100%;
        font-family: var(--font-primary);
        font-size: var(--font-size-body);
        line-height: var(--line-height-normal);
      }
      
      .werewolf-parent canvas {
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        z-index: 0 !important;
      }
      
      .main-container {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 1;
        display: flex;
        background-color: transparent;
        color: var(--night-text);
      }
      
      /* Scrollbar Styling */
      ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
      }
      
      ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.1);
        border-radius: var(--radius-sm);
      }
      
      ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: var(--radius-sm);
        transition: background var(--transition-fast);
      }
      
      ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
      }
    `;
  }

  function generatePanelStyles() {
    return `
      .left-panel {
        width: 350px;
        height: 100%;
        background: linear-gradient(135deg, rgba(44, 62, 80, 0.95) 0%, rgba(52, 73, 94, 0.9) 100%);
        backdrop-filter: blur(15px);
        padding: var(--spacing-lg);
        display: flex;
        flex-direction: column;
        box-sizing: border-box;
        border-right: 1px solid rgba(255, 255, 255, 0.2);
        flex-shrink: 0;
      }
      
      .right-panel {
        flex: 1;
        height: 100%;
        background: linear-gradient(135deg, rgba(44, 62, 80, 0.8) 0%, rgba(52, 73, 94, 0.7) 100%);
        backdrop-filter: blur(10px);
        padding: var(--spacing-lg);
        display: grid;
        grid-template-rows: auto 1fr;
        gap: var(--spacing-md);
        box-sizing: border-box;
        border: 1px solid rgba(255, 255, 255, 0.1);
      }
      
      .right-panel h1, #player-list-area h1 {
        margin: 0;
        text-align: center;
        font-size: var(--font-size-h2);
        font-weight: var(--font-weight-bold);
        color: var(--night-text);
        border-bottom: 2px solid rgba(255, 255, 255, 0.2);
        padding-bottom: var(--spacing-md);
        flex-shrink: 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: linear-gradient(135deg, var(--night-text) 0%, var(--night-text-light) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
      }
      
      /* Game Status Bar */
      .game-status-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: var(--spacing-md);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border-radius: var(--radius-lg);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: var(--spacing-md);
        flex-wrap: wrap;
        gap: var(--spacing-sm);
      }
      
      @media (max-width: 400px) {
        .game-status-bar {
          padding: var(--spacing-sm);
          font-size: var(--font-size-small);
        }
        
        .phase-icon {
          font-size: 1.2em !important;
        }
      }
      
      .phase-indicator {
        display: flex;
        align-items: center;
        gap: var(--spacing-sm);
        font-weight: var(--font-weight-medium);
      }
      
      .phase-icon {
        font-size: 1.5em;
        animation: pulse 2s ease-in-out infinite;
      }
      
      .day-counter {
        font-size: var(--font-size-small);
        color: var(--night-text-light);
        font-weight: var(--font-weight-medium);
      }
      
      .alive-counter {
        display: flex;
        align-items: center;
        gap: var(--spacing-xs);
        font-size: var(--font-size-small);
        color: var(--night-text-light);
      }
      
      .alive-counter .count {
        font-weight: var(--font-weight-bold);
        color: var(--night-text);
      }
    `;
  }

  function generatePlayerCardStyles() {
    return `
      #player-list-area {
        flex: 3;
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        min-height: 0;
      }
      
      #player-list-container {
        overflow-y: auto;
        scrollbar-width: thin;
        flex-grow: 1;
        padding-right: var(--spacing-sm);
        scroll-behavior: smooth;
        position: relative;
        max-height: calc(100vh - 200px); /* Prevent infinite expansion */
      }
      
      /* Scroll Shadow Indicators */
      #player-list-container::before,
      #player-list-container::after {
        content: '';
        position: sticky;
        left: 0;
        right: 0;
        height: 20px;
        pointer-events: none;
        z-index: 1;
      }
      
      #player-list-container::before {
        top: 0;
        background: linear-gradient(to bottom, rgba(44, 62, 80, 0.8) 0%, transparent 100%);
      }
      
      #player-list-container::after {
        bottom: 0;
        background: linear-gradient(to top, rgba(44, 62, 80, 0.8) 0%, transparent 100%);
      }
      
      #player-list {
        list-style: none;
        padding: 0;
        margin: 0;
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: var(--spacing-md);
        padding: var(--spacing-xs) 0;
      }
      
      /* Responsive adjustments for smaller panels */
      @media (max-width: 400px) {
        #player-list {
          grid-template-columns: 1fr;
        }
      }
      
      .player-card {
        position: relative;
        display: flex;
        flex-direction: column;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        padding: var(--spacing-md);
        border-radius: var(--radius-lg);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 4px solid transparent;
        transition: all var(--transition-base);
        cursor: pointer;
        overflow: hidden;
        min-height: 80px;
        container-type: inline-size;
      }
      
      .player-card-header {
        display: flex;
        align-items: center;
        width: 100%;
      }
      
      .player-card-details {
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.3s ease-out, opacity 0.3s ease-out;
        opacity: 0;
        padding: 0;
        will-change: max-height, opacity;
      }
      
      .player-card.expanded .player-card-details {
        max-height: 150px; /* Reduced to prevent excessive expansion */
        opacity: 1;
        padding-top: var(--spacing-md);
        margin-top: var(--spacing-md);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
      }
      
      .detail-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: var(--spacing-xs) 0;
        font-size: var(--font-size-small);
        color: var(--night-text-light);
      }
      
      .detail-label {
        font-weight: var(--font-weight-medium);
        opacity: 0.8;
      }
      
      .detail-value {
        font-weight: var(--font-weight-regular);
        color: var(--night-text);
      }
      
      .expand-indicator {
        position: absolute;
        top: var(--spacing-sm);
        right: 60px;
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        opacity: 0.6;
        transition: transform var(--transition-fast), opacity var(--transition-fast);
      }
      
      .player-card:hover .expand-indicator {
        opacity: 1;
      }
      
      .player-card.expanded .expand-indicator {
        transform: rotate(180deg);
      }
      
      .expand-indicator::after {
        content: '▼';
        font-size: 12px;
      }
      
      /* Container query for responsive player cards */
      @container (max-width: 300px) {
        .player-card {
          padding: var(--spacing-sm);
          min-height: 70px;
        }
        
        .avatar-container {
          width: 48px !important;
          height: 48px !important;
        }
        
        .player-name {
          font-size: var(--font-size-small) !important;
        }
        
        .player-role {
          font-size: var(--font-size-tiny) !important;
        }
        
        .expand-indicator {
          right: 50px !important;
        }
        
        .player-card-details {
          font-size: var(--font-size-tiny) !important;
        }
      }
      
      /* Status Badge */
      .status-badge {
        position: absolute;
        top: var(--spacing-sm);
        right: var(--spacing-sm);
        padding: 2px 8px;
        border-radius: var(--radius-full);
        font-size: var(--font-size-tiny);
        font-weight: var(--font-weight-bold);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.2) 0%, rgba(255, 255, 255, 0.1) 100%);
        border: 1px solid rgba(255, 255, 255, 0.3);
        display: flex;
        align-items: center;
        gap: 4px;
      }
      
      .status-badge.alive {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.3) 0%, rgba(46, 204, 113, 0.2) 100%);
        border-color: var(--doctor-color);
        color: var(--doctor-color);
      }
      
      .status-badge.dead {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.3) 0%, rgba(231, 76, 60, 0.2) 100%);
        border-color: var(--werewolf-color);
        color: var(--werewolf-color);
      }
      
      .status-badge.active {
        background: linear-gradient(135deg, rgba(241, 196, 15, 0.3) 0%, rgba(241, 196, 15, 0.2) 100%);
        border-color: var(--active-border);
        color: var(--active-border);
        animation: pulse 2s ease-in-out infinite;
      }
      
      .player-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, transparent 0%, rgba(255, 255, 255, 0.05) 100%);
        opacity: 0;
        transition: opacity var(--transition-base);
        pointer-events: none;
      }
      
      .player-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.04) 100%);
        border-color: rgba(255, 255, 255, 0.2);
      }
      
      .player-card:hover::before {
        opacity: 1;
      }
      
      .player-card.active {
        border-left-color: var(--active-border);
        box-shadow: 0 0 20px rgba(241, 196, 15, 0.4), inset 0 0 20px rgba(241, 196, 15, 0.1);
        background: linear-gradient(135deg, rgba(241, 196, 15, 0.1) 0%, rgba(241, 196, 15, 0.05) 100%);
      }
      
      .player-card.dead {
        opacity: 0.5;
        transform: scale(0.98);
      }
      
      .player-card.dead:hover {
        transform: scale(0.98) translateY(-1px);
      }
      
      /* Role-specific card styling */
      .player-card.role-werewolf {
        border-left-color: var(--werewolf-color);
      }
      
      .player-card.role-doctor {
        border-left-color: var(--doctor-color);
      }
      
      .player-card.role-seer {
        border-left-color: var(--seer-color);
      }
      
      .player-card.role-villager {
        border-left-color: var(--villager-color);
      }
      
      .avatar-container {
        position: relative;
        width: 56px;
        height: 56px;
        margin-right: var(--spacing-md);
        flex-shrink: 0;
      }
      
      .player-card .avatar {
        width: 100%;
        height: 100%;
        border-radius: var(--radius-full);
        object-fit: cover;
        background-color: rgba(255, 255, 255, 0.1);
        border: 2px solid rgba(255, 255, 255, 0.2);
        transition: all var(--transition-base);
      }
      
      .player-card:hover .avatar {
        border-color: rgba(255, 255, 255, 0.4);
        transform: scale(1.05);
      }
      
      .player-card.dead .avatar {
        filter: var(--dead-filter);
      }
      
      .player-info {
        flex-grow: 1;
        overflow: hidden;
      }
      
      .player-name {
        font-weight: var(--font-weight-bold);
        font-size: var(--font-size-body);
        margin-bottom: var(--spacing-xs);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: var(--night-text);
      }
      
      .player-role, .player-status {
        font-size: var(--font-size-small);
        color: var(--night-text-light);
        font-weight: var(--font-weight-regular);
      }
      
      .threat-indicator {
        position: absolute;
        top: 50%;
        right: var(--spacing-md);
        width: 16px;
        height: 16px;
        border-radius: var(--radius-full);
        transform: translateY(-50%);
        background-color: transparent;
        transition: all var(--transition-base);
        box-shadow: 0 0 10px currentColor;
      }
      
      .player-card:hover .threat-indicator {
        transform: translateY(-50%) scale(1.2);
      }
    `;
  }

  function generateEventLogStyles() {
    return `
      #chat-log {
        list-style: none;
        padding: 0;
        margin: 0;
        flex-grow: 1;
        overflow-y: auto;
        scrollbar-width: thin;
        display: flex;
        flex-direction: column;
        gap: var(--spacing-md);
        scroll-behavior: smooth;
        padding: var(--spacing-sm);
      }
      
      /* Event Categories */
      .event-category {
        display: flex;
        align-items: center;
        gap: var(--spacing-sm);
        padding: var(--spacing-sm) var(--spacing-md);
        margin: var(--spacing-sm) 0;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border-radius: var(--radius-md);
        font-size: var(--font-size-small);
        font-weight: var(--font-weight-medium);
        color: var(--night-text-light);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        position: sticky;
        top: 0;
        z-index: 10;
        backdrop-filter: blur(10px);
      }
      
      .event-category::before,
      .event-category::after {
        content: '';
        flex: 1;
        height: 1px;
        background: rgba(255, 255, 255, 0.1);
      }
      
      .chat-entry {
        display: flex;
        align-items: flex-start;
        animation: slideIn var(--transition-base) ease-out;
      }
      
      @keyframes slideIn {
        from {
          opacity: 0;
          transform: translateX(-20px);
        }
        to {
          opacity: 1;
          transform: translateX(0);
        }
      }
      
      .chat-avatar {
        width: 44px;
        height: 44px;
        border-radius: var(--radius-full);
        margin-right: var(--spacing-md);
        object-fit: cover;
        flex-shrink: 0;
        border: 2px solid rgba(255, 255, 255, 0.2);
        transition: all var(--transition-fast);
      }
      
      .chat-entry:hover .chat-avatar {
        border-color: rgba(255, 255, 255, 0.4);
        transform: scale(1.05);
      }
      
      .message-content {
        display: flex;
        flex-direction: column;
        flex-grow: 1;
        max-width: 80%;
      }
      
      .balloon {
        padding: var(--spacing-md);
        border-radius: var(--radius-lg);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.04) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        word-wrap: break-word;
        transition: all var(--transition-fast);
        position: relative;
        overflow: hidden;
      }
      
      .balloon::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        opacity: 0;
        transition: opacity var(--transition-base);
        pointer-events: none;
      }
      
      .chat-entry:hover .balloon {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
      }
      
      .chat-entry:hover .balloon::before {
        opacity: 1;
      }
      
      .chat-entry.event-day .balloon {
        background: linear-gradient(135deg, rgba(236, 240, 241, 0.15) 0%, rgba(236, 240, 241, 0.08) 100%);
      }
      
      .chat-entry.event-night .balloon {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.3) 0%, rgba(0, 0, 0, 0.2) 100%);
      }
      
      .msg-entry {
        border-left: 4px solid var(--night-accent);
        padding: var(--spacing-md);
        margin: 0;
        border-radius: var(--radius-md);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        transition: all var(--transition-fast);
        animation: slideIn var(--transition-base) ease-out;
      }
      
      .msg-entry:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.04) 100%);
        transform: translateX(4px);
      }
      
      .msg-entry.event-day {
        background: linear-gradient(135deg, rgba(236, 240, 241, 0.15) 0%, rgba(236, 240, 241, 0.08) 100%);
      }
      
      .msg-entry.event-night {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.3) 0%, rgba(0, 0, 0, 0.2) 100%);
      }
      
      .reasoning-text {
        font-size: var(--font-size-small);
        color: var(--night-text-light);
        font-style: italic;
        margin-top: var(--spacing-sm);
        padding-left: var(--spacing-md);
        border-left: 2px solid rgba(255, 255, 255, 0.2);
        opacity: 0.8;
      }
      
      .msg-entry.game-event {
        border-left-color: var(--werewolf-color);
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.1) 0%, rgba(231, 76, 60, 0.05) 100%);
      }
      
      .msg-entry.game-win {
        border-left-color: var(--doctor-color);
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.1) 0%, rgba(46, 204, 113, 0.05) 100%);
        line-height: var(--line-height-relaxed);
      }
      
      #chat-log cite {
        font-style: normal;
        font-weight: var(--font-weight-bold);
        display: block;
        font-size: var(--font-size-small);
        color: var(--night-text);
        margin-bottom: var(--spacing-xs);
        opacity: 0.9;
      }
      
      .moderator-announcement {
        margin: 0;
        animation: slideIn var(--transition-base) ease-out;
      }
      
      .moderator-announcement-content {
        padding: var(--spacing-md);
        border-radius: var(--radius-md);
        border-left: 5px solid var(--doctor-color);
        color: var(--night-text);
        line-height: var(--line-height-normal);
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.1) 0%, rgba(46, 204, 113, 0.05) 100%);
        transition: all var(--transition-fast);
      }
      
      .moderator-announcement:hover .moderator-announcement-content {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.15) 0%, rgba(46, 204, 113, 0.08) 100%);
        transform: translateX(4px);
      }
      
      .moderator-announcement-content.event-day {
        background: linear-gradient(135deg, rgba(236, 240, 241, 0.15) 0%, rgba(236, 240, 241, 0.08) 100%);
      }
      
      .moderator-announcement-content.event-night {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.3) 0%, rgba(0, 0, 0, 0.2) 100%);
      }
      
      .timestamp {
        font-size: var(--font-size-tiny);
        color: var(--night-text-light);
        margin-left: var(--spacing-sm);
        font-weight: var(--font-weight-regular);
        opacity: 0.7;
      }
      
      .msg-text br {
        display: block;
        margin-bottom: var(--spacing-sm);
        content: "";
      }
    `;
  }

  function generateLoadingStyles() {
    return `
      /* Skeleton Screen Styles */
      .skeleton {
        background: linear-gradient(90deg, rgba(255, 255, 255, 0.05) 25%, rgba(255, 255, 255, 0.1) 50%, rgba(255, 255, 255, 0.05) 75%);
        background-size: 200% 100%;
        animation: loading 1.5s infinite;
        border-radius: var(--radius-md);
      }
      
      @keyframes loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
      }
      
      .player-card-skeleton {
        height: 80px;
        border-radius: var(--radius-lg);
        margin-bottom: var(--spacing-sm);
        border: 1px solid rgba(255, 255, 255, 0.1);
        background-color: rgba(255, 255, 255, 0.02);
        position: relative;
        overflow: hidden;
      }
      
      .player-card-skeleton::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.05) 50%, transparent 100%);
        animation: shimmer 1.5s infinite;
      }
      
      @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
      }
      
      .event-skeleton {
        height: 60px;
        border-radius: var(--radius-md);
        margin-bottom: var(--spacing-sm);
        border: 1px solid rgba(255, 255, 255, 0.05);
        background-color: rgba(255, 255, 255, 0.02);
      }
      
      .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        gap: var(--spacing-lg);
      }
      
      .loading-spinner {
        width: 60px;
        height: 60px;
        border: 3px solid rgba(255, 255, 255, 0.1);
        border-top-color: var(--night-accent);
        border-radius: 50%;
        animation: spin 1s linear infinite;
      }
      
      @keyframes spin {
        to { transform: rotate(360deg); }
      }
      
      .loading-text {
        font-size: var(--font-size-h3);
        color: var(--night-text-light);
        font-weight: var(--font-weight-medium);
        animation: pulse 2s ease-in-out infinite;
      }
      
      .skeleton-header {
        height: 40px;
        width: 60%;
        margin: 0 auto var(--spacing-lg);
        border-radius: var(--radius-md);
      }
      
      .skeleton-avatar {
        width: 56px;
        height: 56px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.05);
        flex-shrink: 0;
        margin-right: var(--spacing-md);
      }
      
      .skeleton-content {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: var(--spacing-xs);
      }
      
      .skeleton-line {
        height: 16px;
        border-radius: var(--radius-sm);
        background: rgba(255, 255, 255, 0.05);
      }
      
      .skeleton-line.short {
        width: 60%;
      }
      
      .skeleton-line.medium {
        width: 80%;
      }
      
      .skeleton-line.long {
        width: 100%;
      }
      
      /* Loading overlay for transitions */
      .loading-overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(44, 62, 80, 0.9);
        backdrop-filter: blur(5px);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 100;
        opacity: 0;
        pointer-events: none;
        transition: opacity var(--transition-base);
      }
      
      .loading-overlay.active {
        opacity: 1;
        pointer-events: all;
      }
      
      /* Initial load animation */
      .fade-in {
        animation: fadeIn var(--transition-slow) ease-out;
      }
      
      @keyframes fadeIn {
        from {
          opacity: 0;
          transform: translateY(10px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }
    `;
  }

  function generateUtilityStyles() {
    return `
      .player-capsule {
        display: inline-flex;
        align-items: center;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0.08) 100%);
        border-radius: var(--radius-full);
        padding: 2px 10px 2px 3px;
        font-size: var(--font-size-small);
        font-weight: var(--font-weight-medium);
        margin: 0 2px;
        vertical-align: middle;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all var(--transition-fast);
      }
      
      .player-capsule:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.2) 0%, rgba(255, 255, 255, 0.1) 100%);
        transform: scale(1.05);
      }
      
      .capsule-avatar {
        width: 20px;
        height: 20px;
        border-radius: var(--radius-full);
        margin-right: 6px;
        object-fit: cover;
        border: 1px solid rgba(255, 255, 255, 0.3);
      }
      
      .tts-button {
        cursor: pointer;
        font-size: 1.2em;
        margin-left: var(--spacing-sm);
        display: inline-block;
        vertical-align: middle;
        opacity: 0.6;
        transition: all var(--transition-fast);
      }
      
      .tts-button:hover {
        opacity: 1;
        transform: scale(1.1);
      }
      
      .audio-controls {
        padding: var(--spacing-md) 0;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: var(--spacing-md);
      }
      
      .audio-controls label {
        display: block;
        margin-bottom: var(--spacing-sm);
        font-size: var(--font-size-small);
        color: var(--night-text-light);
        font-weight: var(--font-weight-medium);
      }
      
      .audio-controls input[type="range"] {
        width: 100%;
        height: 6px;
        border-radius: var(--radius-full);
        background: rgba(255, 255, 255, 0.1);
        outline: none;
-appearance: none;
        appearance: none;
        cursor: pointer;
      }
      
      .audio-controls input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 16px;
        height: 16px;
        border-radius: var(--radius-full);
        background: var(--night-text);
        cursor: pointer;
        transition: all var(--transition-fast);
      }
      
      .audio-controls input[type="range"]::-webkit-slider-thumb:hover {
        transform: scale(1.2);
        background: var(--night-accent);
      }
      
      .audio-controls input[type="range"]::-moz-range-thumb {
        width: 16px;
        height: 16px;
        border-radius: var(--radius-full);
        background: var(--night-text);
        cursor: pointer;
        border: none;
        transition: all var(--transition-fast);
      }
      
      .audio-controls input[type="range"]::-moz-range-thumb:hover {
        transform: scale(1.2);
        background: var(--night-accent);
      }
      
      /* Animation keyframes */
      @keyframes fadeOut {
        to {
          opacity: 0.3;
          transform: scale(0.95);
        }
      }
      
      @keyframes pulse {
        0% {
          box-shadow: 0 0 0 0 currentColor;
        }
        70% {
          box-shadow: 0 0 0 10px transparent;
        }
        100% {
          box-shadow: 0 0 0 0 transparent;
        }
      }
      
      /* Eliminated player animation */
      .player-card.eliminated {
        animation: fadeOut 0.6s ease-out forwards;
      }
      
      /* Active player pulse */
      .player-card.active .threat-indicator {
        animation: pulse 2s infinite;
      }
    `;
  }

  // --- Combine all styles ---
  const styles = {
    variables: generateCSSVariables(),
    base: generateBaseStyles(),
    panels: generatePanelStyles(),
    playerCards: generatePlayerCardStyles(),
    eventLog: generateEventLogStyles(),
    loading: generateLoadingStyles(),
    utilities: generateUtilityStyles()
  };

  // --- CSS for the UI ---
  const css = Object.values(styles).join('\n');

  // --- TTS Management ---
  const audioMap = window.AUDIO_MAP || {};

  if (!window.kaggleWerewolf) {
      window.kaggleWerewolf = {
          audioQueue: [],
          isAudioPlaying: false,
          isAudioEnabled: false,
          lastPlayedStep: -1,
          audioPlayer: new Audio(),
          playbackRate: 1.0,
      };
  }
  const audioState = window.kaggleWerewolf;

  function setPlaybackRate(rate) {
      audioState.playbackRate = rate;
      if (audioState.isAudioPlaying) {
          audioState.audioPlayer.playbackRate = rate;
      }
  }

  function playNextInQueue() {
      if (audioState.isAudioPlaying || audioState.audioQueue.length === 0 || !audioState.isAudioEnabled) {
          return;
      }
      audioState.isAudioPlaying = true;
      const event = audioState.audioQueue.shift();
      const audioKey = event.speaker === 'moderator' ? `moderator:${event.message}` : `${event.speaker}:${event.message}`;
      const audioPath = audioMap[audioKey];

      if (audioPath) {
          audioState.audioPlayer.src = audioPath;
          audioState.audioPlayer.playbackRate = audioState.playbackRate;
          audioState.audioPlayer.onended = () => {
              audioState.isAudioPlaying = false;
              playNextInQueue();
          };
          audioState.audioPlayer.onerror = () => {
              console.error("Audio playback failed for key:", audioKey);
              audioState.isAudioPlaying = false;
              playNextInQueue();
          };
          audioState.audioPlayer.play().catch(e => {
              console.error("Audio playback failed:", e);
              audioState.isAudioPlaying = false;
              playNextInQueue();
          });
      } else {
          console.warn(`No audio found for key: "${audioKey}"`);
          audioState.isAudioPlaying = false;
          playNextInQueue();
      }
  }

  // --- Audio state activation ---
  if (!parent.dataset.audioListenerAttached) {
      parent.dataset.audioListenerAttached = 'true';
      parent.addEventListener('click', () => {
          if (!audioState.isAudioEnabled) {
              audioState.isAudioEnabled = true;
              const audio = new Audio('data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA');
              audio.play().catch(e => console.warn("Audio context activation failed:", e));
              playNextInQueue();
          }
      }, { once: true });
  }

  function speak(message, speaker) {
      if (audioState.isAudioEnabled) {
          audioState.audioQueue.push({ message, speaker });
          if (!audioState.isAudioPlaying) {
              playNextInQueue();
          }
      }
  }

  // --- Helper Functions ---
  function formatTimestamp(isoString) {
    if (!isoString) return '';
    try {
        return new Date(isoString).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
    } catch (e) {
        return '';
    }
  }

  function createPlayerCapsule(player) {
    if (!player) return '';
    return `<span class="player-capsule" title="${player.name}">
        <img src="${player.thumbnail}" class="capsule-avatar" alt="${player.name}">
        <span class="capsule-name">${player.name}</span>
    </span>`;
  }

  function replacePlayerIdsWithCapsules(text, playerIds, playerMap) {
      if (!text) return '';
      if (!playerIds || playerIds.length === 0) {
          return text;
      }
      let newText = text;
      // Sort by length descending to match longer names first (e.g. "Player10" before "Player1")
      const sortedPlayerIds = [...playerIds].sort((a, b) => b.length - a.length);

      sortedPlayerIds.forEach(playerId => {
          const player = playerMap.get(playerId);
          if (player) {
              const capsule = createPlayerCapsule(player);
              // Use a regex to replace whole words only to avoid replacing parts of other words.
              // The \b is a word boundary.
              const regex = new RegExp(`\\b${playerId.replace(/[-\/\\^$*+?.()|[\\]{}/g, '\\$&')}\\b`, 'g');
              newText = newText.replace(regex, capsule);
          }
      });
      return newText;
  }

  function replacePlayerIdsWithBold(text, playerIds) {
    if (!text) return '';
    if (!playerIds || playerIds.length === 0) {
        return text;
    }
    let newText = text;
    // Sort by length descending to match longer names first (e.g. "Player10" before "Player1")
    const sortedPlayerIds = [...playerIds].sort((a, b) => b.length - a.length);

    sortedPlayerIds.forEach(playerId => {
        // Use a regex to replace whole words only to avoid replacing parts of other words.
        const regex = new RegExp(`\\b${playerId.replace(/[-\/\\^$*+?.()|[\\]{}/g, '\\$&')}\\b`, 'g');
        newText = newText.replace(regex, `<strong>${playerId}</strong>`);
    });
    return newText;
  }


  function getThreatColor(threatLevel) {
    const value = Math.max(0, Math.min(1, threatLevel));
    // Interpolates from green (hue 120) to red (hue 0)
    const hue = 120 * (1 - value);
    // Use HSL for vibrant colors
    return `hsl(${hue}, 100%, 50%)`;
  }

  function getRoleClass(role) {
    if (!role) return '';
    const roleMap = {
      'Werewolf': 'role-werewolf',
      'Doctor': 'role-doctor',
      'Seer': 'role-seer',
      'Villager': 'role-villager'
    };
    return roleMap[role] || '';
  }

  // --- Skeleton Screen Rendering Functions ---
  function renderPlayerListSkeleton(container) {
    container.innerHTML = '';
    
    // Skeleton header
    const header = document.createElement('div');
    header.className = 'skeleton skeleton-header';
    container.appendChild(header);
    
    // Skeleton player list container
    const listContainer = document.createElement('div');
    listContainer.id = 'player-list-container';
    const playerUl = document.createElement('ul');
    playerUl.id = 'player-list';
    
    // Create skeleton player cards
    for (let i = 0; i < 8; i++) {
      const li = document.createElement('li');
      li.className = 'player-card-skeleton';
      li.innerHTML = `
        <div style="display: flex; align-items: center; padding: var(--spacing-md);">
          <div class="skeleton skeleton-avatar"></div>
          <div class="skeleton-content">
            <div class="skeleton skeleton-line medium"></div>
            <div class="skeleton skeleton-line short"></div>
          </div>
        </div>
      `;
      playerUl.appendChild(li);
    }
    
    listContainer.appendChild(playerUl);
    container.appendChild(listContainer);
  }
  
  function renderEventLogSkeleton(container) {
    container.innerHTML = '<div class="skeleton skeleton-header" style="width: 40%;"></div>';
    const logUl = document.createElement('ul');
    logUl.id = 'chat-log';
    
    // Create skeleton event entries
    for (let i = 0; i < 5; i++) {
      const li = document.createElement('li');
      li.className = 'event-skeleton skeleton';
      li.style.opacity = 1 - (i * 0.15); // Fade out effect
      logUl.appendChild(li);
    }
    
    container.appendChild(logUl);
  }
  
  function renderLoadingScreen(parent) {
    const loadingContainer = document.createElement('div');
    loadingContainer.className = 'loading-container';
    loadingContainer.innerHTML = `
      <div class="loading-spinner"></div>
      <div class="loading-text">Loading Werewolf Game...</div>
    `;
    parent.appendChild(loadingContainer);
  }

  // --- Accessibility Helpers ---
  const a11y = {
    ariaLabels: {
      playerCard: (player) => {
        const status = player.is_alive ? 'alive' : 'dead';
        const role = player.role !== 'Unknown' ? player.role : 'unknown role';
        return `${player.name}, ${role}, ${status}`;
      },
      voteButton: (target) => `Vote to eliminate ${target}`,
      threatLevel: (level) => {
        const percentage = Math.round(level * 100);
        return `Threat level: ${percentage}% dangerous`;
      },
      gamePhase: (phase, day) => `Game phase: ${phase} of day ${day}`,
      eventEntry: (type, actor, target) => {
        switch(type) {
          case 'chat': return `${actor} said a message`;
          case 'vote': return `${actor} voted to eliminate ${target}`;
          case 'elimination': return `${actor} was eliminated`;
          case 'exile': return `${actor} was exiled`;
          default: return `Game event: ${type}`;
        }
      }
    },
    keyboardNav: {
      'Tab': 'Navigate between elements',
      'Enter': 'Select/activate',
      'Space': 'Select/activate',
      'Escape': 'Close modal/cancel',
      'ArrowUp': 'Navigate up in player list',
      'ArrowDown': 'Navigate down in player list',
      'ArrowLeft': 'Navigate to previous day',
      'ArrowRight': 'Navigate to next day'
    }
  };

  // --- Keyboard Navigation Handler ---
  function setupKeyboardNavigation(container) {
    let focusedPlayerIndex = -1;
    const playerCards = container.querySelectorAll('.player-card');
    
    container.addEventListener('keydown', (e) => {
      switch(e.key) {
        case 'ArrowDown':
          e.preventDefault();
          if (focusedPlayerIndex < playerCards.length - 1) {
            focusedPlayerIndex++;
            playerCards[focusedPlayerIndex]?.focus();
          }
          break;
        case 'ArrowUp':
          e.preventDefault();
          if (focusedPlayerIndex > 0) {
            focusedPlayerIndex--;
            playerCards[focusedPlayerIndex]?.focus();
          }
          break;
        case 'Enter':
        case ' ':
          e.preventDefault();
          const focused = document.activeElement;
          if (focused && focused.classList.contains('player-card')) {
            focused.click();
          }
          break;
      }
    });
    
    // Track focus changes
    playerCards.forEach((card, index) => {
      card.addEventListener('focus', () => {
        focusedPlayerIndex = index;
      });
    });
  }

  function renderPlayerList(container, gameState, actingPlayerName, isLoading = false) {
    if (isLoading) {
      renderPlayerListSkeleton(container);
      return;
    }
    
    container.innerHTML = '';
    container.classList.add('fade-in');
    
    // Add game status bar
    const statusBar = document.createElement('div');
    statusBar.className = 'game-status-bar';
    statusBar.setAttribute('role', 'status');
    statusBar.setAttribute('aria-live', 'polite');
    
    const phaseIndicator = document.createElement('div');
    phaseIndicator.className = 'phase-indicator';
    const phaseIcon = gameState.game_state_phase === 'NIGHT' ? '🌙' : '☀️';
    phaseIndicator.innerHTML = `
        <span class="phase-icon" aria-hidden="true">${phaseIcon}</span>
        <span>${gameState.game_state_phase}</span>
    `;
    phaseIndicator.setAttribute('aria-label', a11y.ariaLabels.gamePhase(gameState.game_state_phase, gameState.day));
    
    const dayCounter = document.createElement('div');
    dayCounter.className = 'day-counter';
    dayCounter.textContent = `Day ${gameState.day}`;
    
    const aliveCounter = document.createElement('div');
    aliveCounter.className = 'alive-counter';
    const aliveCount = gameState.players.filter(p => p.is_alive).length;
    aliveCounter.innerHTML = `
        <span>Alive:</span>
        <span class="count">${aliveCount}/${gameState.players.length}</span>
    `;
    
    statusBar.appendChild(phaseIndicator);
    statusBar.appendChild(dayCounter);
    statusBar.appendChild(aliveCounter);
    container.appendChild(statusBar);
    
    // Add players header
    const header = document.createElement('h1');
    header.textContent = 'Players';
    header.id = 'players-heading';
    container.appendChild(header);
    
    const listContainer = document.createElement('div');
    listContainer.id = 'player-list-container';
    listContainer.setAttribute('role', 'region');
    listContainer.setAttribute('aria-labelledby', 'players-heading');
    
    const playerUl = document.createElement('ul');
    playerUl.id = 'player-list';
    playerUl.setAttribute('role', 'list');

    gameState.players.forEach(player => {
        const li = document.createElement('li');
        li.className = 'player-card';
        li.setAttribute('role', 'listitem');
        li.setAttribute('tabindex', '0');
        li.setAttribute('aria-label', a11y.ariaLabels.playerCard(player));
        
        if (!player.is_alive) li.classList.add('dead');
        if (player.name === actingPlayerName) li.classList.add('active');
        
        // Add role-specific class
        const roleClass = getRoleClass(player.role);
        if (roleClass) li.classList.add(roleClass);

        let roleDisplay = player.role;
        let roleIcon = '';
        if (player.role === 'Werewolf') {
            roleDisplay = player.role;
            roleIcon = '🐺';
        } else if (player.role === 'Doctor') {
            roleDisplay = player.role;
            roleIcon = '🩺';
        } else if (player.role === 'Seer') {
            roleDisplay = player.role;
            roleIcon = '🔮';
        } else if (player.role === 'Villager') {
            roleDisplay = player.role;
            roleIcon = '🧑';
        }

        const roleText = player.role !== 'Unknown' ? `${roleIcon} ${roleDisplay}` : 'Role: Unknown';
        
        // Create status badge
        let statusBadge = '';
        if (player.name === actingPlayerName) {
            statusBadge = '<div class="status-badge active">🎯 Acting</div>';
        } else if (!player.is_alive) {
            statusBadge = '<div class="status-badge dead">💀 Dead</div>';
        } else {
            statusBadge = '<div class="status-badge alive">✓ Alive</div>';
        }

        // Calculate additional player stats
        const voteHistory = gameState.eventLog.filter(e =>
            e.type === 'vote' && e.actor_id === player.name
        ).length;
        
        const wasTargeted = gameState.eventLog.filter(e =>
            (e.type === 'vote' && e.target === player.name) ||
            (e.type === 'night_vote' && e.target === player.name)
        ).length;
        
        const lastAction = gameState.eventLog.filter(e =>
            e.actor_id === player.name
        ).pop();
        
        const lastActionText = lastAction ?
            `Day ${lastAction.day} - ${lastAction.type}` :
            'No actions yet';
        
        li.innerHTML = `
            ${statusBadge}
            <div class="expand-indicator" aria-hidden="true"></div>
            <div class="player-card-header">
                <div class="avatar-container">
                    <img src="${player.thumbnail}" alt="${player.name}'s avatar" class="avatar">
                </div>
                <div class="player-info">
                    <div class="player-name" title="${player.name}">${player.name}</div>
                    <div class="player-role" aria-label="Role: ${player.role}">${roleText}</div>
                </div>
                <div class="threat-indicator" role="img" aria-label="${a11y.ariaLabels.threatLevel(0)}"></div>
            </div>
            <div class="player-card-details">
                <div class="detail-row">
                    <span class="detail-label">Status:</span>
                    <span class="detail-value">${player.is_alive ? 'Alive' : 'Dead'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Team:</span>
                    <span class="detail-value">${player.team || 'Unknown'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Votes Cast:</span>
                    <span class="detail-value">${voteHistory}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Times Targeted:</span>
                    <span class="detail-value">${wasTargeted}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Last Action:</span>
                    <span class="detail-value">${lastActionText}</span>
                </div>
            </div>
        `;
        
        // Add click handler for expanding/collapsing
        li.addEventListener('click', (e) => {
            // Don't expand if clicking on a button or link
            if (e.target.tagName === 'BUTTON' || e.target.tagName === 'A') return;
            
            // Toggle expanded state
            li.classList.toggle('expanded');
            
            // Update aria-expanded
            const isExpanded = li.classList.contains('expanded');
            li.setAttribute('aria-expanded', isExpanded);
            
            // Announce state change to screen readers
            if (isExpanded) {
                li.setAttribute('aria-label', `${a11y.ariaLabels.playerCard(player)}, expanded to show details`);
            } else {
                li.setAttribute('aria-label', a11y.ariaLabels.playerCard(player));
            }
        });
        playerUl.appendChild(li);
    });

    listContainer.appendChild(playerUl);
    container.appendChild(listContainer);

    // --- Threat Indicator Management ---
    gameState.players.forEach((player, index) => {
        const li = playerUl.children[index];
        const indicator = li.querySelector('.threat-indicator');
        if (!indicator) return;

        if (player.is_alive) {
            const threatLevel = gameState.playerThreatLevels.get(player.name) || 0;
            indicator.style.backgroundColor = getThreatColor(threatLevel);
            indicator.setAttribute('aria-label', a11y.ariaLabels.threatLevel(threatLevel));
        } else {
            indicator.style.backgroundColor = 'transparent';
            indicator.setAttribute('aria-label', 'No threat (player dead)');
        }
    });

    // --- Audio Controls ---
    const audioControls = document.createElement('div');
    audioControls.className = 'audio-controls';
    audioControls.innerHTML = `
        <label for="playback-speed">Audio Speed: <span id="speed-label">${audioState.playbackRate.toFixed(1)}</span>x</label>
        <input type="range" id="playback-speed" min="0.5" max="2.5" step="0.1" value="${audioState.playbackRate}">
    `;
    container.appendChild(audioControls);
    
    // Setup keyboard navigation
    setupKeyboardNavigation(container);

    const speedSlider = audioControls.querySelector('#playback-speed');
    const speedLabel = audioControls.querySelector('#speed-label');

    speedSlider.addEventListener('input', (e) => {
        const newRate = parseFloat(e.target.value);
        setPlaybackRate(newRate);
        speedLabel.textContent = newRate.toFixed(1);
    });
  }

  function renderEventLog(container, gameState, playerMap, isLoading = false) {
    if (isLoading) {
      renderEventLogSkeleton(container);
      return;
    }
    
    const header = document.createElement('h1');
    header.textContent = 'Event Log';
    header.id = 'event-log-heading';
    container.innerHTML = '';
    container.appendChild(header);
    container.classList.add('fade-in');
    
    const logUl = document.createElement('ul');
    logUl.id = 'chat-log';
    logUl.setAttribute('role', 'log');
    logUl.setAttribute('aria-labelledby', 'event-log-heading');
    logUl.setAttribute('aria-live', 'polite');

    const logEntries = gameState.eventLog;

    if (logEntries.length === 0) {
        const li = document.createElement('li');
        li.className = 'msg-entry';
        li.innerHTML = `<cite>System</cite><div>The game is about to begin...</div>`;
        logUl.appendChild(li);
    } else {
        let lastDay = -1;
        let lastPhase = '';
        
        logEntries.forEach(entry => {
            // Add day/phase separator if changed
            if (entry.day !== lastDay || entry.phase !== lastPhase) {
                if (entry.day !== undefined && entry.day !== Infinity) {
                    const separator = document.createElement('li');
                    separator.className = 'event-category';
                    separator.setAttribute('role', 'separator');
                    const phaseIcon = entry.phase === 'NIGHT' ? '🌙' : '☀️';
                    separator.innerHTML = `Day ${entry.day} - <span aria-hidden="true">${phaseIcon}</span> ${entry.phase}`;
                    separator.setAttribute('aria-label', `Day ${entry.day}, ${entry.phase} phase`);
                    logUl.appendChild(separator);
                    lastDay = entry.day;
                    lastPhase = entry.phase;
                }
            }
            const li = document.createElement('li');
            let reasoningHtml = entry.reasoning ? `<div class="reasoning-text">"${entry.reasoning}"</div>` : '';
            
            // --- Phase Correction Logic ---
            let phase = (entry.phase || 'Day').toUpperCase();
            const entryType = entry.type;
            const systemText = (entry.text || '').toLowerCase();

            if (entryType === 'exile' || entryType === 'vote' || entryType === 'timeout' || (entryType === 'system' && (systemText.includes('discussion') || systemText.includes('voting for exile')))) {
                phase = 'DAY';
            } else if (
                entryType === 'elimination' || entryType === 'save' || entryType === 'night_vote' ||
                entryType === 'seer_inspection' || entryType === 'seer_inspection_result' ||
                entryType === 'doctor_heal_action' ||
                (entryType === 'system' && (systemText.includes('werewolf vote request') || systemText.includes('doctor save request') || systemText.includes('seer inspect request') || systemText.includes('night has begun')))
            ) {
                phase = 'NIGHT';
            }

            const phaseClass = `event-${phase.toLowerCase()}`;
            
            let phaseEmoji = phase;
            if (phase === 'DAY') {
                phaseEmoji = '☀️'; // Sun emoji
            } else if (phase === 'NIGHT') {
                phaseEmoji = '🌙'; // Crescent moon emoji
            }

            const dayPhaseString = entry.day !== Infinity ? `[D${entry.day} ${phaseEmoji}]` : '';
            const timestampHtml = `<span class="timestamp">${dayPhaseString} ${formatTimestamp(entry.timestamp)}</span>`;

            switch (entry.type) {
                case 'chat':
                    const speaker = playerMap.get(entry.speaker);
                    if (!speaker) return;
                    const messageText = replacePlayerIdsWithBold(entry.message, entry.mentioned_player_ids);
                    li.className = `chat-entry event-day`;
                    li.setAttribute('role', 'article');
                    li.setAttribute('aria-label', a11y.ariaLabels.eventEntry('chat', speaker.name, null));
                    li.innerHTML = `
                        <img src="${speaker.thumbnail}" alt="${speaker.name}'s avatar" class="chat-avatar">
                        <div class="message-content">
                            <cite>${speaker.name} ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">
                                    <quote>${messageText}</quote>
                                </div>
                                ${reasoningHtml}
                            </div>
                        </div>
                    `;
                    const balloonText = li.querySelector('.balloon-text');
                    if (balloonText) {
                        const ttsButton = document.createElement('span');
                        ttsButton.className = 'tts-button';
                        ttsButton.textContent = '🔊'; // Speaker icon
                        ttsButton.setAttribute('aria-label', 'Play audio for this message');
                        ttsButton.setAttribute('role', 'button');
                        ttsButton.onclick = () => speak(entry.message, entry.speaker);
                        balloonText.appendChild(ttsButton);
                    }
                    break;
                case 'seer_inspection':
                    const seerInspector = playerMap.get(entry.actor_id);
                    if (!seerInspector) return;
                    const seerTargetCap = createPlayerCapsule(playerMap.get(entry.target));
                    const seerCap = createPlayerCapsule(playerMap.get(entry.actor_id));
                    li.className = `chat-entry event-night`;
                    li.setAttribute('role', 'article');
                    li.setAttribute('aria-label', `Seer inspection: ${entry.actor_id} inspected ${entry.target}`);
                    li.innerHTML = `
                        <img src="${seerInspector.thumbnail}" alt="${seerInspector.name}'s avatar" class="chat-avatar">
                        <div class="message-content">
                            <cite>Secret Inspect by ${seerInspector.name} (Seer) ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">${seerCap} chose to inspect ${seerTargetCap}'s role.</div>
                                ${reasoningHtml}
                            </div>
                        </div>
                    `;
                    break;
                case 'seer_inspection_result':
                    const seerResultViewer = playerMap.get(entry.seer);
                    if (!seerResultViewer) return;
                    const seerCap_ = createPlayerCapsule(playerMap.get(entry.seer));
                    const seerResultTargetCap = createPlayerCapsule(playerMap.get(entry.target));
                    li.className = `chat-entry ${phaseClass}`;
                    li.innerHTML = `
                        <img src="${seerResultViewer.thumbnail}" alt="${seerResultViewer.name}" class="chat-avatar">
                        <div class="message-content">
                            <cite>${entry.seer} (Seer) ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">${seerCap_} saw ${seerResultTargetCap}'s role is a <strong>${entry.role}</strong>.</div>
                            </div>
                        </div>
                    `;
                    break;
                case 'doctor_heal_action':
                    const doctor = playerMap.get(entry.actor_id);
                    if (!doctor) return;
                    const docTargetCap = createPlayerCapsule(playerMap.get(entry.target));
                    const docCap = createPlayerCapsule(playerMap.get(entry.actor_id));
                    li.className = `chat-entry event-night`;
                    li.innerHTML = `
                        <img src="${doctor.thumbnail}" alt="${doctor.name}" class="chat-avatar">
                        <div class="message-content">
                            <cite>Secret Heal by ${doctor.name} (Doctor) ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">${docCap} chose to heal ${docTargetCap}.</div>
                                ${reasoningHtml}
                            </div>
                        </div>
                    `;
                    break;
                case 'system':
                    if (entry.text && entry.text.includes('has begun')) return;

                    let systemText = entry.text;
                    // Regex to find python list of strings and replace it with just the comma-separated content
                    const listRegex = /\\\\[(.*?)\\\\]/g;
                    systemText = systemText.replace(listRegex, (match, listContent) => {
                        // listContent is "'player-1', 'player-2', 'player-3'"
                        return listContent.replace(/'/g, "").replace(/, /g, " "); // Becomes "player-1 player-2 player-3"
                    });

                    const allPlayerIdsForSystem = Array.from(playerMap.keys());
                    const finalSystemText = replacePlayerIdsWithCapsules(systemText, allPlayerIdsForSystem, playerMap);

                    li.className = `moderator-announcement`;
                    li.setAttribute('role', 'article');
                    li.setAttribute('aria-label', `Moderator announcement: ${entry.text.substring(0, 50)}...`);
                    li.innerHTML = `
                        <cite>Moderator <span aria-hidden="true">📢</span> ${timestampHtml}</cite>
                        <div class="moderator-announcement-content ${phaseClass}">
                            <div class="msg-text">${finalSystemText.replace(/\n/g, '<br>')}</div>
                        </div>
                    `;
                    break;
                case 'exile':
                    const exiledPlayerCap = createPlayerCapsule(playerMap.get(entry.name));
                    li.className = `msg-entry game-event event-day`;
                    li.setAttribute('role', 'article');
                    li.setAttribute('aria-label', a11y.ariaLabels.eventEntry('exile', entry.name, null));
                    li.innerHTML = `<cite>Exile ${timestampHtml}</cite><div class="msg-text">${exiledPlayerCap} (${entry.role}) was exiled by vote.</div>`;
                    break;
                case 'elimination':
                    const elimPlayerCap = createPlayerCapsule(playerMap.get(entry.name));
                    li.className = `msg-entry game-event event-night`;
                    li.setAttribute('role', 'article');
                    li.setAttribute('aria-label', a11y.ariaLabels.eventEntry('elimination', entry.name, null));
                    li.innerHTML = `<cite>Elimination ${timestampHtml}</cite><div class="msg-text">${elimPlayerCap} was eliminated. Their role was a ${entry.role}.</div>`;
                    break;
                case 'save':
                     const savedPlayerCap = createPlayerCapsule(playerMap.get(entry.saved_player));
                     li.className = `msg-entry event-night`;
                     li.innerHTML = `<cite>Doctor Save ${timestampHtml}</cite><div class="msg-text">Player ${savedPlayerCap} was attacked but saved by a Doctor!</div>`;
                    break;
                case 'vote':
                    const voter = playerMap.get(entry.actor_id);
                    if (!voter) return;
                    const voterCap = createPlayerCapsule(playerMap.get(entry.actor_id));
                    const voteTargetCap = createPlayerCapsule(playerMap.get(entry.target));
                    li.className = `chat-entry event-day`;
                    li.setAttribute('role', 'article');
                    li.setAttribute('aria-label', a11y.ariaLabels.eventEntry('vote', entry.actor_id, entry.target));
                    li.innerHTML = `
                        <img src="${voter.thumbnail}" alt="${voter.name}'s avatar" class="chat-avatar">
                        <div class="message-content">
                            <cite>${voter.name} ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">${voterCap} votes to eliminate ${voteTargetCap}.</div>
                                ${reasoningHtml}
                            </div>
                        </div>
                     `;
                     break;
                case 'timeout':
                    const to_voter = playerMap.get(entry.actor_id);
                    if (!to_voter) return;
                    const to_voterCap = createPlayerCapsule(playerMap.get(entry.actor_id));
                    li.className = `chat-entry event-day`;
                    li.innerHTML = `
                        <img src="${to_voter.thumbnail}" alt="${to_voter.name}" class="chat-avatar">
                        <div class="message-content">
                            <cite>${to_voter.name} ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">${to_voterCap} timed out and abstained from voting.</div>
                                ${reasoningHtml}
                            </div>
                        </div>
                     `;
                     break;
                case 'night_vote':
                    const nightVoter = playerMap.get(entry.actor_id);
                    if (!nightVoter) return;
                    const nightVoterCap = createPlayerCapsule(playerMap.get(entry.actor_id));
                    const nightVoteTargetCap = createPlayerCapsule(playerMap.get(entry.target));
                    li.className = `chat-entry event-night`;
                    li.innerHTML = `
                        <img src="${nightVoter.thumbnail}" alt="${nightVoter.name}" class="chat-avatar">
                        <div class="message-content">
                            <cite>Secret Vote by ${nightVoter.name} (Werewolf) ${timestampHtml}</cite>
                            <div class="balloon">
                                <div class="balloon-text">${nightVoterCap} votes to eliminate ${nightVoteTargetCap}.</div>
                                ${reasoningHtml}
                            </div>
                        </div>
                    `;
                    break;
                case 'game_over':
                    const winnersText = entry.winners.map(p => createPlayerCapsule(playerMap.get(p))).join(' ');
                    const losersText = entry.losers.map(p => createPlayerCapsule(playerMap.get(p))).join(' ');
                    li.className = `msg-entry game-win ${phaseClass}`;
                    li.setAttribute('role', 'article');
                    li.setAttribute('aria-label', `Game over: ${entry.winner} team won`);
                    li.innerHTML = `
                        <cite>Game Over ${timestampHtml}</cite>
                        <div class="msg-text">
                            <div>The <strong>${entry.winner}</strong> team has won!</div><br>
                            <div><strong>Winning Team:</strong> ${winnersText}</div>
                            <div><strong>Losing Team:</strong> ${losersText}</div>
                        </div>
                    `;
                    break;
            }
            if (li.innerHTML) logUl.appendChild(li);
        });
    }

    container.appendChild(logUl);
    logUl.scrollTop = logUl.scrollHeight;
  }

    // --- Main Rendering Logic ---

    // Clean up previous UI, but not the canvas
    const oldUI = parent.querySelector('.main-container');
    if (oldUI) {
        parent.removeChild(oldUI);
    }
    const oldStyle = parent.querySelector('style');
    if (oldStyle) {
        parent.removeChild(oldStyle);
    }

    initThreeJs(); // Initialize three.js if not already done

    // Show loading screen if no environment data
    if (!environment || !environment.steps || environment.steps.length === 0) {
        const style = document.createElement('style');
        style.textContent = css + `
          /* Accessibility Styles */
          .player-card:focus {
            outline: 2px solid var(--night-accent);
            outline-offset: 2px;
          }
          
          .player-card:focus-visible {
            outline: 2px solid var(--night-accent);
            outline-offset: 2px;
          }
          
          button:focus-visible,
          .tts-button:focus-visible {
            outline: 2px solid var(--night-accent);
            outline-offset: 2px;
          }
          
          /* Skip to content link */
          .skip-link {
            position: absolute;
            top: -40px;
            left: 0;
            background: var(--night-primary);
            color: var(--night-text);
            padding: var(--spacing-sm) var(--spacing-md);
            text-decoration: none;
            border-radius: var(--radius-md);
            z-index: 1000;
          }
          
          .skip-link:focus {
            top: 10px;
            left: 10px;
          }
          
          /* High contrast mode preparation */
          @media (prefers-contrast: high) {
            .player-card {
              border: 2px solid currentColor;
            }
            
            .status-badge {
              border-width: 2px;
            }
            
            .threat-indicator {
              border: 2px solid currentColor;
            }
          }
          
          /* Reduced motion support */
          @media (prefers-reduced-motion: reduce) {
            * {
              animation-duration: 0.01ms !important;
              animation-iteration-count: 1 !important;
              transition-duration: 0.01ms !important;
            }
          }
        `;
        parent.appendChild(style);
        renderLoadingScreen(parent);
        return;
    }
    
    if (step >= environment.steps.length) {
        const tempContainer = document.createElement("div");
        tempContainer.className = 'loading-container';
        tempContainer.innerHTML = '<div class="loading-text">Invalid step...</div>';
        parent.appendChild(tempContainer);
        return;
    }

    // --- State Reconstruction ---
    let gameState = {
        players: [],
        day: 0,
        phase: 'GAME_SETUP',
        game_state_phase: 'DAY',
        gameWinner: null,
        eventLog: [],
        playerThreatLevels: new Map()
    };

    const firstObs = environment.steps[0]?.[0]?.observation?.raw_observation;
    let allPlayerNamesList;
    let playerThumbnails = {};

    if (firstObs && firstObs.all_player_ids) {
        allPlayerNamesList = firstObs.all_player_ids;
        playerThumbnails = firstObs.player_thumbnails || {};
    } else if (environment.configuration && environment.configuration.agents) {
        console.warn("Renderer: Initial observation missing or incomplete. Reconstructing players from configuration.");
        allPlayerNamesList = environment.configuration.agents.map(agent => agent.id);
        environment.configuration.agents.forEach(agent => {
            if (agent.id && agent.thumbnail) {
                playerThumbnails[agent.id] = agent.thumbnail;
            }
        });
    }

    if (!allPlayerNamesList || allPlayerNamesList.length === 0) {
        const style = document.createElement('style');
        style.textContent = css;
        parent.appendChild(style);
        
        const tempContainer = document.createElement("div");
        tempContainer.className = 'loading-container';
        tempContainer.innerHTML = `
            <div class="loading-spinner"></div>
            <div class="loading-text">Waiting for players to join...</div>
        `;
        parent.appendChild(tempContainer);
        return;
    }

    gameState.players = allPlayerNamesList.map(name => ({
        name: name, is_alive: true, role: 'Unknown', team: 'Unknown', status: 'Alive',
        thumbnail: playerThumbnails[name] || `https://via.placeholder.com/40/2c3e50/ecf0f1?text=${name.charAt(0)}`
    }));
    const playerMap = new Map(gameState.players.map(p => [p.name, p]));

    // Initialize all threat levels to 0 (SAFE)
    gameState.players.forEach(p => gameState.playerThreatLevels.set(p.name, 0));

    // Get initial roles from the start of the game
    const roleAndTeamMap = new Map();
    const moderatorInitialLog = environment.info?.MODERATOR_OBSERVATION?.[0] || [];
    moderatorInitialLog.forEach(dataEntry => {
        if (dataEntry.data_type === 'GameStartRoleDataEntry') {
            const historyEvent = JSON.parse(dataEntry.json_str);
            const data = historyEvent.data;
            if (data) roleAndTeamMap.set(data.player_id, { role: data.role, team: data.team });
        }
    });
    roleAndTeamMap.forEach((info, playerId) => {
        const player = playerMap.get(playerId);
        if (player) { player.role = info.role; player.team = info.team; }
    });

    const processedEvents = new Set();
    let lastPhase = null;
    let lastDay = -1;

    function threatStringToLevel(threatString) {
        switch(threatString) {
            case 'SAFE': return 0;
            case 'UNEASY': return 0.5;
            case 'IN_DANGER': return 1.0;
            default: return 0; // Default to safe
        }
    }

    for (let s = 0; s <= step; s++) {
        const stepStateList = environment.steps[s];
        if (!stepStateList) continue;

        // Update overall game state from the first agent in the current step
        const currentObsForStep = stepStateList[0]?.observation?.raw_observation;
        if (currentObsForStep) {
            if (currentObsForStep.day > lastDay) {
                if (currentObsForStep.day > 0) gameState.eventLog.push({ type: 'system', step: s, day: currentObsForStep.day, phase: 'DAY', text: `Day ${currentObsForStep.day} has begun.` });
            }
            lastDay = currentObsForStep.day;
            lastPhase = currentObsForStep.phase;
            gameState.day = currentObsForStep.day;
            gameState.phase = currentObsForStep.phase;
            gameState.game_state_phase = currentObsForStep.game_state_phase;
        }

        // Process confirmed events from the moderator log
        const moderatorLogForStep = environment.info?.MODERATOR_OBSERVATION?.[s] || [];
        moderatorLogForStep.forEach(dataEntry => {
             const eventKey = dataEntry.json_str;
             if (processedEvents.has(eventKey)) return;
             processedEvents.add(eventKey);

             const historyEvent = JSON.parse(dataEntry.json_str);
             const data = historyEvent.data;
             const timestamp = historyEvent.created_at;

            // Update threat level whenever an action with that info appears
            if (data && data.actor_id && data.perceived_threat_level) {
                const threatScore = threatStringToLevel(data.perceived_threat_level);
                gameState.playerThreatLevels.set(data.actor_id, threatScore);
            }

            if (!data) {
                if (historyEvent.entry_type === 'vote_action') {
                    const match = historyEvent.description.match(/P(player_\d+)/);
                    if (match) {
                        const actor_id = match[1];
                    }
                }
                return;
            }

             switch (dataEntry.data_type) {
                case 'ChatDataEntry':
                    gameState.eventLog.push({ type: 'chat', step: s, day: historyEvent.day, phase: historyEvent.phase, speaker: data.actor_id, message: data.message, reasoning: data.reasoning, timestamp, mentioned_player_ids: data.mentioned_player_ids || [] });
                    break;
                case 'DayExileVoteDataEntry':
                    gameState.eventLog.push({ type: 'vote', step: s, day: historyEvent.day, phase: historyEvent.phase, actor_id: data.actor_id, target: data.target_id, reasoning: data.reasoning, timestamp });
                    break;
                case 'WerewolfNightVoteDataEntry':
                    gameState.eventLog.push({ type: 'night_vote', step: s, day: historyEvent.day, phase: historyEvent.phase, actor_id: data.actor_id, target: data.target_id, reasoning: data.reasoning, timestamp });
                    break;
                case 'DoctorHealActionDataEntry':
                    gameState.eventLog.push({ type: 'doctor_heal_action', step: s, day: historyEvent.day, phase: historyEvent.phase, actor_id: data.actor_id, target: data.target_id, reasoning: data.reasoning, timestamp });
                    break;
                case 'SeerInspectActionDataEntry':
                    gameState.eventLog.push({ type: 'seer_inspection', step: s, day: historyEvent.day, phase: historyEvent.phase, actor_id: data.actor_id, target: data.target_id, reasoning: data.reasoning, timestamp });
                    break;
                case 'DayExileElectedDataEntry':
                    gameState.eventLog.push({ type: 'exile', step: s, day: historyEvent.day, phase: 'DAY', name: data.elected_player_id, role: data.elected_player_role_name, timestamp });
                    break;
                case 'WerewolfNightEliminationDataEntry':
                    gameState.eventLog.push({ type: 'elimination', step: s, day: historyEvent.day, phase: 'NIGHT', name: data.eliminated_player_id, role: data.eliminated_player_role_name, timestamp });
                    break;
                case 'SeerInspectResultDataEntry':
                    gameState.eventLog.push({ type: 'seer_inspection_result', step: s, day: historyEvent.day, phase: 'NIGHT', seer: data.actor_id, target: data.target_id, role: data.role, timestamp });
                    break;
                case 'DoctorSaveDataEntry':
                    gameState.eventLog.push({ type: 'save', step: s, day: historyEvent.day, phase: 'NIGHT', saved_player: data.saved_player_id, timestamp });
                    break;
                case 'GameEndResultsDataEntry':
                    gameState.gameWinner = data.winner_team;
                    const winners = gameState.players.filter(p => p.team === data.winner_team).map(p => p.name);
                    const losers = gameState.players.filter(p => p.team !== data.winner_team).map(p => p.name);
                    gameState.eventLog.push({ type: 'game_over', step: s, day: Infinity, phase: 'GAME_OVER', winner: data.winner_team, winners, losers, timestamp });
                    break;
                default:
                    if (historyEvent.entry_type === "moderator_announcement") {
                        gameState.eventLog.push({ type: 'system', step: s, day: historyEvent.day, phase: historyEvent.phase, text: historyEvent.description, timestamp, data: data});
                    }
                    break;
             }
        });
    }

    // --- Audio Playback Management ---
    if (step < audioState.lastPlayedStep) {
        audioState.audioQueue = [];
        audioState.isAudioPlaying = false;
        if (audioState.audioPlayer) {
            audioState.audioPlayer.pause();
        }
    }

    const eventsToPlay = gameState.eventLog.filter(entry =>
        entry.step > audioState.lastPlayedStep && entry.step <= step
    );

    if (eventsToPlay.length > 0) {
        eventsToPlay.forEach(entry => {
            let audioEvent = null;
            if (entry.type === 'chat') {
                audioEvent = { message: entry.message, speaker: entry.speaker };
            } else if (entry.type === 'system') {
                const text = entry.text.toLowerCase();
                if (text.includes('night') && text.includes('begins')) {
                    audioEvent = { message: 'night_begins', speaker: 'moderator' };
                } else if (text.includes('day') && text.includes('begins')) {
                    audioEvent = { message: 'day_begins', speaker: 'moderator' };
                } else if (text.includes('discussion')) {
                    audioEvent = { message: 'discussion_begins', speaker: 'moderator' };
                } else if (text.includes('voting phase begins')) {
                    audioEvent = { message: 'voting_begins', speaker: 'moderator' };
                }
            } else if (entry.type === 'exile') {
                const message = `Player ${entry.name} was exiled by vote. Their role was a ${entry.role}.`;
                audioEvent = { message: message, speaker: 'moderator' };
            } else if (entry.type === 'elimination') {
                const message = `Player ${entry.name} was eliminated. Their role was a ${entry.role}.`;
                audioEvent = { message: message, speaker: 'moderator' };
            } else if (entry.type === 'save') {
                const message = `Player ${entry.saved_player} was attacked but saved by a Doctor!`;
                audioEvent = { message: message, speaker: 'moderator' };
            } else if (entry.type === 'game_over') {
                const message = `The game is over. The ${entry.winner} team has won!`;
                audioEvent = { message: message, speaker: 'moderator' };
            }

            if (audioEvent) {
                audioState.audioQueue.push(audioEvent);
            }
        });

        if (audioState.isAudioEnabled && !audioState.isAudioPlaying) {
            playNextInQueue();
        }
    }
    audioState.lastPlayedStep = step;


    // Update player statuses based on the final log
    gameState.players.forEach(p => { p.is_alive = true; p.status = 'Alive'; });
    gameState.eventLog.forEach(entry => {
        if (entry.type === 'exile' || entry.type === 'elimination') {
            const player = playerMap.get(entry.name);
            if (player) {
                player.is_alive = false;
                player.status = entry.type === 'exile' ? 'Exiled' : 'Eliminated';
            }
        }
    });

    gameState.eventLog.sort((a, b) => {
        if (a.day !== b.day) return a.day - b.day;
        const phaseOrder = { 'DAY': 1, 'NIGHT': 2 };
        const aPhase = (a.phase || '').toUpperCase();
        const bPhase = (b.phase || '').toUpperCase();
        const aOrder = phaseOrder[aPhase] || 99;
        const bOrder = phaseOrder[bPhase] || 99;
        if (aOrder !== bOrder) return aOrder - bOrder;
        if (a.timestamp && b.timestamp) return new Date(a.timestamp) - new Date(b.timestamp);
        return 0;
    });

    const lastStepStateList = environment.steps[step];
    const actingPlayerIndex = lastStepStateList.findIndex(s => s.status === 'ACTIVE');
    const actingPlayerName = actingPlayerIndex !== -1 ? allPlayerNamesList[actingPlayerIndex] : "N.A";

    Object.assign(parent.style, {
      width: `${width}px`,
      height: `${height}px`,
      position: 'relative',
      overflow: 'hidden'
    });
    parent.className = 'werewolf-parent';

    const style = document.createElement('style');
    style.textContent = css;
    parent.appendChild(style);

    const isNight = (gameState.game_state_phase || '').toLowerCase() === 'night';
    updateBackground(isNight);

    // Add skip link for screen readers
    const skipLink = document.createElement('a');
    skipLink.href = '#event-log-heading';
    skipLink.className = 'skip-link';
    skipLink.textContent = 'Skip to event log';
    parent.appendChild(skipLink);
    
    const mainContainer = document.createElement('div');
    mainContainer.className = 'main-container';
    mainContainer.setAttribute('role', 'main');
    parent.appendChild(mainContainer);

    // Create left panel (sidebar)
    const leftPanel = document.createElement('div');
    leftPanel.className = 'left-panel';
    leftPanel.setAttribute('role', 'complementary');
    leftPanel.setAttribute('aria-label', 'Player information sidebar');
    mainContainer.appendChild(leftPanel);
    
    const playerListArea = document.createElement('div');
    playerListArea.id = 'player-list-area';
    leftPanel.appendChild(playerListArea);

    // Create right panel (event log)
    const rightPanel = document.createElement('div');
    rightPanel.className = 'right-panel';
    rightPanel.setAttribute('role', 'region');
    rightPanel.setAttribute('aria-label', 'Game events');
    mainContainer.appendChild(rightPanel);

    // Add loading overlay for transitions
    const loadingOverlay = document.createElement('div');
    loadingOverlay.className = 'loading-overlay';
    loadingOverlay.innerHTML = `
        <div class="loading-spinner"></div>
    `;
    mainContainer.appendChild(loadingOverlay);

    // Check if we're still loading initial data
    const isInitialLoad = step === 0 && gameState.eventLog.length === 0;
    
    // Add transition detection
    if (window.werewolfLastStep !== undefined && Math.abs(step - window.werewolfLastStep) > 1) {
        // Show loading overlay for large step changes
        loadingOverlay.classList.add('active');
        setTimeout(() => {
            loadingOverlay.classList.remove('active');
        }, 300);
    }
    window.werewolfLastStep = step;
    
    renderPlayerList(playerListArea, gameState, actingPlayerName, isInitialLoad);
    renderEventLog(rightPanel, gameState, playerMap, isInitialLoad);
}
                        gameState.eventLog.push({ type: 'timeout', step: s, day: historyEvent.day, phase: historyEvent.phase, actor_id: actor_id, reasoning: "Timed out", timestamp: historyEvent.created_at });