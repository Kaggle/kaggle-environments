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
//    loader.load(
//      'assets/low_poly_medieval_windmill/scene.gltf',
//      function (gltf) {
//        threeState.model = gltf.scene;
//        threeState.model.scale.set(0.5, 0.5, 0.5); // Adjust scale for the new model
//        threeState.model.position.set(0, -2, 0); // Adjust position
//        threeState.scene.add(threeState.model);
//      },
//      undefined,
//      function(error) {
//        console.error('An error occurred loading the model:', error);
//      }
//    );
//
//    threeState.camera.position.z = 5;
//    threeState.initialized = true;
//    animate();
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

  // --- CSS for the UI ---
  const css = `
        :root {
            /* Dramatic Color Palette */
            --night-bg: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
            --day-bg: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            --night-surface: rgba(15, 15, 35, 0.95);
            --day-surface: rgba(255, 255, 255, 0.95);
            --night-text: #ffffff;
            --day-text: #2d3436;
            --night-text-secondary: #a0a6b8;
            --day-text-secondary: #636e72;
            
            /* Vibrant Status Colors */
            --color-alive: #00ff88;
            --color-dead: #ff4757;
            --color-danger: #ff3838;
            --color-warning: #ffb84d;
            --color-success: #00ff88;
            --color-info: #3742fa;
            
            /* Enhanced Role Colors */
            --color-werewolf: linear-gradient(135deg, #d63031 0%, #74b9ff 100%);
            --color-villager: linear-gradient(135deg, #00b894 0%, #55a3ff 100%);
            --color-seer: linear-gradient(135deg, #a29bfe 0%, #fd79a8 100%);
            --color-doctor: linear-gradient(135deg, #fd79a8 0%, #fdcb6e 100%);
            
            /* Legacy variables for compatibility */
            --dead-filter: grayscale(100%) brightness(50%);
            --active-border: #fdcb6e;
            
            /* Animation Variables */
            --transition-fast: 150ms ease-in-out;
            --transition-base: 300ms ease-in-out;
            --transition-slow: 500ms ease-in-out;
            
            /* Spacing */
            --spacing-xs: 4px;
            --spacing-sm: 8px;
            --spacing-md: 16px;
            --spacing-lg: 24px;
            --spacing-xl: 32px;
            
            /* Border Radius */
            --radius-sm: 4px;
            --radius-md: 8px;
            --radius-lg: 12px;
            --radius-xl: 16px;
            --radius-full: 50%;
            
            /* Shadows */
            --shadow-sm: 0 2px 8px rgba(0,0,0,0.1);
            --shadow-md: 0 4px 16px rgba(0,0,0,0.15);
            --shadow-lg: 0 8px 32px rgba(0,0,0,0.2);
        }
        .werewolf-parent {
            position: relative;
            overflow: hidden;
            width: 100%;
            height: 100%;
        }
        .main-container {
            position: relative;
            z-index: 1;
            display: flex;
            height: 100%;
            width: 100%;
            background: var(--night-bg);
            font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--night-text);
            gap: 20px;
            padding: 20px;
            box-sizing: border-box;
        }
        .left-panel, .right-panel {
            background: var(--night-surface);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            height: calc(100% - 40px);
            box-sizing: border-box;
            box-shadow:
                0 25px 50px -12px rgba(0, 0, 0, 0.5),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            position: relative;
            overflow: hidden;
        }
        .left-panel::before, .right-panel::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
            z-index: 1;
        }
        .left-panel {
            width: 350px;
            flex-shrink: 0;
        }
        .right-panel {
            flex-grow: 1;
            min-width: 0;
        }
        .right-panel h1, #player-list-area h1 {
            margin: 0 0 32px 0;
            text-align: center;
            font-size: 2.2em;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 0%, #a0a6b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            position: relative;
            padding-bottom: 16px;
            flex-shrink: 0;
        }
        .right-panel h1::after, #player-list-area h1::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 60px;
            height: 3px;
            background: linear-gradient(90deg, transparent, #ffffff, transparent);
            border-radius: 2px;
        }
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
        }
        #player-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .player-card {
            position: relative;
            display: flex;
            align-items: center;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
            backdrop-filter: blur(10px);
            padding: 16px;
            border-radius: 16px;
            margin-bottom: 16px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            overflow: hidden;
        }
        .player-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), transparent);
            opacity: 0;
            transition: opacity 0.3s ease;
            pointer-events: none;
        }
        .player-card:hover::before {
            opacity: 1;
        }
        .player-card.active {
            background: linear-gradient(135deg, rgba(255, 215, 0, 0.2) 0%, rgba(255, 215, 0, 0.1) 100%);
            border-color: #ffd700;
            box-shadow:
                0 0 30px rgba(255, 215, 0, 0.3),
                inset 0 1px 0 rgba(255, 215, 0, 0.2);
            transform: translateY(-2px);
        }
        .player-card.dead {
            opacity: 0.5;
            background: linear-gradient(135deg, rgba(255, 71, 87, 0.1) 0%, rgba(99, 110, 114, 0.1) 100%);
            border-color: rgba(255, 71, 87, 0.3);
        }
        .avatar-container {
            position: relative;
            width: 64px;
            height: 64px;
            margin-right: 20px;
            flex-shrink: 0;
        }
        .player-card .avatar {
            width: 100%;
            height: 100%;
            border-radius: 50%;
            object-fit: cover;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: 3px solid rgba(255, 255, 255, 0.3);
            box-shadow:
                0 8px 24px rgba(0, 0, 0, 0.3),
                inset 0 2px 4px rgba(255, 255, 255, 0.2);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .player-card:hover .avatar {
            transform: scale(1.1);
            border-color: rgba(255, 255, 255, 0.6);
            box-shadow:
                0 12px 32px rgba(0, 0, 0, 0.4),
                inset 0 2px 4px rgba(255, 255, 255, 0.3),
                0 0 20px rgba(255, 255, 255, 0.2);
        }
        .player-card.dead .avatar {
             filter: var(--dead-filter);
        }
        .player-info {
            flex-grow: 1;
            overflow: hidden;
            min-width: 0;
        }
        
        .player-name {
            font-weight: 700;
            font-size: 1.4em;
            margin-bottom: 8px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            background: linear-gradient(135deg, #ffffff 0%, #a0a6b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.2;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .player-role {
            font-size: 1em;
            font-weight: 600;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 8px;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(160, 166, 184, 0.8) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .role-icon {
            font-size: 1.3em;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.4));
            text-shadow: none;
            -webkit-text-fill-color: initial;
        }
        
        .player-status {
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 4px 8px;
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.2) 0%, rgba(255, 255, 255, 0.1) 100%);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: var(--night-text);
            backdrop-filter: blur(5px);
            display: inline-block;
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
            border: 2px solid rgba(255, 255, 255, 0.2);
            overflow: hidden;
        }
        
        .threat-indicator::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: conic-gradient(
                from 0deg,
                var(--color-alive) 0%,
                var(--color-alive) calc(var(--threat-level, 0) * 100%),
                transparent calc(var(--threat-level, 0) * 100%)
            );
            border-radius: inherit;
        }
        
        .threat-indicator:hover::after {
            content: attr(data-threat-label);
            position: absolute;
            bottom: 100%;
            right: 50%;
            transform: translateX(50%);
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: var(--spacing-xs) var(--spacing-sm);
            border-radius: var(--radius-sm);
            font-size: 0.75em;
            white-space: nowrap;
            z-index: 1000;
            margin-bottom: var(--spacing-xs);
        }
        #chat-log {
            list-style: none;
            padding: 0;
            margin: 0;
            flex-grow: 1;
            overflow-y: auto;
            scrollbar-width: thin;
            scrollbar-color: rgba(255, 255, 255, 0.3) transparent;
        }
        
        #chat-log::-webkit-scrollbar {
            width: 6px;
        }
        
        #chat-log::-webkit-scrollbar-track {
            background: transparent;
        }
        
        #chat-log::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.3);
            border-radius: 3px;
        }
        
        #chat-log::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.5);
        }
        
        .chat-entry {
            display: flex;
            margin-bottom: var(--spacing-md);
            align-items: flex-start;
            padding: var(--spacing-sm);
            border-radius: var(--radius-md);
            transition: all var(--transition-fast);
        }
        
        .chat-entry:hover {
            background: rgba(255, 255, 255, 0.05);
        }
        
        .chat-avatar {
            width: 44px;
            height: 44px;
            border-radius: var(--radius-full);
            margin-right: var(--spacing-md);
            object-fit: cover;
            flex-shrink: 0;
            border: 2px solid rgba(255, 255, 255, 0.1);
            transition: all var(--transition-base);
        }
        
        .chat-entry:hover .chat-avatar {
            border-color: rgba(255, 255, 255, 0.3);
            box-shadow: var(--shadow-sm);
        }
        
        .message-content {
            display: flex;
            flex-direction: column;
            flex-grow: 1;
            min-width: 0;
        }
        
        .balloon {
            padding: 20px 24px;
            border-radius: 24px 24px 24px 8px;
            max-width: 85%;
            word-wrap: break-word;
            transition: all var(--transition-base);
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0.05) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow:
                0 8px 32px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
            position: relative;
            overflow: hidden;
        }
        .balloon::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), transparent);
            opacity: 0;
            transition: opacity 0.3s ease;
            pointer-events: none;
        }
        .balloon:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow:
                0 12px 40px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.3);
        }
        .balloon:hover::before {
            opacity: 1;
        }
        .chat-entry.event-day .balloon {
            background: linear-gradient(135deg, rgba(116, 185, 255, 0.15) 0%, rgba(9, 132, 227, 0.1) 100%);
            color: var(--night-text);
            border-left: 3px solid var(--color-info);
        }
        
        .chat-entry.event-night .balloon {
            background: linear-gradient(135deg, rgba(44, 62, 80, 0.4) 0%, rgba(52, 73, 94, 0.3) 100%);
            border-left: 3px solid var(--night-text-secondary);
        }
        
        .msg-entry {
            border-left: 4px solid var(--color-warning);
            padding: var(--spacing-md);
            margin: var(--spacing-md) 0;
            border-radius: var(--radius-md);
            transition: all var(--transition-base);
            background: rgba(255, 255, 255, 0.05);
            box-shadow: var(--shadow-sm);
        }
        
        .msg-entry:hover {
            transform: translateX(4px);
            box-shadow: var(--shadow-md);
        }
        
        .msg-entry.event-day {
            background: linear-gradient(135deg, rgba(116, 185, 255, 0.1) 0%, rgba(9, 132, 227, 0.05) 100%);
        }
        
        .msg-entry.event-night {
            background: linear-gradient(135deg, rgba(44, 62, 80, 0.3) 0%, rgba(52, 73, 94, 0.2) 100%);
        }
        
        .msg-entry::before {
            content: '';
            position: absolute;
            left: -4px;
            top: 0;
            bottom: 0;
            width: 4px;
            border-radius: 2px;
            background: inherit;
        }
        .reasoning-text {
            font-size: 0.85em;
            color: #bdc3c7;
            font-style: italic;
            margin-top: 5px;
            padding-left: 10px;
            border-left: 2px solid #555;
        }
        .msg-entry.game-event {
             border-left-color: #e74c3c;
        }
        .msg-entry.game-win {
             border-left-color: #2ecc71;
             line-height: 1.5;
        }
        #chat-log cite {
            font-style: normal;
            font-weight: bold;
            display: block;
            font-size: 0.9em;
            color: #ecf0f1;
            margin-bottom: 5px;
        }
        .moderator-announcement {
            margin: 10px 0;
        }
        .moderator-announcement-content {
            padding: 10px;
            border-radius: 5px;
            transition: background-color 0.3s ease;
            border-left: 5px solid #2ecc71;
            color: var(--night-text);
            line-height: 1.2;
        }
        .moderator-announcement-content.event-day {
            background-color: rgba(236, 240, 241, 0.1);
        }
        .moderator-announcement-content.event-night {
            background-color: rgba(0, 0, 0, 0.25);
        }
        .timestamp {
            font-size: 0.8em;
            color: #bdc3c7;
            margin-left: 10px;
            font-weight: normal;
        }
        .msg-text br {
            display: block; /* makes <br> behave like a block element */
            margin-bottom: 0.5em; /* space below the <br> */
            content: ""; /* required for margin to apply */
        }
        .player-capsule {
            display: inline-flex;
            align-items: center;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1px 8px 1px 2px;
            font-size: 0.9em;
            font-weight: bold;
            margin: 0 2px;
            vertical-align: middle;
        }
        .capsule-avatar {
            width: 18px;
            height: 18px;
            border-radius: 50%;
            margin-right: 5px;
            object-fit: cover;
        }
        .balloon, .msg-entry.clickable, .moderator-announcement-content.clickable {
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }
        .balloon:hover, .msg-entry.clickable:hover, .moderator-announcement-content.clickable:hover {
            transform: scale(1.02);
            box-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
            cursor: pointer;
        }
        .playing-audio {
            color: #87CEFA !important; /* Light Sky Blue */
        }
        .playing-audio * {
            color: #87CEFA !important;
        }
        .playing-audio .balloon,
        .playing-audio.msg-entry,
        .playing-audio .moderator-announcement-content {
            background-color: rgba(135, 206, 250, 0.2) !important;
        }
        .audio-controls {
            padding: var(--spacing-md) 0;
            border-top: 2px solid rgba(255,255,255,0.1);
            margin-top: var(--spacing-md);
            background: rgba(255, 255, 255, 0.02);
            border-radius: var(--radius-md);
            backdrop-filter: blur(5px);
        }
        
        .audio-controls label {
            display: block;
            margin-bottom: var(--spacing-sm);
            font-size: 0.9em;
            color: var(--night-text-secondary);
            font-weight: 500;
        }
        
        .audio-controls input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: rgba(255, 255, 255, 0.2);
            outline: none;
            appearance: none;
            transition: all var(--transition-base);
        }
        
        .audio-controls input[type="range"]::-webkit-slider-thumb {
            appearance: none;
            width: 18px;
            height: 18px;
            border-radius: var(--radius-full);
            background: var(--color-info);
            cursor: pointer;
            transition: all var(--transition-base);
            box-shadow: var(--shadow-sm);
        }
        
        .audio-controls input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2);
            box-shadow: var(--shadow-md);
        }
        
        .audio-controls input[type="range"]::-moz-range-thumb {
            width: 18px;
            height: 18px;
            border-radius: var(--radius-full);
            background: var(--color-info);
            cursor: pointer;
            border: none;
            transition: all var(--transition-base);
        }
        
        #pause-audio {
            background: linear-gradient(135deg, var(--color-info) 0%, var(--color-success) 100%);
            border: 2px solid rgba(255, 255, 255, 0.2);
            border-radius: var(--radius-full);
            width: 36px;
            height: 36px;
            cursor: pointer;
            padding: 0;
            background-size: 60%;
            background-repeat: no-repeat;
            background-position: center;
            transition: all var(--transition-base);
            box-shadow: var(--shadow-sm);
            filter: invert(90%) sepia(10%) saturate(100%) hue-rotate(180deg) brightness(100%) contrast(90%);
        }
        
        #pause-audio:hover {
            transform: scale(1.1);
            box-shadow: var(--shadow-md);
            border-color: rgba(255, 255, 255, 0.4);
        }
        
        #pause-audio:active {
            transform: scale(0.95);
        }
        #pause-audio.paused {
            background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cG9seWdvbiBwb2ludHM9IjYgMyAyMCAxMiA2IDIxIi8+PC9zdmc+');
        }
        #pause-audio.playing {
            background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cmVjdCB4PSI2IiB5PSI0IiB3aWR0aD0iNCIgaGVpZ2h0PSIxNiIgcng9IjEiLz48cmVjdCB4PSIxNCIgeT0iNCIgd2lkdGg9IjQiIGhlaWdodD0iMTYiIHJ4PSIxIi8+PC9zdmc+');
        }
        
        /* ===== RESPONSIVE DESIGN ===== */
        @media (max-width: 1024px) {
            .left-panel {
                width: 280px;
            }
            
            .player-card {
                padding: var(--spacing-sm);
            }
            
            .avatar-container {
                width: 48px;
                height: 48px;
            }
            
            .player-name {
                font-size: 1.1em;
            }
        }
        
        @media (max-width: 768px) {
            .main-container {
                flex-direction: column;
                height: 100vh;
            }
            
            .left-panel {
                width: 100%;
                max-height: 40vh;
                order: 2;
                border-top: 2px solid rgba(255, 255, 255, 0.1);
            }
            
            .right-panel {
                order: 1;
                flex: 1;
                min-height: 0;
            }
            
            .player-card {
                padding: var(--spacing-sm);
                margin-bottom: var(--spacing-sm);
            }
            
            .avatar-container {
                width: 40px;
                height: 40px;
                margin-right: var(--spacing-sm);
            }
            
            .player-name {
                font-size: 1em;
            }
            
            .player-role {
                font-size: 0.8em;
            }
            
            .chat-avatar {
                width: 36px;
                height: 36px;
                margin-right: var(--spacing-sm);
            }
            
            .balloon {
                max-width: 95%;
                padding: var(--spacing-sm);
            }
            
            .audio-controls {
                padding: var(--spacing-sm) 0;
            }
        }
        
        @media (max-width: 480px) {
            .player-card {
                flex-direction: column;
                text-align: center;
                padding: var(--spacing-sm);
            }
            
            .avatar-container {
                margin: 0 auto var(--spacing-sm);
            }
            
            .threat-indicator {
                position: static;
                margin: var(--spacing-xs) auto 0;
                transform: none;
            }
            
            .chat-entry {
                flex-direction: column;
                align-items: center;
                text-align: center;
            }
            
            .chat-avatar {
                margin: 0 0 var(--spacing-sm);
            }
            
            .balloon {
                max-width: 100%;
            }
        }
        
        /* ===== ACCESSIBILITY ===== */
        /* Focus indicators */
        *:focus {
            outline: none;
        }
        
        *:focus-visible {
            outline: 2px solid var(--color-info);
            outline-offset: 2px;
            border-radius: var(--radius-sm);
        }
        
        /* Screen reader only content */
        .sr-only {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }
        
        /* High contrast mode support */
        @media (prefers-contrast: high) {
            :root {
                --night-surface: rgba(0, 0, 0, 0.9);
                --night-text: #ffffff;
                --night-text-secondary: #cccccc;
            }
            
            .player-card {
                border: 2px solid rgba(255, 255, 255, 0.5);
            }
            
            .balloon {
                border: 1px solid rgba(255, 255, 255, 0.5);
            }
        }
        
        /* Reduced motion support */
        @media (prefers-reduced-motion: reduce) {
            * {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
            }
            
            .player-card:hover {
                transform: none;
            }
            
            .balloon:hover {
                transform: none;
            }
            
            .msg-entry:hover {
                transform: none;
            }
        }
        
        /* Keyboard navigation */
        .player-card[tabindex]:focus,
        .balloon[tabindex]:focus,
        .msg-entry[tabindex]:focus {
            outline: 2px solid var(--color-info);
            outline-offset: 2px;
        }
        
        /* Better contrast for links and interactive elements */
        [role="button"]:hover,
        [tabindex]:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        
        /* ===== ENHANCED ANIMATIONS ===== */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        @keyframes slideInLeft {
            from { transform: translateX(-100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes slideInRight {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes bounceIn {
            0% { transform: scale(0.3); opacity: 0; }
            50% { transform: scale(1.05); }
            70% { transform: scale(0.9); }
            100% { transform: scale(1); opacity: 1; }
        }
        
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
            20%, 40%, 60%, 80% { transform: translateX(5px); }
        }
        
        @keyframes glow {
            0%, 100% { box-shadow: 0 0 5px rgba(255, 255, 255, 0.2); }
            50% { box-shadow: 0 0 20px rgba(255, 255, 255, 0.6), 0 0 30px rgba(255, 255, 255, 0.4); }
        }
        
        @keyframes fadeInUp {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        
        @keyframes heartbeat {
            0%, 100% { transform: scale(1); }
            14% { transform: scale(1.1); }
            28% { transform: scale(1); }
            42% { transform: scale(1.1); }
            70% { transform: scale(1); }
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
        
        /* Animation Classes */
        .animate-pulse { animation: pulse 2s infinite; }
        .animate-slide-in-left { animation: slideInLeft 0.5s ease-out; }
        .animate-slide-in-right { animation: slideInRight 0.5s ease-out; }
        .animate-bounce-in { animation: bounceIn 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55); }
        .animate-shake { animation: shake 0.82s cubic-bezier(0.36, 0.07, 0.19, 0.97); }
        .animate-glow { animation: glow 2s ease-in-out infinite; }
        .animate-fade-in-up { animation: fadeInUp 0.6s ease-out; }
        .animate-heartbeat { animation: heartbeat 1.5s ease-in-out infinite; }
        .animate-float { animation: float 3s ease-in-out infinite; }
        
        /* Enhanced Interactive Elements */
        .notification-toast {
            position: fixed;
            top: 20px;
            right: 20px;
            background: linear-gradient(135deg, var(--color-info) 0%, var(--color-success) 100%);
            color: white;
            padding: var(--spacing-md) var(--spacing-lg);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-lg);
            z-index: 1000;
            transform: translateX(100%);
            transition: transform var(--transition-base);
            max-width: 300px;
            word-wrap: break-word;
        }
        
        .notification-toast.show {
            transform: translateX(0);
        }
        
        .notification-toast.error {
            background: linear-gradient(135deg, var(--color-danger) 0%, #c0392b 100%);
        }
        
        .notification-toast.warning {
            background: linear-gradient(135deg, var(--color-warning) 0%, #e67e22 100%);
        }
        
        .particle-effect {
            position: absolute;
            pointer-events: none;
            z-index: 100;
        }
        
        .particle {
            position: absolute;
            width: 4px;
            height: 4px;
            background: radial-gradient(circle, var(--color-info) 0%, transparent 70%);
            border-radius: 50%;
            animation: particleFloat 2s ease-out forwards;
        }
        
        @keyframes particleFloat {
            0% {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
            100% {
                opacity: 0;
                transform: translateY(-100px) scale(0.2);
            }
        }
        
        .voting-indicator {
            position: absolute;
            top: -10px;
            right: -10px;
            width: 24px;
            height: 24px;
            background: linear-gradient(45deg, var(--color-warning) 0%, var(--color-danger) 100%);
            border-radius: var(--radius-full);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8em;
            font-weight: bold;
            color: white;
            animation: heartbeat 1s ease-in-out infinite;
            box-shadow: var(--shadow-md);
        }
        
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: var(--color-info);
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .role-badge {
            position: absolute;
            top: -8px;
            left: -8px;
            width: 20px;
            height: 20px;
            border-radius: var(--radius-full);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7em;
            font-weight: bold;
            border: 2px solid white;
            box-shadow: var(--shadow-sm);
        }
        
        .role-badge.werewolf {
            background: var(--color-werewolf);
            color: white;
        }
        
        .role-badge.villager {
            background: var(--color-villager);
            color: white;
        }
        
        .role-badge.seer {
            background: var(--color-seer);
            color: white;
        }
        
        .role-badge.doctor {
            background: var(--color-doctor);
            color: white;
        }
        
        .interactive-background {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            overflow: hidden;
            z-index: -1;
        }
        
        .background-particle {
            position: absolute;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            animation: backgroundFloat 20s linear infinite;
        }
        
        @keyframes backgroundFloat {
            0% {
                transform: translateY(100vh) rotate(0deg);
                opacity: 0;
            }
            10% {
                opacity: 1;
            }
            90% {
                opacity: 1;
            }
            100% {
                transform: translateY(-100px) rotate(360deg);
                opacity: 0;
            }
        }
        
        .status-indicator {
            position: absolute;
            bottom: -5px;
            right: -5px;
            width: 16px;
            height: 16px;
            border-radius: var(--radius-full);
            border: 2px solid white;
            animation: pulse 2s infinite;
        }
        
        .status-indicator.alive {
            background: var(--color-alive);
        }
        
        .status-indicator.dead {
            background: var(--color-dead);
            animation: none;
        }
        
        .interactive-tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: var(--spacing-sm) var(--spacing-md);
            border-radius: var(--radius-md);
            font-size: 0.8em;
            white-space: nowrap;
            z-index: 1000;
            pointer-events: none;
            opacity: 0;
            transform: translateY(10px);
            transition: all var(--transition-base);
        }
        
        .interactive-tooltip.show {
            opacity: 1;
            transform: translateY(0);
        }
        
        .sound-wave {
            position: absolute;
            top: 50%;
            right: 10px;
            transform: translateY(-50%);
            display: flex;
            gap: 2px;
            opacity: 0;
            transition: opacity var(--transition-base);
        }
        
        .sound-wave.active {
            opacity: 1;
        }
        
        .sound-bar {
            width: 3px;
            height: 12px;
            background: var(--color-info);
            border-radius: 2px;
            animation: soundWave 1s ease-in-out infinite;
        }
        
        .sound-bar:nth-child(1) { animation-delay: 0s; }
        .sound-bar:nth-child(2) { animation-delay: 0.1s; }
        .sound-bar:nth-child(3) { animation-delay: 0.2s; }
        .sound-bar:nth-child(4) { animation-delay: 0.3s; }
        
        @keyframes soundWave {
            0%, 100% { transform: scaleY(1); }
            50% { transform: scaleY(0.3); }
        }
    `;

  // --- TTS Management ---
  const audioMap = window.AUDIO_MAP || {};

  if (!window.kaggleWerewolf) {
      window.kaggleWerewolf = {
          audioQueue: [],
          isAudioPlaying: false,
          isAudioEnabled: false,
          isPaused: false,
          lastPlayedStep: -1,
          audioPlayer: new Audio(),
          playbackRate: 1.0,
      };
  }
  const audioState = window.kaggleWerewolf;

  function togglePause() {
      audioState.isPaused = !audioState.isPaused;
      const pauseButton = parent.querySelector('#pause-audio');
      if (pauseButton) {
          pauseButton.classList.toggle('paused', audioState.isPaused);
          pauseButton.classList.toggle('playing', !audioState.isPaused);
      }
      if (!audioState.isPaused && !audioState.isAudioPlaying) {
          playNextInQueue();
      } else if (audioState.isPaused && audioState.isAudioPlaying) {
          audioState.audioPlayer.pause();
      } else if (!audioState.isPaused && audioState.isAudioPlaying) {
          audioState.audioPlayer.play();
      }
  }

  function setPlaybackRate(rate) {
      audioState.playbackRate = rate;
      if (audioState.isAudioPlaying) {
          audioState.audioPlayer.playbackRate = rate;
      }
  }

  function playNextInQueue() {
      if (audioState.isPaused || audioState.isAudioPlaying || audioState.audioQueue.length === 0 || !audioState.isAudioEnabled) {
          return;
      }

      const previouslyPlaying = parent.querySelector('.playing-audio');
      if (previouslyPlaying) {
          previouslyPlaying.classList.remove('playing-audio');
      }

      audioState.isAudioPlaying = true;
      const event = audioState.audioQueue.shift();
      const audioKey = event.speaker === 'moderator' ? `moderator:${event.message}` : `${event.speaker}:${event.message}`;
      const audioPath = audioMap[audioKey];

      const elementToHighlight = parent.querySelector(`[data-audio-key="${audioKey}"]`);
      if (elementToHighlight) {
          elementToHighlight.classList.add('playing-audio');
      }

      if (audioPath) {
          audioState.audioPlayer.src = audioPath;
          audioState.audioPlayer.playbackRate = audioState.playbackRate;
          audioState.audioPlayer.onended = () => {
              if (elementToHighlight) elementToHighlight.classList.remove('playing-audio');
              audioState.isAudioPlaying = false;
              if (!audioState.isPaused) {
                playNextInQueue();
              }
          };
          audioState.audioPlayer.onerror = () => {
              console.error("Audio playback failed for key:", audioKey);
              if (elementToHighlight) elementToHighlight.classList.remove('playing-audio');
              audioState.isAudioPlaying = false;
              playNextInQueue();
          };
          audioState.audioPlayer.play().catch(e => {
              console.error("Audio playback failed:", e);
              if (elementToHighlight) elementToHighlight.classList.remove('playing-audio');
              audioState.isAudioPlaying = false;
              playNextInQueue();
          });
      } else {
          console.warn(`No audio found for key: "${audioKey}"`);
          if (elementToHighlight) elementToHighlight.classList.remove('playing-audio');
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

  function getAudioEventForEntry(entry) {
    let audioEvent = null;
    if (entry.type === 'chat') {
        audioEvent = { message: entry.message, speaker: entry.speaker };
    } else if (entry.type === 'system') {
        const text = (entry.text || '').toLowerCase();
        let audioKey = null;
        if (text.includes('night') && text.includes('begins')) {
            audioKey = 'night_begins';
        } else if (text.includes('day') && text.includes('begins')) {
            audioKey = 'day_begins';
        } else if (text.includes('discussion')) {
            audioKey = 'discussion_begins';
        } else if (text.includes('voting phase begins')) {
            audioKey = 'voting_begins';
        }
        if (audioKey) {
            audioEvent = { message: audioKey, speaker: 'moderator' };
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
    return audioEvent;
  }

  function speakFrom(startIndex, logEntries) {
      if (!audioState.isAudioEnabled) {
          audioState.isAudioEnabled = true;
          const audio = new Audio('data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA');
          audio.play().catch(e => console.warn("Audio context activation failed:", e));
      }

      // Stop current playback and clear highlight
      if (audioState.isAudioPlaying) {
          audioState.audioPlayer.pause();
          audioState.isAudioPlaying = false;
          const previouslyPlaying = parent.querySelector('.playing-audio');
          if (previouslyPlaying) {
              previouslyPlaying.classList.remove('playing-audio');
          }
      }

      audioState.audioQueue = [];
      if (audioState.isPaused) {
          audioState.isPaused = false;
          const pauseButton = parent.querySelector('#pause-audio');
          if (pauseButton) {
              pauseButton.classList.remove('paused');
              pauseButton.classList.add('playing');
          }
      }

      const eventsToQueue = [];
      for (let i = startIndex; i < logEntries.length; i++) {
          const entry = logEntries[i];
          const audioEvent = getAudioEventForEntry(entry);
          if (audioEvent) {
              eventsToQueue.push(audioEvent);
          }
      }

      audioState.audioQueue = eventsToQueue;
      playNextInQueue();
  }

  // --- Enhanced Interactive Features ---
  
  function showNotification(message, type = 'info', duration = 3000) {
      const toast = document.createElement('div');
      toast.className = `notification-toast ${type}`;
      toast.textContent = message;
      
      document.body.appendChild(toast);
      
      // Trigger animation
      setTimeout(() => toast.classList.add('show'), 100);
      
      // Remove after duration
      setTimeout(() => {
          toast.classList.remove('show');
          setTimeout(() => {
              if (document.body.contains(toast)) {
                  document.body.removeChild(toast);
              }
          }, 300);
      }, duration);
  }
  
  function createParticleEffect(element, count = 8) {
      const rect = element.getBoundingClientRect();
      const container = document.createElement('div');
      container.className = 'particle-effect';
      container.style.left = rect.left + rect.width / 2 + 'px';
      container.style.top = rect.top + rect.height / 2 + 'px';
      
      for (let i = 0; i < count; i++) {
          const particle = document.createElement('div');
          particle.className = 'particle';
          
          const angle = (360 / count) * i;
          const distance = 50 + Math.random() * 30;
          const x = Math.cos(angle * Math.PI / 180) * distance;
          const y = Math.sin(angle * Math.PI / 180) * distance;
          
          particle.style.left = '0px';
          particle.style.top = '0px';
          particle.style.transform = `translate(${x}px, ${y}px)`;
          
          container.appendChild(particle);
      }
      
      document.body.appendChild(container);
      
      // Remove after animation
      setTimeout(() => {
          if (document.body.contains(container)) {
              document.body.removeChild(container);
          }
      }, 2000);
  }
  
  function addInteractiveTooltip(element, text) {
      const tooltip = document.createElement('div');
      tooltip.className = 'interactive-tooltip';
      tooltip.textContent = text;
      document.body.appendChild(tooltip);
      
      element.addEventListener('mouseenter', (e) => {
          const rect = element.getBoundingClientRect();
          tooltip.style.left = rect.left + rect.width / 2 + 'px';
          tooltip.style.top = rect.top - 10 + 'px';
          tooltip.style.transform = 'translateX(-50%) translateY(-100%)';
          tooltip.classList.add('show');
      });
      
      element.addEventListener('mouseleave', () => {
          tooltip.classList.remove('show');
      });
      
      // Cleanup function
      return () => {
          if (document.body.contains(tooltip)) {
              document.body.removeChild(tooltip);
          }
      };
  }
  
  function createBackgroundParticles() {
      if (parent.querySelector('.interactive-background')) return;
      
      const background = document.createElement('div');
      background.className = 'interactive-background';
      
      // Create floating particles
      for (let i = 0; i < 15; i++) {
          const particle = document.createElement('div');
          particle.className = 'background-particle';
          
          const size = Math.random() * 8 + 4;
          particle.style.width = size + 'px';
          particle.style.height = size + 'px';
          particle.style.left = Math.random() * 100 + '%';
          particle.style.animationDelay = Math.random() * 20 + 's';
          particle.style.animationDuration = (15 + Math.random() * 10) + 's';
          
          background.appendChild(particle);
      }
      
      parent.appendChild(background);
  }
  
  function addSoundWaveIndicator(element) {
      const soundWave = document.createElement('div');
      soundWave.className = 'sound-wave';
      
      for (let i = 0; i < 4; i++) {
          const bar = document.createElement('div');
          bar.className = 'sound-bar';
          soundWave.appendChild(bar);
      }
      
      element.style.position = 'relative';
      element.appendChild(soundWave);
      
      return soundWave;
  }
  
  function animatePlayerCard(card, animationType = 'bounce-in') {
      card.classList.add(`animate-${animationType}`);
      card.addEventListener('animationend', () => {
          card.classList.remove(`animate-${animationType}`);
      }, { once: true });
  }
  
  function addHoverEffects(element) {
      element.addEventListener('mouseenter', () => {
          element.style.transform = 'translateY(-2px) scale(1.02)';
          element.style.boxShadow = 'var(--shadow-lg)';
      });
      
      element.addEventListener('mouseleave', () => {
          element.style.transform = '';
          element.style.boxShadow = '';
      });
  }
  
  function createLoadingSpinner() {
      const spinner = document.createElement('div');
      spinner.className = 'loading-spinner';
      return spinner;
  }
  
  function addRoleBadge(avatarContainer, role) {
      if (role === 'Unknown') return;
      
      const badge = document.createElement('div');
      badge.className = `role-badge ${role.toLowerCase()}`;
      
      let badgeContent = '';
      switch(role) {
          case 'Werewolf': badgeContent = '🐺'; break;
          case 'Villager': badgeContent = '🧑'; break;
          case 'Seer': badgeContent = '🔮'; break;
          case 'Doctor': badgeContent = '🩺'; break;
      }
      
      badge.textContent = badgeContent;
      avatarContainer.appendChild(badge);
  }
  
  function addStatusIndicator(avatarContainer, isAlive) {
      const indicator = document.createElement('div');
      indicator.className = `status-indicator ${isAlive ? 'alive' : 'dead'}`;
      avatarContainer.appendChild(indicator);
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
              const regex = new RegExp(`\b${playerId.replace(/[-\/\\^$*+?.()|[\\]{}/g, '\\$&')}\b`, 'g');
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
        const regex = new RegExp(`\b${playerId.replace(/[-\/\\^$*+?.()|[\\]{}/g, '\\$&')}\b`, 'g');
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

  function renderPlayerList(container, gameState, actingPlayerName) {
    container.innerHTML = '<h1>Players</h1>';
    const listContainer = document.createElement('div');
    listContainer.id = 'player-list-container';
    const playerUl = document.createElement('ul');
    playerUl.id = 'player-list';

    // Initialize animation flag if it doesn't exist
    if (!window.werewolfAnimationsShown) {
        window.werewolfAnimationsShown = {
            playerListEntrance: false,
            panelEntrance: false
        };
    }

    gameState.players.forEach((player, index) => {
        const li = document.createElement('li');
        li.className = 'player-card';
        if (!player.is_alive) li.classList.add('dead');
        if (player.name === actingPlayerName) li.classList.add('active');

        let roleDisplay = player.role;
        if (player.role === 'Werewolf') {
            roleDisplay = `\uD83D\uDC3A ${player.role}`;
        } else if (player.role === 'Doctor') {
            roleDisplay = `\uD83E\uDE7A ${player.role}`;
        } else if (player.role === 'Seer') {
            roleDisplay = `\uD83D\uDD2E ${player.role}`;
        } else if (player.role === 'Villager') {
            roleDisplay = `\uD83E\uDDD1 ${player.role}`;
        }

        const roleText = player.role !== 'Unknown' ? `Role: ${roleDisplay}` : 'Role: Unknown';

        // Set accessibility attributes
        li.setAttribute('data-role', player.role);
        li.setAttribute('role', 'listitem');
        li.setAttribute('tabindex', '0');
        
        const statusText = player.is_alive ? 'Alive' : (player.status || 'Dead');
        const threatLevel = gameState.playerThreatLevels.get(player.name) || 0;
        const threatLabels = ['Safe', 'Uneasy', 'In Danger'];
        const threatLabel = threatLabels[Math.floor(threatLevel * 2)] || 'Safe';
        
        // Set ARIA label for the entire card
        const ariaLabel = `${player.name}, ${player.role}, ${statusText}${threatLevel > 0 ? `, threat level: ${threatLabel}` : ''}`;
        li.setAttribute('aria-label', ariaLabel);

        li.innerHTML = `
            <div class="avatar-container">
                <img src="${player.thumbnail}" alt="${player.name} avatar" class="avatar">
            </div>
            <div class="player-info">
                <div class="player-name" title="${player.name}">${player.name}</div>
                <div class="player-role">${roleText}</div>
                <div class="player-status">${statusText}</div>
            </div>
            <div class="threat-indicator"
                 style="--threat-level: ${threatLevel}"
                 data-threat-label="${threatLabel}"
                 aria-label="Threat level: ${threatLabel}"
                 role="img"></div>
        `;
        
        // Add enhanced interactive features
        const avatarContainer = li.querySelector('.avatar-container');
        addRoleBadge(avatarContainer, player.role);
        addStatusIndicator(avatarContainer, player.is_alive);
        
        // Add hover effects
        addHoverEffects(li);
        
        // Add entrance animation only once at the beginning
        if (!window.werewolfAnimationsShown.playerListEntrance) {
            setTimeout(() => animatePlayerCard(li, 'slide-in-left'), index * 100);
        }
        
        // Add interactive tooltip
        addInteractiveTooltip(li, `${player.name} - ${player.role} (${statusText})`);
        
        // Add click effects
        li.addEventListener('click', () => {
            createParticleEffect(li);
            if (player.is_alive) {
                showNotification(`Selected ${player.name}`, 'info', 2000);
            }
        });
        
        playerUl.appendChild(li);
    });

    // Mark player list entrance animation as shown
    if (!window.werewolfAnimationsShown.playerListEntrance) {
        window.werewolfAnimationsShown.playerListEntrance = true;
    }

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
        } else {
            indicator.style.backgroundColor = 'transparent';
        }
    });

    // --- Audio Controls ---
    const audioControls = document.createElement('div');
    audioControls.className = 'audio-controls';
    const pauseButtonClass = audioState.isPaused ? 'paused' : 'playing';
    audioControls.innerHTML = `
        <label for="playback-speed">Audio Speed: <span id="speed-label">${audioState.playbackRate.toFixed(1)}</span>x</label>
        <div style="display: flex; align-items: center; gap: 10px; margin-top: 5px;">
            <input type="range" 
                   id="playback-speed" 
                   min="0.5" 
                   max="2.5" 
                   step="0.1" 
                   value="${audioState.playbackRate}" 
                   style="flex-grow: 1;"
                   aria-label="Audio playback speed"
                   aria-describedby="speed-label">
            <button id="pause-audio" 
                    class="${pauseButtonClass}"
                    aria-label="${audioState.isPaused ? 'Play audio' : 'Pause audio'}"
                    aria-pressed="${!audioState.isPaused}"
                    type="button"></button>
        </div>
    `;
    container.appendChild(audioControls);

    const speedSlider = audioControls.querySelector('#playback-speed');
    const speedLabel = audioControls.querySelector('#speed-label');
    const pauseButton = audioControls.querySelector('#pause-audio');

    speedSlider.addEventListener('input', (e) => {
        const newRate = parseFloat(e.target.value);
        setPlaybackRate(newRate);
        speedLabel.textContent = newRate.toFixed(1);
    });

    pauseButton.addEventListener('click', () => {
        togglePause();
    });
  }

  function renderEventLog(container, gameState, playerMap) {
    container.innerHTML = '<h1>Event Log</h1>';
    const logUl = document.createElement('ul');
    logUl.id = 'chat-log';

    const logEntries = gameState.eventLog;

    if (logEntries.length === 0) {
        const li = document.createElement('li');
        li.className = 'msg-entry';
        li.innerHTML = `<cite>System</cite><div>The game is about to begin...</div>`;
        logUl.appendChild(li);
    } else {
        logEntries.forEach((entry, index) => {
            const li = document.createElement('li');
            const audioEvent = getAudioEventForEntry(entry);
            if (audioEvent) {
                const audioKey = audioEvent.speaker === 'moderator' ? `moderator:${audioEvent.message}` : `${audioEvent.speaker}:${audioEvent.message}`;
                li.setAttribute('data-audio-key', audioKey);
            }
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
                phaseEmoji = '\u2600\uFE0F'; // Sun emoji
            } else if (phase === 'NIGHT') {
                phaseEmoji = '\uD83C\uDF19'; // Crescent moon emoji
            }

            const dayPhaseString = entry.day !== Infinity ? `[D${entry.day} ${phaseEmoji}]` : '';
            const timestampHtml = `<span class="timestamp">${dayPhaseString} ${formatTimestamp(entry.timestamp)}</span>`;

            switch (entry.type) {
                case 'chat':
                    const speaker = playerMap.get(entry.speaker);
                    if (!speaker) return;
                    const messageText = replacePlayerIdsWithBold(entry.message, entry.mentioned_player_ids);
                    li.className = `chat-entry event-day`;
                    li.innerHTML = `
                        <img src="${speaker.thumbnail}" alt="${speaker.name}" class="chat-avatar">
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
                    const balloon = li.querySelector('.balloon');
                    if (balloon) {
                        balloon.onclick = () => speakFrom(index, logEntries);
                    }
                    break;
                case 'seer_inspection':
                    const seerInspector = playerMap.get(entry.actor_id);
                    if (!seerInspector) return;
                    const seerTargetCap = createPlayerCapsule(playerMap.get(entry.target));
                    const seerCap = createPlayerCapsule(playerMap.get(entry.actor_id));
                    li.className = `chat-entry event-night`;
                    li.innerHTML = `
                        <img src="${seerInspector.thumbnail}" alt="${seerInspector.name}" class="chat-avatar">
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
                    const listRegex = /\\\[(.*?)\\\]/g;
                    systemText = systemText.replace(listRegex, (match, listContent) => {
                        // listContent is "'player-1', 'player-2', 'player-3'"
                        return listContent.replace(/'/g, "").replace(/, /g, " "); // Becomes "player-1 player-2 player-3"
                    });

                    const allPlayerIdsForSystem = Array.from(playerMap.keys());
                    const finalSystemText = replacePlayerIdsWithCapsules(systemText, allPlayerIdsForSystem, playerMap);

                    li.className = `moderator-announcement`;
                    li.innerHTML = `
                        <cite>Moderator \u{1F4E2} ${timestampHtml}</cite>
                        <div class="moderator-announcement-content ${phaseClass}">
                            <div class="msg-text">${finalSystemText.replace(/\n/g, '<br>')}</div>
                        </div>
                    `;
                    const announcementContent = li.querySelector('.moderator-announcement-content');
                    if (announcementContent) {
                        const text = (entry.text || '').toLowerCase();
                        let audioKey = null;
                        if (text.includes('night') && text.includes('begins')) {
                            audioKey = 'night_begins';
                        } else if (text.includes('day') && text.includes('begins')) {
                            audioKey = 'day_begins';
                        } else if (text.includes('discussion')) {
                            audioKey = 'discussion_begins';
                        } else if (text.includes('voting phase begins')) {
                            audioKey = 'voting_begins';
                        }

                        if (audioKey) {
                            announcementContent.classList.add('clickable');
                            announcementContent.onclick = () => speakFrom(index, logEntries);
                        }
                    }
                    break;
                case 'exile':
                    const exiledPlayerCap = createPlayerCapsule(playerMap.get(entry.name));
                    li.className = `msg-entry game-event event-day clickable`;
                    li.innerHTML = `<cite>Exile ${timestampHtml}</cite><div class="msg-text">${exiledPlayerCap} (${entry.role}) was exiled by vote.</div>`;
                    li.onclick = () => speakFrom(index, logEntries);
                    break;
                case 'elimination':
                    const elimPlayerCap = createPlayerCapsule(playerMap.get(entry.name));
                    li.className = `msg-entry game-event event-night clickable`;
                    li.innerHTML = `<cite>Elimination ${timestampHtml}</cite><div class="msg-text">${elimPlayerCap} was eliminated. Their role was a ${entry.role}.</div>`;
                    li.onclick = () => speakFrom(index, logEntries);
                    break;
                case 'save':
                     const savedPlayerCap = createPlayerCapsule(playerMap.get(entry.saved_player));
                     li.className = `msg-entry event-night clickable`;
                     li.innerHTML = `<cite>Doctor Save ${timestampHtml}</cite><div class="msg-text">Player ${savedPlayerCap} was attacked but saved by a Doctor!</div>`;
                     li.onclick = () => speakFrom(index, logEntries);
                    break;
                case 'vote':
                    const voter = playerMap.get(entry.actor_id);
                    if (!voter) return;
                    const voterCap = createPlayerCapsule(playerMap.get(entry.actor_id));
                    const voteTargetCap = createPlayerCapsule(playerMap.get(entry.target));
                    li.className = `chat-entry event-day`;
                    li.innerHTML = `
                        <img src="${voter.thumbnail}" alt="${voter.name}" class="chat-avatar">
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
                    li.className = `msg-entry game-win ${phaseClass} clickable`;
                    li.innerHTML = `
                        <cite>Game Over ${timestampHtml}</cite>
                        <div class="msg-text">
                            <div>The <strong>${entry.winner}</strong> team has won!</div><br>
                            <div><strong>Winning Team:</strong> ${winnersText}</div>
                            <div><strong>Losing Team:</strong> ${losersText}</div>
                        </div>
                    `;
                    li.onclick = () => speakFrom(index, logEntries);
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

    if (!environment || !environment.steps || environment.steps.length === 0 || step >= environment.steps.length) {
        const tempContainer = document.createElement("div");
        tempContainer.textContent = "Waiting for game data or invalid step...";
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
        const tempContainer = document.createElement("div");
        tempContainer.textContent = "Waiting for game data: No players found in observation or configuration.";
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
                        gameState.eventLog.push({ type: 'timeout', step: s, day: historyEvent.day, phase: historyEvent.phase, actor_id: actor_id, reasoning: "Timed out", timestamp: historyEvent.created_at });
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

    Object.assign(parent.style, { width: `${width}px`, height: `${height}px` });
    parent.className = 'werewolf-parent';

    const style = document.createElement('style');
    style.textContent = css;
    parent.appendChild(style);

    const isNight = (gameState.game_state_phase || '').toLowerCase() === 'night';
    updateBackground(isNight);

    const mainContainer = document.createElement('div');
    mainContainer.className = 'main-container';
    parent.appendChild(mainContainer);

    const leftPanel = document.createElement('div');
    leftPanel.className = 'left-panel';
    mainContainer.appendChild(leftPanel);
    const playerListArea = document.createElement('div');
    playerListArea.id = 'player-list-area';
    leftPanel.appendChild(playerListArea);

    const rightPanel = document.createElement('div');
    rightPanel.className = 'right-panel';
    mainContainer.appendChild(rightPanel);

    renderPlayerList(playerListArea, gameState, playerMap, actingPlayerName);
    renderEventLog(rightPanel, gameState, playerMap);
    
    // Add enhanced interactive features
    createBackgroundParticles();
    
    // Add sound wave indicators to playing audio elements
    const playingElements = parent.querySelectorAll('.playing-audio');
    playingElements.forEach(element => {
        if (!element.querySelector('.sound-wave')) {
            const soundWave = addSoundWaveIndicator(element);
            soundWave.classList.add('active');
        }
    });
    
    // Add entrance animations to panels only once
    if (!window.werewolfAnimationsShown.panelEntrance) {
        setTimeout(() => {
            leftPanel.classList.add('animate-slide-in-left');
            rightPanel.classList.add('animate-slide-in-right');
            window.werewolfAnimationsShown.panelEntrance = true;
        }, 100);
    }
    
    // Show game status notifications
    if (gameState.gameWinner) {
        showNotification(`🎉 ${gameState.gameWinner} team wins!`, 'success', 5000);
    } else if (gameState.phase === 'NIGHT') {
        showNotification('🌙 Night phase - Roles take action', 'info', 3000);
    } else if (gameState.phase === 'DAY') {
        showNotification('☀️ Day phase - Discussion and voting', 'info', 3000);
    }
}