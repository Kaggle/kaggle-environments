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
      demo: null,
    };
  }
  const threeState = window.werewolfThreeJs;

  function initThreeJs() {
    if (threeState.initialized) {
      if (threeState.demo && threeState.demo._parent && !parent.contains(threeState.demo._parent)) {
          parent.appendChild(threeState.demo._threejs.domElement);
          parent.appendChild(threeState.demo._labelRenderer.domElement);
      }
      return;
    }

    const loadAndSetup = async () => {
        try {
            const THREE = await import('https://cdn.jsdelivr.net/npm/three@0.118/build/three.module.js');
            const { OrbitControls } = await import('https://cdn.jsdelivr.net/npm/three@0.118/examples/jsm/controls/OrbitControls.js');
            const { FBXLoader } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/loaders/FBXLoader.js');
            const { SkeletonUtils } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/utils/SkeletonUtils.js');
            const { CSS2DRenderer, CSS2DObject } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/renderers/CSS2DRenderer.js');
            const { EffectComposer } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/postprocessing/EffectComposer.js');
            const { RenderPass } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/postprocessing/RenderPass.js');
            const { UnrealBloomPass } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/postprocessing/UnrealBloomPass.js');
            const { ShaderPass } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/postprocessing/ShaderPass.js');
            const { FilmPass } = await import('https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/postprocessing/FilmPass.js');

            class BasicWorldDemo {
              constructor(options) {
                this._Initialize(options, THREE, OrbitControls, FBXLoader, SkeletonUtils, CSS2DRenderer, CSS2DObject, EffectComposer, RenderPass, UnrealBloomPass, ShaderPass, FilmPass);
              }

              _Initialize(options, THREE, OrbitControls, FBXLoader, SkeletonUtils, CSS2DRenderer, CSS2DObject, EffectComposer, RenderPass, UnrealBloomPass, ShaderPass, FilmPass) {
                this._parent = options.parent;
                this._width = options.width;
                this._height = options.height;

                // WebGL Renderer with enhanced settings
                this._threejs = new THREE.WebGLRenderer({
                  antialias: true,
                  alpha: true,
                  powerPreference: "high-performance"
                });
                this._threejs.shadowMap.enabled = true;
                this._threejs.shadowMap.type = THREE.PCFSoftShadowMap;
                this._threejs.shadowMap.autoUpdate = true;
                this._threejs.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                this._threejs.setSize(this._width, this._height);
                this._threejs.outputEncoding = THREE.sRGBEncoding;
                this._threejs.toneMapping = THREE.ACESFilmicToneMapping;
                this._threejs.toneMappingExposure = 1.2;
                this._threejs.domElement.style.position = 'absolute';
                this._threejs.domElement.style.top = '0';
                this._threejs.domElement.style.left = '0';
                this._threejs.domElement.style.zIndex = '0';
                this._parent.appendChild(this._threejs.domElement);

                // CSS2D Renderer
                this._labelRenderer = new CSS2DRenderer();
                this._labelRenderer.setSize(this._width, this._height);
                this._labelRenderer.domElement.style.position = 'absolute';
                this._labelRenderer.domElement.style.top = '0px';
                this._labelRenderer.domElement.style.left = '0px';
                this._labelRenderer.domElement.style.zIndex = '1'; // On top of 3D, behind UI
                this._labelRenderer.domElement.style.pointerEvents = 'none';
                this._parent.appendChild(this._labelRenderer.domElement);

                const fov = 60;
                const aspect = this._width / this._height;
                const near = 1.0;
                const far = 100000.0;
                this._camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
                this._camera.position.set(0, 0, 50);

                this._scene = new THREE.Scene();
                this._scene.fog = new THREE.FogExp2(0x0a0a1a, 0.01);

                this._createSkybox(THREE);
                this._createAdvancedLighting(THREE);
                this._setupPostProcessing(THREE, EffectComposer, RenderPass, UnrealBloomPass, ShaderPass, FilmPass);

                this._controls = new OrbitControls(this._camera, this._threejs.domElement);
                this._controls.target.set(0, 0, 0);
                this._controls.update();

                this._LoadModels(THREE, FBXLoader, SkeletonUtils, CSS2DObject);
                this._RAF();
              }

              _createSkybox(THREE) {
                const skyboxSize = 1000;
                const skyboxGeo = new THREE.BoxGeometry(skyboxSize, skyboxSize, skyboxSize);
                const black = 0x000000;

                const backPanelMaterial = new THREE.MeshBasicMaterial({ side: THREE.BackSide });

                const materials = [
                    new THREE.MeshBasicMaterial({ color: black, side: THREE.BackSide }), // right
                    new THREE.MeshBasicMaterial({ color: black, side: THREE.BackSide }), // left
                    new THREE.MeshBasicMaterial({ color: black, side: THREE.BackSide }), // top
                    new THREE.MeshBasicMaterial({ color: black, side: THREE.BackSide }), // bottom
                    backPanelMaterial,
                    new THREE.MeshBasicMaterial({ color: black, side: THREE.BackSide })
                ];

                const skybox = new THREE.Mesh(skyboxGeo, materials);
                this._scene.add(skybox);

                const backCanvas = document.createElement('canvas');
                const backContext = backCanvas.getContext('2d');

                const canvasSize = 2048
                backCanvas.width = canvasSize;
                backCanvas.height = canvasSize;

                const moonImage = new Image();
                moonImage.onload = () => {
                    backContext.fillStyle = 'black';
                    backContext.fillRect(0, 0, backCanvas.width, backCanvas.height);
                    const moonSize = 500;
                    backContext.drawImage(moonImage, moonSize, moonSize, moonSize, moonSize);

                    const backTexture = new THREE.CanvasTexture(backCanvas);

                    backPanelMaterial.map = backTexture;
                    backPanelMaterial.needsUpdate = true;
                };
                moonImage.onerror = () => {
                    console.error("Failed to load moon texture for skybox.");
                };
                moonImage.src = 'assets/moon4.png';
              }

              _createAdvancedLighting(THREE) {
                // Enhanced ambient lighting with color variation
                const ambientLight = new THREE.AmbientLight(0x404080, 0.4);
                ambientLight.name = 'ambientLight';
                this._scene.add(ambientLight);

                // Main directional light (moon/sun)
                const mainLight = new THREE.DirectionalLight(0xffffff, 1.8);
                mainLight.position.set(30, 50, 20);
                mainLight.castShadow = true;
                mainLight.shadow.mapSize.width = 2048;
                mainLight.shadow.mapSize.height = 2048;
                mainLight.shadow.camera.near = 0.5;
                mainLight.shadow.camera.far = 100;
                mainLight.shadow.camera.left = -50;
                mainLight.shadow.camera.right = 50;
                mainLight.shadow.camera.top = 50;
                mainLight.shadow.camera.bottom = -50;
                mainLight.shadow.bias = -0.001;
                mainLight.shadow.normalBias = 0.02;
                this._scene.add(mainLight);

                // Rim light for dramatic effect
                const rimLight = new THREE.DirectionalLight(0x8080ff, 0.8);
                rimLight.position.set(-20, 10, -30);
                this._scene.add(rimLight);

                // Atmospheric hemisphere light
                const hemiLight = new THREE.HemisphereLight(0x87ceeb, 0x1e1e3f, 0.6);
                this._scene.add(hemiLight);

                // Ground fill light
                const fillLight = new THREE.DirectionalLight(0x404080, 0.3);
                fillLight.position.set(0, -1, 0);
                this._scene.add(fillLight);

                // Store references for phase updates
                this._mainLight = mainLight;
                this._rimLight = rimLight;
                this._hemiLight = hemiLight;
                this._fillLight = fillLight;
              }

              _createMysticalCircles(THREE, radius) {
                // Create multiple concentric circles with mystical patterns
                for (let i = 0; i < 3; i++) {
                  const circleRadius = radius - (i * 2) - 1;
                  const circleGeometry = new THREE.RingGeometry(circleRadius - 0.1, circleRadius + 0.1, 64);
                  const circleMaterial = new THREE.MeshStandardMaterial({
                    color: new THREE.Color().setHSL(0.6 + i * 0.1, 0.8, 0.3 + i * 0.1),
                    emissive: new THREE.Color().setHSL(0.6 + i * 0.1, 0.5, 0.1),
                    emissiveIntensity: 0.2,
                    transparent: true,
                    opacity: 0.6 - i * 0.1,
                    side: THREE.DoubleSide
                  });
                  
                  const circle = new THREE.Mesh(circleGeometry, circleMaterial);
                  circle.rotation.x = -Math.PI / 2;
                  circle.position.y = 0.01 + i * 0.001;
                  this._scene.add(circle);
                }

                // Add runic symbols around the outer circle
                this._createRunicSymbols(THREE, radius);
              }

              _createRunicSymbols(THREE, radius) {
                const symbolCount = 8;
                const symbolGeometry = new THREE.PlaneGeometry(1, 1);
                
                for (let i = 0; i < symbolCount; i++) {
                  const angle = (i / symbolCount) * Math.PI * 2;
                  const x = (radius + 3) * Math.sin(angle);
                  const z = (radius + 3) * Math.cos(angle);
                  
                  // Create a simple runic-like pattern using canvas
                  const canvas = document.createElement('canvas');
                  canvas.width = 64;
                  canvas.height = 64;
                  const ctx = canvas.getContext('2d');
                  
                  ctx.fillStyle = 'rgba(100, 150, 255, 0.8)';
                  ctx.font = '40px serif';
                  ctx.textAlign = 'center';
                  ctx.fillText(['ᚠ', 'ᚡ', 'ᚢ', 'ᚣ', 'ᚤ', 'ᚥ', 'ᚦ', 'ᚧ'][i], 32, 45);
                  
                  const symbolMaterial = new THREE.MeshBasicMaterial({
                    map: new THREE.CanvasTexture(canvas),
                    transparent: true,
                    alphaTest: 0.1
                  });
                  
                  const symbol = new THREE.Mesh(symbolGeometry, symbolMaterial);
                  symbol.position.set(x, 0.1, z);
                  symbol.rotation.x = -Math.PI / 2;
                  symbol.rotation.z = angle;
                  this._scene.add(symbol);
                }
              }

              _createParticleSystem(THREE) {
                // Create floating mystical particles
                const particleCount = 150;
                const particles = new THREE.BufferGeometry();
                const positions = new Float32Array(particleCount * 3);
                const colors = new Float32Array(particleCount * 3);
                const sizes = new Float32Array(particleCount);
                
                for (let i = 0; i < particleCount; i++) {
                  const i3 = i * 3;
                  
                  // Random position within a larger area
                  positions[i3] = (Math.random() - 0.5) * 80;
                  positions[i3 + 1] = Math.random() * 30 + 5;
                  positions[i3 + 2] = (Math.random() - 0.5) * 80;
                  
                  // Mystical colors (blues, purples, greens)
                  const hue = Math.random() * 0.3 + 0.5; // 0.5-0.8 range
                  const color = new THREE.Color().setHSL(hue, 0.8, 0.6);
                  colors[i3] = color.r;
                  colors[i3 + 1] = color.g;
                  colors[i3 + 2] = color.b;
                  
                  sizes[i] = Math.random() * 2 + 0.5;
                }
                
                particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                particles.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                particles.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
                
                const particleMaterial = new THREE.ShaderMaterial({
                  uniforms: {
                    time: { value: 0 }
                  },
                  vertexShader: `
                    attribute float size;
                    attribute vec3 color;
                    varying vec3 vColor;
                    uniform float time;
                    
                    void main() {
                      vColor = color;
                      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                      gl_PointSize = size * (300.0 / -mvPosition.z) * (1.0 + sin(time * 2.0 + position.x * 0.1) * 0.3);
                      gl_Position = projectionMatrix * mvPosition;
                    }
                  `,
                  fragmentShader: `
                    varying vec3 vColor;
                    
                    void main() {
                      float dist = distance(gl_PointCoord, vec2(0.5));
                      if (dist > 0.5) discard;
                      
                      float alpha = 1.0 - (dist * 2.0);
                      alpha *= alpha; // Softer edges
                      
                      gl_FragColor = vec4(vColor, alpha * 0.6);
                    }
                  `,
                  transparent: true,
                  blending: THREE.AdditiveBlending,
                  depthWrite: false
                });
                
                this._particles = new THREE.Points(particles, particleMaterial);
                this._scene.add(this._particles);
                this._particleMaterial = particleMaterial;
              }

              _setupPostProcessing(THREE, EffectComposer, RenderPass, UnrealBloomPass, ShaderPass, FilmPass) {
                // Create effect composer
                this._composer = new EffectComposer(this._threejs);
                
                // Render pass
                const renderPass = new RenderPass(this._scene, this._camera);
                this._composer.addPass(renderPass);

                // Bloom pass for glowing effects
                const bloomPass = new UnrealBloomPass(
                  new THREE.Vector2(this._width, this._height),
                  0.5,  // strength
                  0.8,  // radius
                  0.3   // threshold
                );
                this._composer.addPass(bloomPass);

                // Film grain for atmosphere
                const filmPass = new FilmPass(
                  0.15,  // noise intensity
                  0.1,   // scanline intensity
                  0,     // scanline count
                  false  // grayscale
                );
                this._composer.addPass(filmPass);

                // Custom atmospheric shader
                const atmosphereShader = {
                  uniforms: {
                    'tDiffuse': { value: null },
                    'time': { value: 0.0 },
                    'phase': { value: 0.0 } // 0 = day, 1 = night
                  },
                  vertexShader: `
                    varying vec2 vUv;
                    void main() {
                      vUv = uv;
                      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                    }
                  `,
                  fragmentShader: `
                    uniform sampler2D tDiffuse;
                    uniform float time;
                    uniform float phase;
                    varying vec2 vUv;
                    
                    void main() {
                      vec4 color = texture2D(tDiffuse, vUv);
                      
                      // Add subtle color grading based on phase
                      if (phase > 0.5) {
                        // Night - add blue tint and increase contrast
                        color.rgb = mix(color.rgb, color.rgb * vec3(0.8, 0.9, 1.2), 0.3);
                        color.rgb = pow(color.rgb, vec3(1.1));
                      } else {
                        // Day - add warm tint
                        color.rgb = mix(color.rgb, color.rgb * vec3(1.1, 1.05, 0.95), 0.2);
                      }
                      
                      // Add subtle vignette
                      vec2 center = vec2(0.5, 0.5);
                      float dist = distance(vUv, center);
                      float vignette = 1.0 - smoothstep(0.3, 0.8, dist);
                      color.rgb *= mix(0.7, 1.0, vignette);
                      
                      gl_FragColor = color;
                    }
                  `
                };

                this._atmospherePass = new ShaderPass(atmosphereShader);
                this._composer.addPass(this._atmospherePass);

                // Store references
                this._bloomPass = bloomPass;
                this._filmPass = filmPass;
              }

              _LoadModels(THREE, FBXLoader, SkeletonUtils, CSS2DObject) {
                this._playerObjects = new Map();
                this._playerGroup = new THREE.Group();
                this._playerGroup.name = 'playerGroup';

                // Create enhanced ground circle with better materials
                const radius = 15;
                const groundGeometry = new THREE.CircleGeometry(radius + 5, 64);
                const groundMaterial = new THREE.MeshStandardMaterial({
                    color: 0x1a1a2a,
                    roughness: 0.9,
                    metalness: 0.1,
                    normalScale: new THREE.Vector2(0.5, 0.5),
                    transparent: true,
                    opacity: 0.95
                });
                
                // Add a subtle normal map pattern
                const canvas = document.createElement('canvas');
                canvas.width = 512;
                canvas.height = 512;
                const ctx = canvas.getContext('2d');
                const imageData = ctx.createImageData(512, 512);
                for (let i = 0; i < imageData.data.length; i += 4) {
                    const noise = Math.random() * 0.1 + 0.5;
                    imageData.data[i] = Math.floor(noise * 255);     // R
                    imageData.data[i + 1] = Math.floor(noise * 255); // G
                    imageData.data[i + 2] = 255;                     // B
                    imageData.data[i + 3] = 255;                     // A
                }
                ctx.putImageData(imageData, 0, 0);
                groundMaterial.normalMap = new THREE.CanvasTexture(canvas);
                
                const ground = new THREE.Mesh(groundGeometry, groundMaterial);
                ground.rotation.x = -Math.PI / 2;
                ground.position.y = -0.1;
                ground.receiveShadow = true;
                this._scene.add(ground);

                // Add mystical circle patterns
                this._createMysticalCircles(THREE, radius);

                this._scene.add(this._playerGroup);

                // Store references for later use
                this._THREE = THREE;
                this._CSS2DObject = CSS2DObject;
                
                // Create particle system for atmosphere
                this._createParticleSystem(THREE);
                
                // Frame the empty group initially with better camera positioning
                this._camera.position.set(25, 30, 35);
                this._controls.target.set(0, 8, 0);
                this._controls.enableDamping = true;
                this._controls.dampingFactor = 0.05;
                this._controls.minDistance = 20;
                this._controls.maxDistance = 80;
                this._controls.maxPolarAngle = Math.PI * 0.75;
                this._controls.update();
              }

              updatePlayerStatus(playerName, status) {
                const player = this._playerObjects.get(playerName);
                if (!player) return;

                const { orb, orbLight, body, head, shoulders, glow, pedestal, container } = player;
                
                switch(status) {
                    case 'active':
                        // Yellow glow for active player
                        orb.material.color.setHex(0xffff00);
                        orb.material.emissive.setHex(0xffff00);
                        orbLight.color.setHex(0xffff00);
                        orbLight.intensity = 1.5;
                        glow.material.color.setHex(0xffff00);
                        glow.material.emissive.setHex(0xffff00);
                        glow.material.emissiveIntensity = 0.5;
                        glow.visible = true;
                        // Slight scale up animation
                        container.scale.setScalar(1.1);
                        pedestal.material.emissive.setHex(0x444400);
                        pedestal.material.emissiveIntensity = 0.3;
                        break;
                    case 'dead':
                        // Gray out dead players
                        orb.material.color.setHex(0x333333);
                        orb.material.emissive.setHex(0x111111);
                        orb.material.emissiveIntensity = 0.1;
                        orb.material.opacity = 0.3;
                        orbLight.color.setHex(0x333333);
                        orbLight.intensity = 0.1;
                        body.material.color.setHex(0x444444);
                        body.material.emissive.setHex(0x000000);
                        shoulders.material.color.setHex(0x444444);
                        shoulders.material.emissive.setHex(0x000000);
                        head.material.color.setHex(0x666666);
                        head.material.emissive.setHex(0x000000);
                        glow.visible = false;
                        pedestal.material.emissive.setHex(0x000000);
                        // Sink into ground
                        container.position.y = -1.5;
                        // Tilt slightly
                        container.rotation.x = 0.2;
                        // Fade out nameplate
                        if (player.nameplate && player.nameplate.element) {
                            player.nameplate.element.style.transition = 'opacity 2s ease-out';
                            player.nameplate.element.style.opacity = '0.2';
                        }
                        player.isAlive = false;
                        break;
                    case 'werewolf':
                        // Red/purple glow for werewolves
                        orb.material.color.setHex(0xff0000);
                        orb.material.emissive.setHex(0xff0000);
                        orb.material.emissiveIntensity = 1.0;
                        orbLight.color.setHex(0xff0000);
                        orbLight.intensity = 1.2;
                        body.material.color.setHex(0x880000);
                        body.material.emissive.setHex(0x440000);
                        body.material.emissiveIntensity = 0.3;
                        shoulders.material.color.setHex(0x880000);
                        shoulders.material.emissive.setHex(0x440000);
                        shoulders.material.emissiveIntensity = 0.3;
                        glow.material.color.setHex(0xff0000);
                        glow.material.emissive.setHex(0xff0000);
                        glow.material.emissiveIntensity = 0.4;
                        glow.visible = true;
                        pedestal.material.emissive.setHex(0x440000);
                        pedestal.material.emissiveIntensity = 0.2;
                        break;
                    case 'voting':
                        // Orange pulse for voting
                        orb.material.color.setHex(0xff8800);
                        orb.material.emissive.setHex(0xff8800);
                        orb.material.emissiveIntensity = 0.9;
                        orbLight.color.setHex(0xff8800);
                        orbLight.intensity = 1.0;
                        glow.material.color.setHex(0xff8800);
                        glow.material.emissive.setHex(0xff8800);
                        glow.material.emissiveIntensity = 0.3;
                        glow.visible = true;
                        break;
                    case 'speaking':
                        // Blue pulse for speaking
                        orb.material.color.setHex(0x00aaff);
                        orb.material.emissive.setHex(0x00aaff);
                        orb.material.emissiveIntensity = 1.0;
                        orbLight.color.setHex(0x00aaff);
                        orbLight.intensity = 1.5;
                        glow.material.color.setHex(0x00aaff);
                        glow.material.emissive.setHex(0x00aaff);
                        glow.material.emissiveIntensity = 0.4;
                        glow.visible = true;
                        container.scale.setScalar(1.05);
                        break;
                    default:
                        // Default green for alive
                        if (player.isAlive) {
                            orb.material.color.setHex(0x00ff00);
                            orb.material.emissive.setHex(0x00ff00);
                            orb.material.emissiveIntensity = 0.8;
                            orb.material.opacity = 0.9;
                            orbLight.color.setHex(0x00ff00);
                            orbLight.intensity = 0.8;
                            body.material.color.setHex(0x4466ff);
                            body.material.emissive.setHex(0x111166);
                            body.material.emissiveIntensity = 0.2;
                            shoulders.material.color.setHex(0x4466ff);
                            shoulders.material.emissive.setHex(0x111166);
                            shoulders.material.emissiveIntensity = 0.2;
                            head.material.color.setHex(0xfdbcb4);
                            head.material.emissive.setHex(0x442211);
                            head.material.emissiveIntensity = 0.1;
                            glow.material.color.setHex(0x00ff00);
                            glow.material.emissive.setHex(0x00ff00);
                            glow.material.emissiveIntensity = 0.3;
                            glow.visible = true;
                            pedestal.material.emissive.setHex(0x111122);
                            pedestal.material.emissiveIntensity = 0.1;
                            container.scale.setScalar(1.0);
                            container.position.y = 0;
                            container.rotation.x = 0;
                        }
                        break;
                }
              }

              updatePhase(phase) {
                if (!this._scene) return;
                
                // Update enhanced lighting based on phase
                if (phase === 'NIGHT') {
                    // Night phase - cooler, darker lighting
                    if (this._mainLight) {
                      this._mainLight.color.setHex(0x9999ff);
                      this._mainLight.intensity = 1.2;
                    }
                    if (this._rimLight) {
                      this._rimLight.intensity = 1.0;
                      this._rimLight.color.setHex(0x6666ff);
                    }
                    if (this._hemiLight) {
                      this._hemiLight.intensity = 0.4;
                    }
                    this._scene.fog = new this._THREE.FogExp2(0x0a0a2a, 0.02);
                    
                    // Update atmosphere shader
                    if (this._atmospherePass) {
                      this._atmospherePass.uniforms.phase.value = 1.0;
                    }
                } else {
                    // Day phase - warmer, brighter lighting
                    if (this._mainLight) {
                      this._mainLight.color.setHex(0xffffcc);
                      this._mainLight.intensity = 1.6;
                    }
                    if (this._rimLight) {
                      this._rimLight.intensity = 0.6;
                      this._rimLight.color.setHex(0xffcc99);
                    }
                    if (this._hemiLight) {
                      this._hemiLight.intensity = 0.8;
                    }
                    this._scene.fog = new this._THREE.FogExp2(0x1a1a3a, 0.015);
                    
                    // Update atmosphere shader
                    if (this._atmospherePass) {
                      this._atmospherePass.uniforms.phase.value = 0.0;
                    }
                }
              }

              _createNameplate(name, imageUrl, CSS2DObject) {
                const container = document.createElement('div');
                container.style.backgroundColor = 'rgba(255, 255, 255, 0)';
                container.style.padding = '8px 12px';
                container.style.borderRadius = '8px';
                container.style.display = 'flex';
                container.style.alignItems = 'center';
                container.style.justifyContent = 'center';
                container.style.gap = '10px';
                container.style.textAlign = 'center';

                const img = document.createElement('img');
                img.src = imageUrl;
                img.style.width = '60px';
                img.style.height = '60px';
                img.style.borderRadius = '80%';
                img.style.objectFit = 'cover';
                img.style.backgroundColor = 'white';

                const text = document.createElement('div');
                text.textContent = name;
                text.style.color = 'white';
                text.style.fontFamily = 'Arial, sans-serif';
                text.style.fontSize = '16px';

                container.appendChild(img);
                container.appendChild(text);

                const label = new CSS2DObject(container);
                return label;
              }

              _FrameGroup(group, THREE) {
                const box = new THREE.Box3().setFromObject(group);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());

                const maxDim = Math.max(size.x, size.y, size.z);
                const fov = this._camera.fov * (Math.PI / 180);

                let cameraZ = Math.abs(maxDim / Math.tan(fov / 2));
                cameraZ *= 0.5;

                this._camera.position.set(center.x, center.y, center.z - cameraZ);

                const shiftY = size.y / 2.5;
                this._camera.position.y += shiftY;

                const newTarget = center.clone();
                newTarget.y += shiftY;
                this._controls.target.copy(newTarget);
                this._controls.update();
              }

              _NormalizeModel(model, THREE) {
                const box = new THREE.Box3().setFromObject(model);
                const size = box.getSize(new THREE.Vector3());
                const center = box.getCenter(new THREE.Vector3());
                model.position.copy(center).negate();
                const wrapper = new THREE.Group();
                wrapper.add(model);
                const scale = 1.0 / size.y;
                wrapper.scale.set(scale, scale, scale);
                return wrapper;
              }

              _RAF() {
                requestAnimationFrame((time) => {
                  // Update time-based uniforms
                  if (this._particleMaterial) {
                    this._particleMaterial.uniforms.time.value = time * 0.001;
                  }
                  if (this._atmospherePass) {
                    this._atmospherePass.uniforms.time.value = time * 0.001;
                  }
                  
                  // Animate particle system
                  if (this._particles) {
                    this._particles.rotation.y = time * 0.0001;
                    const positions = this._particles.geometry.attributes.position.array;
                    for (let i = 0; i < positions.length; i += 3) {
                      positions[i + 1] += Math.sin(time * 0.001 + positions[i] * 0.01) * 0.02;
                      // Wrap around if particles fall too low
                      if (positions[i + 1] < 0) {
                        positions[i + 1] = 35;
                      }
                    }
                    this._particles.geometry.attributes.position.needsUpdate = true;
                  }
                  
                  // Animate player objects with enhanced effects
                  if (this._playerObjects) {
                    this._playerObjects.forEach((player, name) => {
                      if (player.isAlive) {
                        // Enhanced floating animation for alive players
                        const floatOffset = Math.sin(time * 0.001 + player.baseAngle) * 0.2;
                        const bobOffset = Math.cos(time * 0.0015 + player.baseAngle * 2) * 0.05;
                        player.container.position.y = floatOffset + bobOffset;
                        
                        // More dynamic orb rotation
                        player.orb.rotation.y = time * 0.003;
                        player.orb.rotation.x = Math.sin(time * 0.002) * 0.15;
                        player.orb.rotation.z = Math.cos(time * 0.0025) * 0.1;
                        
                        // Enhanced glow animation
                        if (player.glow && player.glow.visible) {
                          player.glow.rotation.y = -time * 0.002;
                          const glowScale = 1 + Math.sin(time * 0.004 + player.baseAngle) * 0.15;
                          player.glow.scale.setScalar(glowScale);
                          
                          // Pulsing emissive intensity
                          const pulseIntensity = 0.3 + Math.sin(time * 0.005 + player.baseAngle) * 0.1;
                          player.glow.material.emissiveIntensity = pulseIntensity;
                        }
                        
                        // Enhanced pulse effect for active players
                        if (player.container.scale.x > 1.0) {
                          const pulseScale = 1.05 + Math.sin(time * 0.008) * 0.08;
                          player.container.scale.setScalar(pulseScale);
                        }
                        
                        // Enhanced breathing effect
                        if (player.body) {
                          const breathScale = 1 + Math.sin(time * 0.002 + player.baseAngle) * 0.03;
                          player.body.scale.y = breathScale;
                          if (player.shoulders) {
                            player.shoulders.scale.y = 0.6 * breathScale;
                          }
                        }
                        
                        // Subtle head movement
                        if (player.head) {
                          player.head.rotation.y = Math.sin(time * 0.001 + player.baseAngle) * 0.1;
                        }
                      } else {
                        // Dead players have reduced animation
                        if (player.orb) {
                          player.orb.rotation.y = time * 0.0008;
                        }
                      }
                    });
                  }

                  // Use post-processing composer if available, otherwise fallback to direct render
                  if (this._composer) {
                    this._composer.render();
                  } else {
                    this._threejs.render(this._scene, this._camera);
                  }
                  this._labelRenderer.render(this._scene, this._camera);
                  this._RAF();
                });
              }
            }

            setupScene(BasicWorldDemo);
        } catch (error) {
            console.error("Failed to load Three.js modules:", error);
            parent.textContent = "Error loading 3D assets. Please refresh.";
        }
    };

    loadAndSetup();
  }

  function setupScene(BasicWorldDemo) {
    if (threeState.initialized) return;
    threeState.demo = new BasicWorldDemo({ parent, width, height });
    threeState.initialized = true;
  }

  function updateSceneFromGameState(gameState, playerMap, actingPlayerName) {
    if (!threeState.demo || !threeState.demo._playerObjects) return;

    // Update player statuses
    gameState.players.forEach(player => {
      const playerObj = threeState.demo._playerObjects.get(player.name);
      if (!playerObj) return;

      if (!player.is_alive) {
        threeState.demo.updatePlayerStatus(player.name, 'dead');
      } else if (player.name === actingPlayerName) {
        threeState.demo.updatePlayerStatus(player.name, 'active');
      } else if (player.role === 'Werewolf' && gameState.phase === 'NIGHT') {
        threeState.demo.updatePlayerStatus(player.name, 'werewolf');
      } else {
        threeState.demo.updatePlayerStatus(player.name, 'default');
      }
    });

    // Update phase lighting
    threeState.demo.updatePhase(gameState.phase);

    // Handle recent events for animations
    const recentEvents = gameState.eventLog.slice(-5); // Last 5 events
    recentEvents.forEach(event => {
      if (event.type === 'chat' && playerMap.has(event.speaker)) {
        threeState.demo.updatePlayerStatus(event.speaker, 'speaking');
        // Reset after a delay
        setTimeout(() => {
          const player = playerMap.get(event.speaker);
          if (player && player.is_alive && event.speaker !== actingPlayerName) {
            threeState.demo.updatePlayerStatus(event.speaker, 'default');
          }
        }, 2000);
      } else if (event.type === 'vote' && playerMap.has(event.actor_id)) {
        threeState.demo.updatePlayerStatus(event.actor_id, 'voting');
        // Reset after a delay
        setTimeout(() => {
          const player = playerMap.get(event.actor_id);
          if (player && player.is_alive && event.actor_id !== actingPlayerName) {
            threeState.demo.updatePlayerStatus(event.actor_id, 'default');
          }
        }, 1500);
      }
    });
  }

  // --- CSS for the UI ---
  const css = `
        :root {
            --night-bg: linear-gradient(135deg, #1a1a2e, #16213e);
            --day-bg: linear-gradient(135deg, #74b9ff, #0984e3);
            --night-text: #f8f9fa;
            --day-text: #2d3436;
            --dead-filter: grayscale(100%) brightness(40%) contrast(0.8);
            --active-border: #fdcb6e;
            --active-glow: rgba(253, 203, 110, 0.4);
            --werewolf-color: #e17055;
            --villager-color: #00b894;
            --doctor-color: #6c5ce7;
            --seer-color: #fd79a8;
            --panel-bg: rgba(33, 40, 54, 0.95);
            --panel-border: rgba(116, 185, 255, 0.2);
            --hover-bg: rgba(116, 185, 255, 0.1);
            --card-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
            --text-primary: #f8f9fa;
            --text-secondary: #b2bec3;
            --text-muted: #74b9ff;
        }

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        .werewolf-parent {
            position: relative;
            overflow: hidden;
            width: 100%;
            height: 100%;
            background: radial-gradient(ellipse at center, #0f1419 0%, #000000 100%);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .main-container {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: 2;
            pointer-events: none;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: var(--text-primary);
            font-weight: 400;
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        
        /* Enhanced Panel Styling */
        .left-panel, .right-panel {
            position: fixed;
            top: 20px;
            max-height: calc(100vh - 40px);
            background: var(--panel-bg);
            backdrop-filter: blur(20px) saturate(1.5);
            border-radius: 16px;
            border: 1px solid var(--panel-border);
            padding: 20px;
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
            pointer-events: auto;
            box-shadow: var(--card-shadow), 0 0 40px rgba(116, 185, 255, 0.05);
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .left-panel {
            left: 20px;
            width: 320px;
        }
        
        .right-panel {
            right: 20px;
            width: 420px;
        }
        
        .left-panel:hover, .right-panel:hover {
            border-color: rgba(116, 185, 255, 0.3);
            box-shadow: var(--card-shadow), 0 0 60px rgba(116, 185, 255, 0.08);
        }
        
        /* Enhanced Headers */
        .right-panel h1, #player-list-area h1 {
            margin: 0 0 20px 0;
            text-align: center;
            font-size: 1.75rem;
            font-weight: 600;
            color: var(--text-primary);
            background: linear-gradient(135deg, #74b9ff, #0984e3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            position: relative;
            padding-bottom: 15px;
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
            background: linear-gradient(90deg, transparent, #74b9ff, transparent);
            border-radius: 2px;
        }
        
        /* Enhanced Player List */
        #player-list-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-height: 0;
        }
        
        #player-list-container {
            overflow-y: auto;
            flex-grow: 1;
            padding-right: 8px;
            margin-right: -8px;
        }
        
        #player-list-container::-webkit-scrollbar {
            width: 6px;
        }
        
        #player-list-container::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 3px;
        }
        
        #player-list-container::-webkit-scrollbar-thumb {
            background: rgba(116, 185, 255, 0.3);
            border-radius: 3px;
        }
        
        #player-list-container::-webkit-scrollbar-thumb:hover {
            background: rgba(116, 185, 255, 0.5);
        }
        
        #player-list {
            list-style: none;
            padding: 0;
            margin: 0;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        /* Enhanced Player Cards */
        .player-card {
            position: relative;
            display: flex;
            align-items: center;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.03));
            padding: 16px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
            overflow: hidden;
        }
        
        .player-card::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 4px;
            background: transparent;
            transition: all 0.3s ease;
        }
        
        .player-card:hover {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.06));
            border-color: rgba(116, 185, 255, 0.3);
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
        }
        
        .player-card.active {
            background: linear-gradient(135deg, rgba(253, 203, 110, 0.15), rgba(253, 203, 110, 0.05));
            border-color: var(--active-border);
            box-shadow: 0 0 20px var(--active-glow), 0 4px 20px rgba(0, 0, 0, 0.1);
        }
        
        .player-card.active::before {
            background: linear-gradient(180deg, var(--active-border), rgba(253, 203, 110, 0.5));
        }
        
        .player-card.dead {
            opacity: 0.5;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.01));
            filter: brightness(0.7);
        }
        
        /* Enhanced Avatar */
        .avatar-container {
            position: relative;
            width: 56px;
            height: 56px;
            margin-right: 16px;
            flex-shrink: 0;
        }
        
        .player-card .avatar {
            width: 100%;
            height: 100%;
            border-radius: 50%;
            object-fit: cover;
            background-color: #ffffff;
            border: 2px solid rgba(116, 185, 255, 0.2);
            transition: all 0.3s ease;
        }
        
        .player-card:hover .avatar {
            border-color: rgba(116, 185, 255, 0.4);
            box-shadow: 0 0 15px rgba(116, 185, 255, 0.2);
        }
        
        .player-card.active .avatar {
            border-color: var(--active-border);
            box-shadow: 0 0 15px var(--active-glow);
        }
        
        .player-card.dead .avatar {
            filter: var(--dead-filter);
            border-color: rgba(255, 255, 255, 0.1);
        }
        
        /* Enhanced Player Info */
        .player-info {
            flex-grow: 1;
            overflow: hidden;
            min-width: 0;
        }
        
        .player-name {
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 4px;
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            letter-spacing: -0.01em;
        }
        
        .player-role {
            font-size: 0.875rem;
            color: var(--text-secondary);
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .player-role.werewolf { color: var(--werewolf-color); }
        .player-role.villager { color: var(--villager-color); }
        .player-role.doctor { color: var(--doctor-color); }
        .player-role.seer { color: var(--seer-color); }
        
        /* Enhanced Threat Indicator */
        .threat-indicator {
            position: absolute;
            top: 12px;
            right: 12px;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: transparent;
            transition: all 0.3s ease;
            box-shadow: 0 0 8px rgba(0, 0, 0, 0.3);
        }
        
        /* Enhanced Chat/Event Log */
        #chat-log {
            list-style: none;
            padding: 0;
            margin: 0;
            flex-grow: 1;
            overflow-y: auto;
            padding-right: 8px;
            margin-right: -8px;
        }
        
        #chat-log::-webkit-scrollbar {
            width: 6px;
        }
        
        #chat-log::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 3px;
        }
        
        #chat-log::-webkit-scrollbar-thumb {
            background: rgba(116, 185, 255, 0.3);
            border-radius: 3px;
        }
        
        #chat-log::-webkit-scrollbar-thumb:hover {
            background: rgba(116, 185, 255, 0.5);
        }
        
        /* Enhanced Chat Entries */
        .chat-entry {
            display: flex;
            margin-bottom: 20px;
            align-items: flex-start;
            animation: fadeInUp 0.3s ease-out;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .chat-avatar {
            width: 44px;
            height: 44px;
            border-radius: 50%;
            margin-right: 12px;
            object-fit: cover;
            flex-shrink: 0;
            border: 2px solid rgba(116, 185, 255, 0.2);
            transition: all 0.3s ease;
        }
        
        .chat-entry:hover .chat-avatar {
            border-color: rgba(116, 185, 255, 0.4);
        }
        
        .message-content {
            display: flex;
            flex-direction: column;
            flex-grow: 1;
            min-width: 0;
        }
        
        /* Enhanced Message Bubbles */
        .balloon {
            padding: 14px 16px;
            border-radius: 16px 16px 16px 4px;
            max-width: 85%;
            word-wrap: break-word;
            background: linear-gradient(135deg, rgba(116, 185, 255, 0.1), rgba(116, 185, 255, 0.05));
            border: 1px solid rgba(116, 185, 255, 0.2);
            transition: all 0.3s ease;
            position: relative;
            line-height: 1.4;
            font-size: 0.95rem;
        }
        
        .balloon:hover {
            background: linear-gradient(135deg, rgba(116, 185, 255, 0.15), rgba(116, 185, 255, 0.08));
            border-color: rgba(116, 185, 255, 0.3);
        }
        
        .chat-entry.event-day .balloon {
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.1), rgba(255, 193, 7, 0.05));
            border-color: rgba(255, 193, 7, 0.2);
            color: var(--text-primary);
        }
        
        .chat-entry.event-night .balloon {
            background: linear-gradient(135deg, rgba(108, 92, 231, 0.1), rgba(108, 92, 231, 0.05));
            border-color: rgba(108, 92, 231, 0.2);
        }
        
        /* Enhanced System Messages */
        .msg-entry {
            border-left: 4px solid #f39c12;
            padding: 16px;
            margin: 16px 0;
            border-radius: 8px;
            background: linear-gradient(135deg, rgba(243, 156, 18, 0.1), rgba(243, 156, 18, 0.05));
            border: 1px solid rgba(243, 156, 18, 0.2);
            transition: all 0.3s ease;
            animation: fadeInUp 0.3s ease-out;
        }
        
        .msg-entry:hover {
            background: linear-gradient(135deg, rgba(243, 156, 18, 0.15), rgba(243, 156, 18, 0.08));
            border-color: rgba(243, 156, 18, 0.3);
        }
        
        .msg-entry.event-day {
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.1), rgba(255, 193, 7, 0.05));
            border-color: rgba(255, 193, 7, 0.2);
        }
        
        .msg-entry.event-night {
            background: linear-gradient(135deg, rgba(108, 92, 231, 0.1), rgba(108, 92, 231, 0.05));
            border-color: rgba(108, 92, 231, 0.2);
        }
        
        .msg-entry.game-event {
            border-left-color: #e74c3c;
            background: linear-gradient(135deg, rgba(231, 76, 60, 0.1), rgba(231, 76, 60, 0.05));
            border-color: rgba(231, 76, 60, 0.2);
        }
        
        .msg-entry.game-win {
            border-left-color: #2ecc71;
            background: linear-gradient(135deg, rgba(46, 204, 113, 0.1), rgba(46, 204, 113, 0.05));
            border-color: rgba(46, 204, 113, 0.2);
            line-height: 1.6;
        }
        
        /* Enhanced Reasoning Text */
        .reasoning-text {
            font-size: 0.85rem;
            color: var(--text-muted);
            font-style: italic;
            margin-top: 8px;
            padding-left: 12px;
            border-left: 2px solid rgba(116, 185, 255, 0.3);
            line-height: 1.4;
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* Enhanced Citations */
        #chat-log cite {
            font-style: normal;
            font-weight: 600;
            display: flex;
            align-items: center;
            font-size: 0.875rem;
            color: var(--text-primary);
            margin-bottom: 6px;
            gap: 8px;
        }
        
        /* Enhanced Moderator Announcements */
        .moderator-announcement {
            margin: 16px 0;
            animation: fadeInUp 0.3s ease-out;
        }
        
        .moderator-announcement-content {
            padding: 16px;
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(46, 204, 113, 0.1), rgba(46, 204, 113, 0.05));
            border: 1px solid rgba(46, 204, 113, 0.2);
            border-left: 4px solid #2ecc71;
            color: var(--text-primary);
            line-height: 1.5;
            transition: all 0.3s ease;
        }
        
        .moderator-announcement-content:hover {
            background: linear-gradient(135deg, rgba(46, 204, 113, 0.15), rgba(46, 204, 113, 0.08));
            border-color: rgba(46, 204, 113, 0.3);
        }
        
        /* Enhanced Timestamps */
        .timestamp {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-weight: 500;
            font-family: 'JetBrains Mono', monospace;
            background: rgba(116, 185, 255, 0.1);
            padding: 2px 6px;
            border-radius: 4px;
            margin-left: auto;
        }
        
        /* Enhanced Player Capsules */
        .player-capsule {
            display: inline-flex;
            align-items: center;
            background: linear-gradient(135deg, rgba(116, 185, 255, 0.15), rgba(116, 185, 255, 0.08));
            border: 1px solid rgba(116, 185, 255, 0.2);
            border-radius: 16px;
            padding: 2px 10px 2px 2px;
            font-size: 0.875rem;
            font-weight: 500;
            margin: 0 2px;
            vertical-align: middle;
            transition: all 0.3s ease;
        }
        
        .player-capsule:hover {
            background: linear-gradient(135deg, rgba(116, 185, 255, 0.2), rgba(116, 185, 255, 0.1));
            border-color: rgba(116, 185, 255, 0.3);
        }
        
        .capsule-avatar {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 6px;
            object-fit: cover;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* Enhanced TTS Button */
        .tts-button {
            cursor: pointer;
            font-size: 1.1rem;
            margin-left: 12px;
            padding: 4px;
            border-radius: 50%;
            transition: all 0.3s ease;
            opacity: 0.6;
        }
        
        .tts-button:hover {
            opacity: 1;
            background: rgba(116, 185, 255, 0.1);
            transform: scale(1.1);
        }
        
        /* Enhanced Audio Controls */
        .audio-controls {
            padding: 16px 0;
            border-top: 1px solid rgba(116, 185, 255, 0.2);
            margin-top: 16px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 8px;
            padding: 16px;
        }
        
        .audio-controls label {
            display: block;
            margin-bottom: 8px;
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-secondary);
        }
        
        .audio-controls input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: rgba(116, 185, 255, 0.2);
            outline: none;
            -webkit-appearance: none;
        }
        
        .audio-controls input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #74b9ff;
            cursor: pointer;
            border: 2px solid #ffffff;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
        }
        
        #pause-audio {
            background-color: rgba(116, 185, 255, 0.1);
            border: 1px solid rgba(116, 185, 255, 0.3);
            border-radius: 50%;
            width: 36px;
            height: 36px;
            cursor: pointer;
            padding: 0;
            background-size: 16px;
            background-repeat: no-repeat;
            background-position: center;
            transition: all 0.3s ease;
            filter: none;
        }
        
        #pause-audio:hover {
            background-color: rgba(116, 185, 255, 0.2);
            border-color: rgba(116, 185, 255, 0.5);
            transform: scale(1.1);
        }
        
        #pause-audio.paused {
            background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiM3NGI5ZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cG9seWdvbiBwb2ludHM9IjYgMyAyMCAxMiA2IDIxIi8+PC9zdmc+');
        }
        
        #pause-audio.playing {
            background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiM3NGI5ZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cmVjdCB4PSI2IiB5PSI0IiB3aWR0aD0iNCIgaGVpZ2h0PSIxNiIgcng9IjEiLz48cmVjdCB4PSIxNCIgeT0iNCIgd2lkdGg9IjQiIGhlaWdodD0iMTYiIHJ4PSIxIi8+PC9zdmc+');
        }
        
        /* Message text formatting */
        .msg-text {
            line-height: 1.5;
            font-size: 0.95rem;
        }
        
        .msg-text br {
            display: block;
            margin-bottom: 0.5em;
            content: "";
        }
        
        /* Smooth scrolling */
        * {
            scrollbar-width: thin;
            scrollbar-color: rgba(116, 185, 255, 0.3) transparent;
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
          playbackRate: 1.4,
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
      audioState.isAudioPlaying = true;
      const event = audioState.audioQueue.shift();
      const audioKey = event.speaker === 'moderator' ? `moderator:${event.message}` : `${event.speaker}:${event.message}`;
      const audioPath = audioMap[audioKey];

      if (audioPath) {
          audioState.audioPlayer.src = audioPath;
          audioState.audioPlayer.playbackRate = audioState.playbackRate;
          audioState.audioPlayer.onended = () => {
              audioState.isAudioPlaying = false;
              if (!audioState.isPaused) {
                playNextInQueue();
              }
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
      const sortedPlayerIds = [...playerIds].sort((a, b) => b.length - a.length);

      sortedPlayerIds.forEach(playerId => {
          const player = playerMap.get(playerId);
          if (player) {
              const capsule = createPlayerCapsule(player);
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
    const sortedPlayerIds = [...playerIds].sort((a, b) => b.length - a.length);

    sortedPlayerIds.forEach(playerId => {
        const regex = new RegExp(`\b${playerId.replace(/[-\/\\^$*+?.()|[\\]{}/g, '\\$&')}\b`, 'g');
        newText = newText.replace(regex, `<strong>${playerId}</strong>`);
    });
    return newText;
  }


  function getThreatColor(threatLevel) {
    const value = Math.max(0, Math.min(1, threatLevel));
    const hue = 120 * (1 - value);
    return `hsl(${hue}, 100%, 50%)`;
  }

  function updatePlayerList(container, gameState, actingPlayerName) {
    // Get or create header
    let header = container.querySelector('h1');
    if (!header) {
        header = document.createElement('h1');
        header.textContent = 'Players';
        container.appendChild(header);
    }

    // Get or create list container
    let listContainer = container.querySelector('#player-list-container');
    if (!listContainer) {
        listContainer = document.createElement('div');
        listContainer.id = 'player-list-container';
        container.appendChild(listContainer);
    }

    // Get or create player list
    let playerUl = listContainer.querySelector('#player-list');
    if (!playerUl) {
        playerUl = document.createElement('ul');
        playerUl.id = 'player-list';
        listContainer.appendChild(playerUl);
    }

    // Update player cards
    gameState.players.forEach((player, index) => {
        let li = playerUl.children[index];
        if (!li) {
            li = document.createElement('li');
            playerUl.appendChild(li);
        }

        // Update player card classes
        li.className = 'player-card';
        if (!player.is_alive) li.classList.add('dead');
        if (player.name === actingPlayerName) li.classList.add('active');

        let roleDisplay = player.role;
        if (player.role === 'Werewolf') {
            roleDisplay = `&#x1F43A; ${player.role}`;
        } else if (player.role === 'Doctor') {
            roleDisplay = `&#x1FA7A; ${player.role}`;
        } else if (player.role === 'Seer') {
            roleDisplay = `&#x1F52E; ${player.role}`;
        } else if (player.role === 'Villager') {
            roleDisplay = `&#x1F9D1; ${player.role}`;
        }

        const roleText = player.role !== 'Unknown' ? `Role: ${roleDisplay}` : 'Role: Unknown';

        // Update content
        li.innerHTML = `
            <div class="avatar-container">
                <img src="${player.thumbnail}" alt="${player.name}" class="avatar">
            </div>
            <div class="player-info">
                <div class="player-name" title="${player.name}">${player.name}</div>
                <div class="player-role">${roleText}</div>
            </div>
            <div class="threat-indicator"></div>
        `;

        // Update threat indicator
        const indicator = li.querySelector('.threat-indicator');
        if (indicator && player.is_alive) {
            const threatLevel = gameState.playerThreatLevels.get(player.name) || 0;
            indicator.style.backgroundColor = getThreatColor(threatLevel);
        } else if (indicator) {
            indicator.style.backgroundColor = 'transparent';
        }
    });

    // Remove excess player cards
    while (playerUl.children.length > gameState.players.length) {
        playerUl.removeChild(playerUl.lastChild);
    }

    // Get or create audio controls
    let audioControls = container.querySelector('.audio-controls');
    if (!audioControls) {
        audioControls = document.createElement('div');
        audioControls.className = 'audio-controls';
        const pauseButtonClass = audioState.isPaused ? 'paused' : 'playing';
        audioControls.innerHTML = `
            <label for="playback-speed">Audio Speed: <span id="speed-label">${audioState.playbackRate.toFixed(1)}</span>x</label>
            <div style="display: flex; align-items: center; gap: 10px; margin-top: 5px;">
                <input type="range" id="playback-speed" min="0.5" max="2.5" step="0.1" value="${audioState.playbackRate}" style="flex-grow: 1;">
                <button id="pause-audio" class="${pauseButtonClass}"></button>
            </div>
        `;
        container.appendChild(audioControls);

        // Add event listeners only once
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
  }

  function updateEventLog(container, gameState, playerMap) {
    // Get or create header
    let header = container.querySelector('h1');
    if (!header) {
        header = document.createElement('h1');
        header.textContent = 'Event Log';
        container.appendChild(header);
    }

    // Get or create log container
    let logUl = container.querySelector('#chat-log');
    if (!logUl) {
        logUl = document.createElement('ul');
        logUl.id = 'chat-log';
        container.appendChild(logUl);
    }

    const logEntries = gameState.eventLog;
    
    // Store current scroll position
    const wasScrolledToBottom = logUl.scrollHeight - logUl.clientHeight <= logUl.scrollTop + 1;

    // Only add new entries, don't rebuild the entire log
    const currentEntryCount = logUl.children.length;
    
    if (logEntries.length === 0 && currentEntryCount === 0) {
        const li = document.createElement('li');
        li.className = 'msg-entry';
        li.innerHTML = `<cite>System</cite><div>The game is about to begin...</div>`;
        logUl.appendChild(li);
    } else if (logEntries.length > currentEntryCount) {
        // Only process new entries
        const newEntries = logEntries.slice(currentEntryCount);
        newEntries.forEach(entry => {
            const li = document.createElement('li');
            let reasoningHtml = entry.reasoning ? `<div class="reasoning-text">"${entry.reasoning}"</div>` : '';

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
                phaseEmoji = '&#x2600;&#xFE0F;';
            } else if (phase === 'NIGHT') {
                phaseEmoji = '&#x1F319;';
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
                    const balloonText = li.querySelector('.balloon-text');
                    if (balloonText) {
                        const ttsButton = document.createElement('span');
                        ttsButton.className = 'tts-button';
                        ttsButton.innerHTML = '&#x1F50A;';
                        ttsButton.onclick = () => speak(entry.message, entry.speaker);
                        balloonText.appendChild(ttsButton);
                    }
                    break;
                default:
                    // Handle other message types here if needed
                    break;
            }
            
            if (li.innerHTML) {
                logUl.appendChild(li);
            }
        });

        // Maintain scroll position
        if (wasScrolledToBottom) {
            logUl.scrollTop = logUl.scrollHeight;
        }
    }
  }

  function renderPlayerList(container, gameState, actingPlayerName) {
    container.innerHTML = '<h1>Players</h1>';
    const listContainer = document.createElement('div');
    listContainer.id = 'player-list-container';
    const playerUl = document.createElement('ul');
    playerUl.id = 'player-list';

    gameState.players.forEach(player => {
        const li = document.createElement('li');
        li.className = 'player-card';
        if (!player.is_alive) li.classList.add('dead');
        if (player.name === actingPlayerName) li.classList.add('active');

        let roleDisplay = player.role;
        if (player.role === 'Werewolf') {
            roleDisplay = `&#x1F43A; ${player.role}`;
        } else if (player.role === 'Doctor') {
            roleDisplay = `&#x1FA7A; ${player.role}`;
        } else if (player.role === 'Seer') {
            roleDisplay = `&#x1F52E; ${player.role}`;
        } else if (player.role === 'Villager') {
            roleDisplay = `&#x1F9D1; ${player.role}`;
        }

        const roleText = player.role !== 'Unknown' ? `Role: ${roleDisplay}` : 'Role: Unknown';

        li.innerHTML = `
            <div class="avatar-container">
                <img src="${player.thumbnail}" alt="${player.name}" class="avatar">
            </div>
            <div class="player-info">
                <div class="player-name" title="${player.name}">${player.name}</div>
                <div class="player-role">${roleText}</div>
            </div>
            <div class="threat-indicator"></div>
        `;
        playerUl.appendChild(li);
    });

    listContainer.appendChild(playerUl);
    container.appendChild(listContainer);

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

    const audioControls = document.createElement('div');
    audioControls.className = 'audio-controls';
    const pauseButtonClass = audioState.isPaused ? 'paused' : 'playing';
    audioControls.innerHTML = `
        <label for="playback-speed">Audio Speed: <span id="speed-label">${audioState.playbackRate.toFixed(1)}</span>x</label>
        <div style="display: flex; align-items: center; gap: 10px; margin-top: 5px;">
            <input type="range" id="playback-speed" min="0.5" max="2.5" step="0.1" value="${audioState.playbackRate}" style="flex-grow: 1;">
            <button id="pause-audio" class="${pauseButtonClass}"></button>
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
        logEntries.forEach(entry => {
            const li = document.createElement('li');
            let reasoningHtml = entry.reasoning ? `<div class="reasoning-text">"${entry.reasoning}"</div>` : '';

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
                phaseEmoji = '&#x2600;&#xFE0F;';
            } else if (phase === 'NIGHT') {
                phaseEmoji = '&#x1F319;';
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
                    const balloonText = li.querySelector('.balloon-text');
                    if (balloonText) {
                        const ttsButton = document.createElement('span');
                        ttsButton.className = 'tts-button';
                        ttsButton.innerHTML = '&#x1F50A;';
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
                    const listRegex = /\\\\[(.*?)\\\\]/g;
                    systemText = systemText.replace(listRegex, (match, listContent) => {
                        return listContent.replace(/'/g, "").replace(/, /g, " ");
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
                    break;
                case 'exile':
                    const exiledPlayerCap = createPlayerCapsule(playerMap.get(entry.name));
                    li.className = `msg-entry game-event event-day`;
                    li.innerHTML = `<cite>Exile ${timestampHtml}</cite><div class="msg-text">${exiledPlayerCap} (${entry.role}) was exiled by vote.</div>`;
                    break;
                case 'elimination':
                    const elimPlayerCap = createPlayerCapsule(playerMap.get(entry.name));
                    li.className = `msg-entry game-event event-night`;
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
                    li.className = `msg-entry game-win ${phaseClass}`;
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

    // --- Main Rendering Logic (Incremental) ---
    // Only create UI elements if they don't exist
    let mainContainer = parent.querySelector('.main-container');
    let style = parent.querySelector('style');
    
    if (!style) {
        style = document.createElement('style');
        style.textContent = css;
        parent.appendChild(style);
    }

    initThreeJs();

    if (!environment || !environment.steps || environment.steps.length === 0 || step >= environment.steps.length) {
        if (!mainContainer) {
            const tempContainer = document.createElement("div");
            tempContainer.textContent = "Waiting for game data or invalid step...";
            parent.appendChild(tempContainer);
        }
        return;
    }

    // Initialize player mapping for 3D scene
    let playerNamesFor3D = [];
    let playerThumbnailsFor3D = {};

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
        playerNamesFor3D = [...allPlayerNamesList];
        playerThumbnailsFor3D = {...playerThumbnails};
    } else if (environment.configuration && environment.configuration.agents) {
        console.warn("Renderer: Initial observation missing or incomplete. Reconstructing players from configuration.");
        allPlayerNamesList = environment.configuration.agents.map(agent => agent.id);
        environment.configuration.agents.forEach(agent => {
            if (agent.id && agent.thumbnail) {
                playerThumbnails[agent.id] = agent.thumbnail;
            }
        });
        playerNamesFor3D = [...allPlayerNamesList];
        playerThumbnailsFor3D = {...playerThumbnails};
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

    gameState.players.forEach(p => gameState.playerThreatLevels.set(p.name, 0));

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
            default: return 0;
        }
    }

    for (let s = 0; s <= step; s++) {
        const stepStateList = environment.steps[s];
        if (!stepStateList) continue;

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

        const moderatorLogForStep = environment.info?.MODERATOR_OBSERVATION?.[s] || [];
        moderatorLogForStep.forEach(dataEntry => {
             const eventKey = dataEntry.json_str;
             if (processedEvents.has(eventKey)) return;
             processedEvents.add(eventKey);

             const historyEvent = JSON.parse(dataEntry.json_str);
             const data = historyEvent.data;
             const timestamp = historyEvent.created_at;

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

    // Create or get existing main container
    if (!mainContainer) {
        mainContainer = document.createElement('div');
        mainContainer.className = 'main-container';
        parent.appendChild(mainContainer);
    }

    // Create or get existing panels
    let leftPanel = mainContainer.querySelector('.left-panel');
    if (!leftPanel) {
        leftPanel = document.createElement('div');
        leftPanel.className = 'left-panel';
        mainContainer.appendChild(leftPanel);
    }

    let playerListArea = leftPanel.querySelector('#player-list-area');
    if (!playerListArea) {
        playerListArea = document.createElement('div');
        playerListArea.id = 'player-list-area';
        leftPanel.appendChild(playerListArea);
    }

    let rightPanel = mainContainer.querySelector('.right-panel');
    if (!rightPanel) {
        rightPanel = document.createElement('div');
        rightPanel.className = 'right-panel';
        mainContainer.appendChild(rightPanel);
    }

    // Update existing content instead of clearing and rebuilding
    updatePlayerList(playerListArea, gameState, actingPlayerName);
    updateEventLog(rightPanel, gameState, playerMap);

    // Update 3D scene based on game state
    updateSceneFromGameState(gameState, playerMap, actingPlayerName);
    
    // Initialize 3D players if needed
    if (threeState.demo && threeState.demo._playerObjects && threeState.demo._playerObjects.size === 0 && playerNamesFor3D.length > 0) {
        initializePlayers3D(playerNamesFor3D, playerThumbnailsFor3D, threeState);
    }
}

function initializePlayers3D(playerNames, playerThumbnails, threeState) {
    if (!threeState || !threeState.demo || !threeState.demo._playerObjects) return;
    
    // Clear existing player objects
    if (threeState.demo._playerGroup) {
        // Remove all children from the group
        while(threeState.demo._playerGroup.children.length > 0) {
            threeState.demo._playerGroup.remove(threeState.demo._playerGroup.children[0]);
        }
    }
    threeState.demo._playerObjects.clear();
    
    const numPlayers = playerNames.length;
    const radius = 18; // Increased radius to use more space
    const playerHeight = 4;
    
    const THREE = threeState.demo._THREE;
    const CSS2DObject = threeState.demo._CSS2DObject;
    
    // Create a circular platform
    const platformGeometry = new THREE.RingGeometry(radius - 2, radius + 3, 64);
    const platformMaterial = new THREE.MeshStandardMaterial({
        color: 0x2a2a3a,
        roughness: 0.9,
        metalness: 0.1,
        transparent: true,
        opacity: 0.5,
        side: THREE.DoubleSide
    });
    const platform = new THREE.Mesh(platformGeometry, platformMaterial);
    platform.rotation.x = -Math.PI / 2;
    platform.position.y = -0.05;
    platform.receiveShadow = true;
    threeState.demo._playerGroup.add(platform);
    
    // Create center decoration
    const centerGeometry = new THREE.CylinderGeometry(3, 3, 0.2, 32);
    const centerMaterial = new THREE.MeshStandardMaterial({
        color: 0x444466,
        roughness: 0.7,
        metalness: 0.3,
        emissive: 0x222244,
        emissiveIntensity: 0.2
    });
    const centerPlatform = new THREE.Mesh(centerGeometry, centerMaterial);
    centerPlatform.position.y = 0.1;
    centerPlatform.receiveShadow = true;
    threeState.demo._playerGroup.add(centerPlatform);
    
    // Add decorative lines from center to each player position
    const linesMaterial = new THREE.LineBasicMaterial({
        color: 0x334455,
        transparent: true,
        opacity: 0.3
    });
    
    playerNames.forEach((name, i) => {
        const playerContainer = new THREE.Group();
        // Use full circle (360 degrees)
        const angle = (i / numPlayers) * Math.PI * 2;
        
        const x = radius * Math.sin(angle);
        const z = radius * Math.cos(angle);
        playerContainer.position.set(x, 0, z);
        
        // Create line from center to player
        const lineGeometry = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0.05, 0),
            new THREE.Vector3(x, 0.05, z)
        ]);
        const line = new THREE.Line(lineGeometry, linesMaterial);
        threeState.demo._playerGroup.add(line);
        
        // Create pedestal for each player
        const pedestalGeometry = new THREE.CylinderGeometry(1.5, 1.8, 0.4, 16);
        const pedestalMaterial = new THREE.MeshStandardMaterial({
            color: 0x333344,
            roughness: 0.8,
            metalness: 0.2,
            emissive: 0x111122,
            emissiveIntensity: 0.1
        });
        const pedestal = new THREE.Mesh(pedestalGeometry, pedestalMaterial);
        pedestal.position.y = 0.2;
        pedestal.castShadow = true;
        pedestal.receiveShadow = true;
        playerContainer.add(pedestal);
        
        // Create player body (more detailed)
        const bodyGeometry = new THREE.CylinderGeometry(0.8, 1, playerHeight * 0.6, 16);
        const bodyMaterial = new THREE.MeshStandardMaterial({
            color: 0x4466ff,
            roughness: 0.5,
            metalness: 0.3,
            emissive: 0x111166,
            emissiveIntensity: 0.2
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.y = playerHeight * 0.4;
        body.castShadow = true;
        body.receiveShadow = true;
        playerContainer.add(body);
        
        // Create shoulders
        const shoulderGeometry = new THREE.SphereGeometry(1, 16, 8);
        const shoulderMaterial = new THREE.MeshStandardMaterial({
            color: 0x4466ff,
            roughness: 0.5,
            metalness: 0.3,
            emissive: 0x111166,
            emissiveIntensity: 0.2
        });
        const shoulders = new THREE.Mesh(shoulderGeometry, shoulderMaterial);
        shoulders.position.y = playerHeight * 0.65;
        shoulders.scale.set(1.2, 0.6, 0.8);
        shoulders.castShadow = true;
        playerContainer.add(shoulders);
        
        // Create player head (sphere)
        const headGeometry = new THREE.SphereGeometry(0.7, 16, 16);
        const headMaterial = new THREE.MeshStandardMaterial({
            color: 0xfdbcb4,
            roughness: 0.7,
            metalness: 0.1,
            emissive: 0x442211,
            emissiveIntensity: 0.1
        });
        const head = new THREE.Mesh(headGeometry, headMaterial);
        head.position.y = playerHeight * 0.85;
        head.castShadow = true;
        head.receiveShadow = true;
        playerContainer.add(head);
        
        // Create eyes
        const eyeGeometry = new THREE.SphereGeometry(0.08, 8, 6);
        const eyeMaterial = new THREE.MeshStandardMaterial({
            color: 0x000000,
            roughness: 0.3,
            metalness: 0.8
        });
        const leftEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
        leftEye.position.set(-0.2, playerHeight * 0.87, 0.6);
        playerContainer.add(leftEye);
        
        const rightEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
        rightEye.position.set(0.2, playerHeight * 0.87, 0.6);
        playerContainer.add(rightEye);
        
        // Create glowing orb for status (more dramatic)
        const orbGeometry = new THREE.IcosahedronGeometry(0.3, 2);
        const orbMaterial = new THREE.MeshStandardMaterial({
            color: 0x00ff00,
            emissive: 0x00ff00,
            emissiveIntensity: 0.8,
            transparent: true,
            opacity: 0.9
        });
        const orb = new THREE.Mesh(orbGeometry, orbMaterial);
        orb.position.y = playerHeight * 1.2;
        orb.name = 'statusOrb';
        playerContainer.add(orb);
        
        // Add outer glow sphere
        const glowGeometry = new THREE.SphereGeometry(0.5, 12, 8);
        const glowMaterial = new THREE.MeshStandardMaterial({
            color: 0x00ff00,
            emissive: 0x00ff00,
            emissiveIntensity: 0.3,
            transparent: true,
            opacity: 0.3
        });
        const glow = new THREE.Mesh(glowGeometry, glowMaterial);
        glow.position.y = playerHeight * 1.2;
        playerContainer.add(glow);
        
        // Add point light for glow effect
        const orbLight = new THREE.PointLight(0x00ff00, 0.8, 8);
        orbLight.position.y = playerHeight * 1.2;
        orbLight.name = 'orbLight';
        orbLight.castShadow = true;
        playerContainer.add(orbLight);
        
        // Make player face center
        playerContainer.lookAt(new THREE.Vector3(0, 0, 0));
        playerContainer.rotation.y += Math.PI; // Face inward
        
        // Create nameplate with actual player thumbnail
        const thumbnailUrl = playerThumbnails[name] || `https://via.placeholder.com/60/2c3e50/ecf0f1?text=${name.charAt(0)}`;
        const nameplate = threeState.demo._createNameplate(name, thumbnailUrl, CSS2DObject);
        nameplate.position.set(0, playerHeight * 1.4, 0);
        playerContainer.add(nameplate);
        
        // Store references
        threeState.demo._playerObjects.set(name, {
            container: playerContainer,
            body: body,
            head: head,
            shoulders: shoulders,
            orb: orb,
            glow: glow,
            orbLight: orbLight,
            nameplate: nameplate,
            pedestal: pedestal,
            originalPosition: playerContainer.position.clone(),
            baseAngle: angle,
            isAlive: true
        });
        
        threeState.demo._playerGroup.add(playerContainer);
    });
    
    // Adjust camera to see the full circle
    if (threeState.demo._camera) {
        threeState.demo._camera.position.set(25, 30, 25);
        threeState.demo._controls.target.set(0, 5, 0);
        threeState.demo._controls.update();
    }
}
