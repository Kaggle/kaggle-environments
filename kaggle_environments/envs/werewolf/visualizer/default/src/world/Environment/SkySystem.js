export class SkySystem {
  constructor(scene, THREE, Sky) {
    this.scene = scene;
    this.THREE = THREE;
    this.Sky = Sky;

    this.sky = null;
    this.sunLight = null;
    this.moonLight = null;
    this.moonMesh = null;
    this.moonSphere = null; // Actual rotating sphere inside the moonMesh group
    this.moonGlow = null;
    this.godRayGroup = null;
    this.godRays = [];
    this.godRayIntensity = 1.0;
    this.stars = null;
    this.starsMaterial = null;
    this.clouds = [];
    this.sunPosition = new THREE.Vector3();

    this.init();
  }

  init() {
    this.createSkyShader();
    this.createSunMoon();
    this.createMoonMesh();
    this.createGodRays();
    this.createStars();
    this.createClouds();

    // Initial update
    // Initialize with day settings - set initial sun position for visibility
    const phi = ((90 - 45) * Math.PI) / 180; // 45 degrees elevation
    const theta = (180 * Math.PI) / 180; // 180 degrees azimuth
    const sunX = Math.sin(phi) * Math.cos(theta);
    const sunY = Math.cos(phi);
    const sunZ = Math.sin(phi) * Math.sin(theta);
    this.sunPosition = new this.THREE.Vector3(sunX, sunY, sunZ);
    this.sky.material.uniforms['sunPosition'].value.copy(this.sunPosition);

    this.updateSkySystem(0.25);
  }

  createSkyShader() {
    this.sky = new this.Sky();
    this.sky.scale.setScalar(450000);
    this.scene.add(this.sky);

    const skyUniforms = this.sky.material.uniforms;
    skyUniforms['turbidity'].value = 4.7;
    skyUniforms['rayleigh'].value = 0.2;
    skyUniforms['mieCoefficient'].value = 0.001;
    skyUniforms['mieDirectionalG'].value = 0.9;
  }

  createSunMoon() {
    // Sun Light
    this.sunLight = new this.THREE.DirectionalLight(0xffffff, 0.8);
    this.sunLight.castShadow = true;
    this.sunLight.shadow.mapSize.width = 2048;
    this.sunLight.shadow.mapSize.height = 2048;
    this.sunLight.shadow.camera.near = 0.5;
    this.sunLight.shadow.camera.far = 500;
    this.sunLight.shadow.camera.left = -75;
    this.sunLight.shadow.camera.right = 75;
    this.sunLight.shadow.camera.top = 75;
    this.sunLight.shadow.camera.bottom = -75;
    this.sunLight.shadow.bias = -0.001;
    this.sunLight.shadow.normalBias = 0.02;
    this.scene.add(this.sunLight);
    this.scene.add(this.sunLight.target);

    // Moon Light
    this.moonLight = new this.THREE.DirectionalLight(0xff6633, 0.6); // Blood orange, slightly brighter
    this.moonLight.castShadow = true;
    this.moonLight.shadow.mapSize.width = 1024;
    this.moonLight.shadow.mapSize.height = 1024;
    this.moonLight.shadow.camera.near = 0.5;
    this.moonLight.shadow.camera.far = 500;
    this.moonLight.shadow.camera.left = -100;
    this.moonLight.shadow.camera.right = 100;
    this.moonLight.shadow.camera.top = 100;
    this.moonLight.shadow.camera.bottom = -100;
    this.moonLight.visible = false;
    this.scene.add(this.moonLight);
    this.scene.add(this.moonLight.target);
  }

  createMoonMesh() {
    const textureLoader = new this.THREE.TextureLoader();
    const moonTexture = textureLoader.load(`${import.meta.env.BASE_URL}static/moon_texture.jpg`);

    // Group to hold moon mesh and glow (handles orbit and facing)
    this.moonMesh = new this.THREE.Group();

    // Giant blood moon sphere
    const moonGeometry = new this.THREE.SphereGeometry(25, 64, 64);
    const moonMaterial = new this.THREE.MeshStandardMaterial({
      map: moonTexture,
      emissiveMap: moonTexture,
      color: 0xff6633,
      emissive: 0xdd5522,
      emissiveIntensity: 0.9,
      roughness: 1.0,
      metalness: 0.0,
    });

    this.moonSphere = new this.THREE.Mesh(moonGeometry, moonMaterial);
    this.moonSphere.castShadow = false;
    this.moonSphere.receiveShadow = false;
    this.moonMesh.add(this.moonSphere);

    // Red surrounding glow
    const moonGlowGeometry = new this.THREE.SphereGeometry(30, 32, 32);
    const moonGlowMaterial = new this.THREE.MeshBasicMaterial({
      color: 0xdd5522,
      transparent: true,
      opacity: 0.1,
      side: this.THREE.BackSide,
    });
    this.moonGlow = new this.THREE.Mesh(moonGlowGeometry, moonGlowMaterial);
    this.moonMesh.add(this.moonGlow);

    this.moonMesh.visible = false;
    this.scene.add(this.moonMesh);
  }

  createGodRays() {
    this.godRayGroup = new this.THREE.Group();
    this.godRayGroup.name = 'godRays';
    this.godRays = [];
    const godRayCount = 12;

    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');
    const gradient = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
    gradient.addColorStop(0, 'rgba(255, 255, 255, 1.0)');
    gradient.addColorStop(1, 'rgba(255, 255, 255, 0.0)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 128, 128);
    const beamTexture = new this.THREE.CanvasTexture(canvas);

    for (let i = 0; i < godRayCount; i++) {
      const rayLength = 350 + Math.random() * 100;
      const rayWidth = 4 + Math.random() * 3;
      const beamGeometry = new this.THREE.CylinderGeometry(rayWidth, rayWidth * 0.5, rayLength, 16, 1, true);

      const beamMaterial = new this.THREE.MeshBasicMaterial({
        map: beamTexture,
        color: 0xffffff,
        transparent: true,
        opacity: 0.1 + Math.random() * 0.05,
        blending: this.THREE.AdditiveBlending,
        side: this.THREE.DoubleSide,
        depthWrite: false,
      });

      const beam = new this.THREE.Mesh(beamGeometry, beamMaterial);
      const angle = Math.random() * Math.PI * 2;
      const spread = Math.random() * 0.1;

      beam.position.set(Math.sin(angle) * spread * 150, 0, Math.cos(angle) * spread * 150);
      beam.lookAt(new this.THREE.Vector3(Math.sin(angle), spread, Math.cos(angle)));

      beam.userData = {
        originalOpacity: beamMaterial.opacity,
        phase: Math.random() * Math.PI * 2,
        speed: 0.3 + Math.random() * 0.4,
        baseRotationZ: beam.rotation.z,
        baseRotationX: beam.rotation.x,
      };

      this.godRays.push(beam);
      this.godRayGroup.add(beam);
    }

    this.godRayIntensity = 1.0;
    this.godRayGroup.visible = true;
    this.scene.add(this.godRayGroup);
  }

  createStars() {
    const starsGeometry = new this.THREE.BufferGeometry();
    const starCount = 2000;
    const positions = new Float32Array(starCount * 3);
    const colors = new Float32Array(starCount * 3);
    const sizes = new Float32Array(starCount);

    for (let i = 0; i < starCount; i++) {
      const i3 = i * 3;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const radius = 400 + Math.random() * 100;

      positions[i3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i3 + 2] = radius * Math.cos(phi);

      const starColor = new this.THREE.Color();
      const colorChoice = Math.random();
      if (colorChoice < 0.3) {
        starColor.setHSL(0.6, 0.1, 0.9);
      } else if (colorChoice < 0.6) {
        starColor.setHSL(0.1, 0.1, 0.95);
      } else {
        starColor.setHSL(0, 0, 1);
      }
      colors[i3] = starColor.r;
      colors[i3 + 1] = starColor.g;
      colors[i3 + 2] = starColor.b;

      sizes[i] = Math.random() * 2 + 0.5;
    }

    starsGeometry.setAttribute('position', new this.THREE.BufferAttribute(positions, 3));
    starsGeometry.setAttribute('color', new this.THREE.BufferAttribute(colors, 3));
    starsGeometry.setAttribute('size', new this.THREE.BufferAttribute(sizes, 1));

    const starsMaterial = new this.THREE.ShaderMaterial({
      uniforms: {
        phase: { value: 0.0 },
      },
      vertexShader: `
                attribute float size;
                attribute vec3 color;
                varying vec3 vColor;
                uniform float phase;
                void main() {
                    vColor = color;
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    gl_PointSize = size * (300.0 / -mvPosition.z) * phase;
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
      fragmentShader: `
                varying vec3 vColor;
                uniform float phase;
                void main() {
                    float dist = distance(gl_PointCoord, vec2(0.5));
                    if (dist > 0.5) discard;
                    float alpha = (1.0 - dist * 2.0) * phase * 0.8;
                    gl_FragColor = vec4(vColor, alpha);
                }
            `,
      transparent: true,
      blending: this.THREE.AdditiveBlending,
      depthWrite: false,
    });

    this.stars = new this.THREE.Points(starsGeometry, starsMaterial);
    this.starsMaterial = starsMaterial;
    this.scene.add(this.stars);
  }

  createClouds() {
    this.clouds = [];
    const cloudCount = 50;

    // 1. Generate Soft Puff Texture
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');
    const grad = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
    grad.addColorStop(0, 'rgba(255, 255, 255, 0.8)');
    grad.addColorStop(0.4, 'rgba(255, 255, 255, 0.3)');
    grad.addColorStop(1, 'rgba(255, 255, 255, 0.0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 128, 128);
    const texture = new this.THREE.CanvasTexture(canvas);

    const baseMaterial = new this.THREE.SpriteMaterial({
      map: texture,
      color: 0xffffff,
      transparent: true,
      opacity: 0.6,
      depthWrite: false, // Prevents z-fighting artifacts between puffs
    });

    for (let i = 0; i < cloudCount; i++) {
      const cloudGroup = new this.THREE.Group();
      const puffs = 10 + Math.floor(Math.random() * 15);

      for (let j = 0; j < puffs; j++) {
        const sprite = new this.THREE.Sprite(baseMaterial);
        
        // Randomize scale for each puff
        const scale = 20 + Math.random() * 25;
        sprite.scale.set(scale, scale, 1);

        // Position puff within the cloud volume (flattened ellipsoid)
        const spreadX = 35;
        const spreadY = 10;
        const spreadZ = 35;
        
        sprite.position.set(
          (Math.random() - 0.5) * spreadX,
          (Math.random() - 0.5) * spreadY,
          (Math.random() - 0.5) * spreadZ
        );

        cloudGroup.add(sprite);
      }

      // Position the Cloud Group in the world
      const angle = Math.random() * Math.PI * 2;
      const orbitRadius = 60 + Math.random() * 200; // Surrounding the rock
      const height = -40 + Math.random() * 20; // Below town level

      cloudGroup.position.set(Math.cos(angle) * orbitRadius, height, Math.sin(angle) * orbitRadius);

      // Animation Data (Compatible with World.js loop)
      cloudGroup.userData = {
        initialAngle: angle,
        radius: orbitRadius,
        height: height,
        speed: 0.0001 + Math.random() * 0.0001
      };

      this.scene.add(cloudGroup);
      this.clouds.push(cloudGroup);
    }
  }

  updatePhase(phase, currentEventIndex) {
    if (!window.werewolfGamePlayer || !window.werewolfGamePlayer.allEvents) return;

    const normalizedPhase = (phase || 'DAY').toUpperCase();
    const allEvents = window.werewolfGamePlayer.allEvents;
    if (allEvents.length === 0) return;

    const safeCurrentIndex = Math.min(Math.max(0, currentEventIndex || 0), allEvents.length - 1);

    // 1. Find the start of the current phase (search backwards)
    let phaseStartIndex = 0;
    for (let i = safeCurrentIndex; i >= 0; i--) {
      const event = allEvents[i];
      if (event.event_name === 'day_start' || event.event_name === 'night_start') {
        phaseStartIndex = i;
        break;
      }
    }

    // 2. Find the end of the current phase (search forwards)
    let phaseEndIndex = allEvents.length - 1;
    for (let i = safeCurrentIndex + 1; i < allEvents.length; i++) {
      const event = allEvents[i];
      if (event.event_name === 'day_start' || event.event_name === 'night_start') {
        phaseEndIndex = i;
        break;
      }
    }

    // 3. Calculate phase progress (0.0 to 1.0)
    const totalPhaseEvents = phaseEndIndex - phaseStartIndex;
    const currentPhaseEvents = safeCurrentIndex - phaseStartIndex;
    let phaseProgress = 0;
    if (totalPhaseEvents > 0) {
      phaseProgress = Math.max(0, Math.min(1, currentPhaseEvents / totalPhaseEvents));
    }

    // 4. Map progress to the 0.0-1.0 time-of-day scale
    let targetPhase;
    if (normalizedPhase === 'NIGHT') {
      targetPhase = 0.5 + phaseProgress * 0.5;
    } else {
      targetPhase = phaseProgress * 0.5;
    }

    this.updateSkySystem(targetPhase);
    return targetPhase;
  }

  updateSkySystem(phase) {
    if (!this.sky || !this.sunLight || !this.moonLight) return;

    const sunDistance = 400;
    const moonDistance = 400;
    const cycleAngle = phase * 2 * Math.PI; // 0 to 2PI

    // --- Sun Trajectory (Overhead) ---
    // Rise East (+X), Set West (-X)
    const sunX = sunDistance * Math.cos(cycleAngle);
    const sunY = sunDistance * Math.sin(cycleAngle);
    const sunZ = 0;

    // --- Moon Trajectory (Tilted Arc) ---
    // Opposite to Sun
    const moonAngle = cycleAngle + Math.PI;
    const moonTilt = 0.5; // ~28 degrees tilt
    const moonX = moonDistance * Math.cos(moonAngle);
    const moonY = moonDistance * Math.sin(moonAngle) * Math.cos(moonTilt);
    const moonZ = moonDistance * Math.sin(moonAngle) * Math.sin(moonTilt);

    // Update Sun
    this.sunPosition.set(sunX, sunY, sunZ);
    this.sky.material.uniforms['sunPosition'].value.copy(this.sunPosition);

    this.sunLight.position.copy(this.sunPosition);
    this.sunLight.target.position.set(0, 0, 0);
    const sunIntensity = Math.max(0, Math.sin(cycleAngle));
    this.sunLight.intensity = sunIntensity * 1.2;
    this.sunLight.visible = sunY > 0;

    const sunColorTemp = sunY > 0 && sunY < 100 ? new this.THREE.Color(0xffaa66) : new this.THREE.Color(0xffffff);
    this.sunLight.color = sunColorTemp;

    // Update Moon
    if (this.moonMesh) {
      this.moonMesh.position.set(moonX, moonY, moonZ);
      this.moonMesh.lookAt(0, 0, 0);
      this.moonMesh.visible = moonY > 0;

      const moonScale = 1 + Math.max(0, (1 - Math.abs(moonY) / 100) * 0.5);
      this.moonMesh.scale.setScalar(moonScale);

      if (this.moonSphere) {
        this.moonSphere.rotation.y = phase * Math.PI * 4;
      }

      if (this.moonGlow && this.moonGlow.material) {
        this.moonGlow.material.opacity = moonY > 0 ? 0.3 : 0;
      }
    }

    this.moonLight.position.set(moonX, moonY, moonZ);
    this.moonLight.target.position.set(0, 0, 0);
    this.moonLight.visible = moonY > 0;
    this.moonLight.intensity = moonY > 0 ? 0.8 : 0;

    // --- Atmosphere & Post-Processing (Phase Dependent) ---
    const skyUniforms = this.sky.material.uniforms;
    let dayProgress = 0;

    if (phase <= 0.5) {
      // DAY
      dayProgress = phase * 2;
      if (dayProgress < 0.1 || dayProgress > 0.9) {
        const transitionFactor = dayProgress < 0.1 ? dayProgress * 10 : (1 - dayProgress) * 10;
        const turbidity_noon = 4.7,
          rayleigh_noon = 0.2,
          mieCoeff_noon = 0.001,
          mieG_noon = 0.9;
        const turbidity_dusk = 8.0,
          rayleigh_dusk = 1.0,
          mieCoeff_dusk = 0.01,
          mieG_dusk = 0.95;
        skyUniforms['turbidity'].value = turbidity_dusk + (turbidity_noon - turbidity_dusk) * transitionFactor;
        skyUniforms['rayleigh'].value = rayleigh_dusk + (rayleigh_noon - rayleigh_dusk) * transitionFactor;
        skyUniforms['mieCoefficient'].value = mieCoeff_dusk + (mieCoeff_noon - mieCoeff_dusk) * transitionFactor;
        skyUniforms['mieDirectionalG'].value = mieG_dusk + (mieG_noon - mieG_dusk) * transitionFactor;
      } else {
        skyUniforms['turbidity'].value = 4.7;
        skyUniforms['rayleigh'].value = 0.2;
        skyUniforms['mieCoefficient'].value = 0.001;
        skyUniforms['mieDirectionalG'].value = 0.9;
      }
    } else {
      // NIGHT
      const nightProgress = (phase - 0.5) * 2;
      if (nightProgress < 0.1 || nightProgress > 0.9) {
        const transitionFactor = nightProgress < 0.1 ? 1 - nightProgress * 10 : nightProgress * 10;
        skyUniforms['turbidity'].value = 0.6 + (10 - 0.6) * transitionFactor;
        skyUniforms['rayleigh'].value = 1.9 - (1.9 - 0.1) * transitionFactor;
        skyUniforms['mieCoefficient'].value = 0.01 - (0.01 - 0.005) * transitionFactor;
        skyUniforms['mieDirectionalG'].value = 1.0 - (1.0 - 0.7) * transitionFactor;
      } else {
        skyUniforms['turbidity'].value = 10;
        skyUniforms['rayleigh'].value = 2.0;
        skyUniforms['mieCoefficient'].value = 0.005;
        skyUniforms['mieDirectionalG'].value = 0.8;
      }
    }

    if (this.starsMaterial) {
      if (phase > 0.5) {
        const nightProgress = (phase - 0.5) * 2;
        if (nightProgress < 0.1) {
          this.starsMaterial.uniforms.phase.value = nightProgress * 10;
        } else if (nightProgress > 0.9) {
          this.starsMaterial.uniforms.phase.value = (1 - nightProgress) * 10;
        } else {
          this.starsMaterial.uniforms.phase.value = 1;
        }
      } else {
        this.starsMaterial.uniforms.phase.value = 0;
      }
    }

    if (this.clouds) {
      this.clouds.forEach((cloud) => {
        if (!cloud || !cloud.material) return;
        cloud.material.opacity = 0.3;
      });
    }

    if (this.godRayGroup && this.godRays) {
      let godRayVisible = false;
      let godRayPosition = null;
      let godRayIntensity = 0;
      let godRayColor = 0xffeeaa;

      if (phase <= 0.5 && sunY > 0) {
        if (sunY < 100) {
          godRayIntensity = 1.0;
          godRayColor = 0xffaa66;
        } else if (sunY < 200) {
          godRayIntensity = 0.6;
          godRayColor = 0xffddaa;
        } else {
          godRayIntensity = 0.3;
          godRayColor = 0xffffff;
        }
        godRayVisible = true;
        godRayPosition = this.sunPosition.clone();
      } else if (phase > 0.5 && moonY > 0) {
        // Optional: Moon god rays
        // godRayIntensity = 0.8;
        // godRayColor = 0xff3333;
        // godRayVisible = true;
        // godRayPosition = this.moonMesh.position.clone();
        godRayVisible = false;
      }

      this.godRayGroup.visible = godRayVisible && this.godRayIntensity > 0;
      if (godRayVisible && godRayPosition) {
        this.godRayGroup.position.copy(godRayPosition);
        this.godRayGroup.lookAt(0, 0, 0);
        this.godRays.forEach((ray) => {
          if (ray.material) {
            ray.material.color.setHex(godRayColor);
            const finalOpacity = ray.userData.originalOpacity * godRayIntensity * this.godRayIntensity;
            ray.material.opacity = finalOpacity;
          }
        });
      }
    }
  }

  setGodRayIntensity(intensity) {
    this.godRayIntensity = Math.max(0, Math.min(2, intensity));
  }
}
