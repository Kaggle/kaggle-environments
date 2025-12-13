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
    this.birds = [];
    this.sunPosition = new THREE.Vector3();

    this.init();
  }

  init() {
    this.createSkyShader();
    this.createSunMoon();
    this.createMoonMesh();
    this.createGodRays();
    // this.createStars(); // Removed per user request
    this.createClouds();
    this.createBirds();

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
    skyUniforms['turbidity'].value = 2.0;
    skyUniforms['rayleigh'].value = 0.2;
    skyUniforms['mieCoefficient'].value = 0.001;
    skyUniforms['mieDirectionalG'].value = 0.9;
  }

  createSunMoon() {
    // Sun Light
    this.sunLight = new this.THREE.DirectionalLight(0xffffff, 0.8);
    this.sunLight.castShadow = true;
    this.sunLight.shadow.mapSize.width = 1024;
    this.sunLight.shadow.mapSize.height = 1024;
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
    const moonGeometry = new this.THREE.SphereGeometry(80, 64, 64);
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

  createClouds() {
    this.clouds = [];
    const cloudCount = 5;

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
      const puffs = 5 + Math.floor(Math.random() * 8);

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
      const height = -50 + Math.random() * 20; // Below town level

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

  createBirds() {
    this.birds = [];
    const birdCount = 15; // Reduced density

    // Simple Body + Wings Geometry (9 vertices)
    const vertices = new Float32Array([
        // Body (Triangle)
        0, 0, 0.4,      // 0: Nose
        0.15, -0.05, -0.3, // 1: Tail Right
        -0.15, -0.05, -0.3,// 2: Tail Left

        // Right Wing
        0.05, 0, 0.1,   // 3: Shoulder
        1.2, 0, -0.1,   // 4: Wing Tip (ANIMATION TARGET)
        0.05, 0, -0.2,  // 5: Back joint

        // Left Wing
        -0.05, 0, 0.1,  // 6: Shoulder
        -1.2, 0, -0.1,  // 7: Wing Tip (ANIMATION TARGET)
        -0.05, 0, -0.2  // 8: Back joint
    ]);
    
    // Use Standard material for lighting interactions
    const material = new this.THREE.MeshStandardMaterial({
        color: 0x222222,
        roughness: 0.9,
        side: this.THREE.DoubleSide
    });

    for(let i=0; i<birdCount; i++) {
        const geometry = new this.THREE.BufferGeometry();
        geometry.setAttribute('position', new this.THREE.BufferAttribute(vertices.slice(), 3));
        geometry.computeVertexNormals(); // Needed for Standard material
        
        const bird = new this.THREE.Mesh(geometry, material);
        bird.castShadow = true;
        bird.receiveShadow = false;
        
        bird.position.set(
            (Math.random() - 0.5) * 100,
            35 + Math.random() * 20, // Reduced height
            (Math.random() - 0.5) * 100
        );
        
        // Boids parameters
        const speed = 0.2 + Math.random() * 0.1;
        const angle = Math.random() * Math.PI * 2;
        
        bird.userData = {
            velocity: new this.THREE.Vector3(Math.cos(angle) * speed, 0, Math.sin(angle) * speed),
            wingSpeed: 0.03 + Math.random() * 0.03, // Reduced flapping speed
            wingPhase: Math.random() * Math.PI * 2,
        };
        
        bird.rotation.y = -angle;
        
        // Scale down to avoid looking like fake triangles
        bird.scale.setScalar(1.0);

        this.scene.add(bird);
        this.birds.push(bird);
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

    // 1. Determine Day/Night (Strict Threshold)
    const isNight = phase >= 0.5;

    const sunDistance = 400;
    const moonDistance = 900;

    // 2. Define Restricted Arcs for Day Sun
    const minElevation = Math.PI / 12; // 15 degrees
    const maxElevation = Math.PI - Math.PI / 12; // 165 degrees
    const angularRange = maxElevation - minElevation;

    let sunAngle, moonAngle;
    let moonX, moonY, moonZ; // Declare variables here

    if (!isNight) {
        // --- DAY PHASE (0.0 to 0.5) ---
        const dayProgress = Math.min(1.0, Math.max(0.0, phase / 0.5));
        sunAngle = minElevation + dayProgress * angularRange;
        
        // Moon is hidden (opposite side)
        moonAngle = sunAngle + Math.PI;
        
        // Standard Day Moon Position (Opposite Sun)
        const moonTilt = 0.2; 
        const mX = moonDistance * Math.cos(moonAngle);
        const mYBase = moonDistance * Math.sin(moonAngle);
        moonX = mX;
        moonY = mYBase * Math.cos(moonTilt);
        moonZ = mYBase * Math.sin(moonTilt);
    } else {
        // --- NIGHT PHASE (0.5 to 1.0) ---
        const nightProgress = Math.min(1.0, Math.max(0.0, (phase - 0.5) / 0.5));
        
        // Horizontal Travel (< 120 degrees)
        const sweepRange = 100 * (Math.PI / 180);
        const azimuth = (0.5 - nightProgress) * sweepRange; // +50 to -50
        
        // Base Position on horizontal circle (XZ plane)
        const arcRadius = moonDistance;
        let mX = arcRadius * Math.sin(azimuth);
        let mZ = -arcRadius * Math.cos(azimuth);
        
        // "Sit 5 degrees above horizon"
        const baseElevationAngle = 5 * (Math.PI / 180);
        let mY = arcRadius * Math.sin(baseElevationAngle);
        
        // "5 degree tilt"
        const tiltAngle = 5 * (Math.PI / 180);
        
        // Apply tilt rotation to X/Y
        const tiltedX = mX * Math.cos(tiltAngle) - mY * Math.sin(tiltAngle);
        const tiltedY = mX * Math.sin(tiltAngle) + mY * Math.cos(tiltAngle);
        
        // Assign final positions
        moonX = tiltedX;
        moonY = tiltedY;
        moonZ = mZ;

        // Hide Sun
        sunAngle = -Math.PI / 2; // Below horizon
    }

    // --- Calculate Positions ---
    // Sun
    const sunX = sunDistance * Math.cos(sunAngle);
    const sunY = sunDistance * Math.sin(sunAngle);
    const sunZ = 0;
    
    // Update Sun Object
    this.sunPosition.set(sunX, sunY, sunZ);
    this.sky.material.uniforms['sunPosition'].value.copy(this.sunPosition);
    this.sunLight.position.copy(this.sunPosition);
    
    // Sun Intensity logic
    this.sunLight.intensity = (!isNight) ? 0.8 : 0.0;
    this.sunLight.visible = !isNight;
    
    // Sun Color logic
    if (!isNight) {
        // ... (Keep existing day color logic)
        const distFromZenith = Math.abs(sunAngle - Math.PI/2);
        const maxDist = Math.PI/2 - minElevation;
        const normalizedHeight = 1.0 - (distFromZenith / maxDist);
        if (normalizedHeight < 0.3) {
            this.sunLight.color.setHex(0xffaa66);
        } else {
            this.sunLight.color.setHex(0xffffff);
        }
    }

    // Update Moon Object
    if (this.moonMesh) {
      this.moonMesh.position.set(moonX, moonY, moonZ);
      this.moonMesh.lookAt(0, 0, 0);
      this.moonMesh.visible = isNight;
      
      // Update Moon Rotation/Color
      if (this.moonSphere && this.moonSphere.material) {
         // ... (Keep existing color logic)
         if (isNight) {
             this.moonSphere.material.color.setHex(0xff6633);
             this.moonSphere.material.emissive.setHex(0xdd5522);
             this.moonSphere.material.emissiveIntensity = 0.9;
             this.moonLight.color.setHex(0xff6633);
         } else {
             this.moonSphere.material.color.setHex(0xffffff);
             this.moonSphere.material.emissive.setHex(0x222222);
             this.moonSphere.material.emissiveIntensity = 0.2;
             this.moonLight.color.setHex(0xaaccff);
         }
         this.moonSphere.rotation.y = phase * Math.PI * 4;
      }
    }

    this.moonLight.position.set(moonX, moonY, moonZ);
    this.moonLight.visible = isNight;
    this.moonLight.intensity = isNight ? 0.4 : 0.0;


    // --- Atmosphere & Post-Processing ---
    const skyUniforms = this.sky.material.uniforms;

    if (!isNight) {
      // DAY ATMOSPHERE
      const dayProgress = phase / 0.5;
      // Simple logic: Middle of day (0.5 progress) = clearest
      // Edges = more turbidity
      const distFromNoon = Math.abs(dayProgress - 0.5) * 2; // 0 (noon) to 1 (edges)
      
      // Interpolate
      skyUniforms['turbidity'].value = 4.7 + distFromNoon * 3.0;
      skyUniforms['rayleigh'].value = 0.2 + distFromNoon * 0.8;
      skyUniforms['mieCoefficient'].value = 0.001 + distFromNoon * 0.005;
      skyUniforms['mieDirectionalG'].value = 0.9;
    } else {
      // NIGHT ATMOSPHERE
      // Dark, high turbidity to simulate night sky
      skyUniforms['turbidity'].value = 10;
      skyUniforms['rayleigh'].value = 2.0;
      skyUniforms['mieCoefficient'].value = 0.005;
      skyUniforms['mieDirectionalG'].value = 0.8;
    }

    // Stars
    if (this.starsMaterial) {
      if (isNight) {
          this.starsMaterial.uniforms.phase.value = 1.0; 
      } else {
        this.starsMaterial.uniforms.phase.value = 0;
      }
    }

    // Clouds
    if (this.clouds) {
      this.clouds.forEach((cloud) => {
        if (!cloud || !cloud.material) return;
        cloud.material.opacity = 0.3;
      });
    }

    // Birds (Day only)
    if (this.birds) {
        this.birds.forEach(bird => {
            bird.visible = !isNight;
        });
    }

    // God Rays
    if (this.godRayGroup && this.godRays) {
      let godRayVisible = false;
      let godRayPosition = null;
      let godRayColor = 0xffeeaa;
      let intensityMult = 1.0;

      if (!isNight) {
        // Sun God Rays
        godRayVisible = true;
        godRayPosition = this.sunPosition.clone();
        
        // Color based on height
        if (this.sunLight.color.r > 0.9 && this.sunLight.color.g < 0.8) {
            // Orange sun
            godRayColor = 0xffaa66;
            intensityMult = 1.0;
        } else {
            // White sun
            godRayColor = 0xffffff;
            intensityMult = 0.4;
        }
      } 
      // Disable moon rays for now or re-enable if desired
      // else { ... }

      this.godRayGroup.visible = godRayVisible && this.godRayIntensity > 0;
      if (godRayVisible && godRayPosition) {
        this.godRayGroup.position.copy(godRayPosition);
        this.godRayGroup.lookAt(0, 0, 0);
        this.godRays.forEach((ray) => {
          if (ray.material) {
            ray.material.color.setHex(godRayColor);
            const finalOpacity = ray.userData.originalOpacity * intensityMult * this.godRayIntensity;
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
