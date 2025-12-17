export class SkySystem {
  constructor(scene, THREE, Sky, assetManager) {
    this.scene = scene;
    this.THREE = THREE;
    this.Sky = Sky;
    this.assetManager = assetManager;

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

    // Reusable Vectors for Boids
    this._tempVec1 = new THREE.Vector3();
    this._tempVec2 = new THREE.Vector3();
    this._tempVec3 = new THREE.Vector3();

    this.init();
  }

  // ... (init, createSkyShader, createSunMoon, createMoonMesh, createGodRays, createClouds, createBirds, updatePhase, updateSkySystem methods remain unchanged)

  update(time, phaseValue) {
      // Animate clouds
      if (this.clouds) {
          this.clouds.forEach(cloud => {
              if (cloud && cloud.userData) {
                  cloud.userData.initialAngle += cloud.userData.speed;
                  cloud.position.x = Math.cos(cloud.userData.initialAngle) * cloud.userData.radius;
                  cloud.position.z = Math.sin(cloud.userData.initialAngle) * cloud.userData.radius;
                  cloud.position.y = cloud.userData.height + Math.sin(time * 0.0005) * 2;
              }
          });
      }

      // Animate stars (Shader based)
      if (this.stars && this.starsMaterial) {
          this.starsMaterial.uniforms.time.value = time;
      }

      // Animate god rays
      if (this.godRays) {
          this.godRays.forEach((ray, index) => {
              if (ray.userData) {
                  const pulse = Math.sin(time * 0.0008 * ray.userData.speed + ray.userData.phase);
                  if (ray.material) {
                      const baseOpacity = ray.userData.originalOpacity * this.godRayIntensity;
                      ray.material.opacity = baseOpacity * (0.85 + pulse * 0.15);
                  }
                  ray.rotation.z = ray.userData.baseRotationZ + Math.sin(time * 0.0002 + index * 0.5) * 0.03;
                  ray.rotation.x = ray.userData.baseRotationX + Math.cos(time * 0.00025 + index * 0.3) * 0.02;
                  const lengthVariation = 1 + Math.sin(time * 0.0003 + ray.userData.phase) * 0.05;
                  ray.scale.y = lengthVariation;
              }
          });
      }

      if (this.moonMesh && this.moonMesh.visible) {
          this.moonMesh.rotation.y = time * 0.00002;
      }

      // Animate Birds (Boids Swarm with No-Flight Zone)
      if (this.birds && phaseValue < 0.5) {
          const birds = this.birds;
          const perceptionRadius = 40;
          const separationRadius = 15;
          const maxSpeed = 0.15;
          const maxForce = 0.001;
          
          birds.forEach(bird => {
              if(!bird.visible) return;

              // Boids Rules
              // Reset temp vectors
              const alignment = this._tempVec1.set(0, 0, 0);
              const cohesion = this._tempVec2.set(0, 0, 0);
              const separation = this._tempVec3.set(0, 0, 0);
              let count = 0;

              birds.forEach(other => {
                  if (bird !== other) {
                      const dist = bird.position.distanceTo(other.position);
                      if (dist < perceptionRadius) {
                          alignment.add(other.userData.velocity);
                          cohesion.add(other.position);
                          count++;
                          
                          if (dist < separationRadius) {
                              // Re-use separation vector for diff calculation temporarily is tricky without a 4th vector
                              // So we calculate components directly to save allocations
                              const diffX = bird.position.x - other.position.x;
                              const diffY = bird.position.y - other.position.y;
                              const diffZ = bird.position.z - other.position.z;
                              
                              // Divide by dist
                              separation.x += diffX / dist;
                              separation.y += diffY / dist;
                              separation.z += diffZ / dist;
                          }
                      }
                  }
              });

              if (count > 0) {
                  alignment.divideScalar(count).normalize().multiplyScalar(maxSpeed).sub(bird.userData.velocity).clampLength(0, maxForce);
                  cohesion.divideScalar(count).sub(bird.position).normalize().multiplyScalar(maxSpeed).sub(bird.userData.velocity).clampLength(0, maxForce);
                  separation.divideScalar(count).normalize().multiplyScalar(maxSpeed).sub(bird.userData.velocity).clampLength(0, maxForce * 1.5);
              }

              // Environmental Constraints
              
              // 1. No Flight Zone (Center Exclusion)
              const distCenter = Math.sqrt(bird.position.x * bird.position.x + bird.position.z * bird.position.z);
              if (distCenter < 50) {
                  // Normalize directly
                  const invLen = 1 / distCenter;
                  const pushX = bird.position.x * invLen;
                  const pushZ = bird.position.z * invLen;
                  
                  bird.userData.velocity.x += pushX * 0.05;
                  bird.userData.velocity.z += pushZ * 0.05;
              }

              // 2. Outer Boundary
              if (distCenter > 110) {
                  const invLen = 1 / distCenter;
                  const pullX = -bird.position.x * invLen;
                  const pullZ = -bird.position.z * invLen;
                  
                  bird.userData.velocity.x += pullX * 0.005;
                  bird.userData.velocity.z += pullZ * 0.005;
              }

              // 3. Height Control
              const targetH = 45;
              bird.userData.velocity.y += (targetH - bird.position.y) * 0.0005;

              // Apply Forces
              bird.userData.velocity.add(alignment.multiplyScalar(1.0));
              bird.userData.velocity.add(cohesion.multiplyScalar(0.8));
              bird.userData.velocity.add(separation.multiplyScalar(1.2));

              // Limit Speed
              bird.userData.velocity.clampLength(0.1, maxSpeed);

              // Update Position
              bird.position.add(bird.userData.velocity);

              // Orientation
              // We need a target to look at. We can reuse tempVec1
              this._tempVec1.copy(bird.position).add(bird.userData.velocity);
              bird.lookAt(this._tempVec1);

              // Flap Wings
              const flapSpeed = bird.userData.wingSpeed * (1 + bird.userData.velocity.length() * 2);
              const flap = Math.sin(time * 0.01 * flapSpeed + bird.userData.wingPhase) * 0.15;
              
              if (bird.userData.wing1 && bird.userData.wing2) {
                  // GLTF Node Animation
                  // Assuming wings extend along X and rotate around Z (or local equivalent)
                  // Adjust axis based on model orientation. Usually Z for up/down.
                  bird.userData.wing1.rotation.z = flap;
                  bird.userData.wing2.rotation.z = -flap;
              } else if (bird.geometry && bird.geometry.attributes.position) {
                  // Fallback Vertex Animation
                  const positions = bird.geometry.attributes.position.array;
                  // Simple hack: flap vertices 3-8 (wing1) and 9-14 (wing2)
                  // indices: 13 (y of wing1 tip), 22 (y of wing2 tip) based on old logic
                  if (positions.length > 22) {
                      positions[13] = flap;
                      positions[22] = flap; // In old logic, both went same direction relative to Y?
                      // Wait, old logic: positions[13] = flap; positions[22] = flap;
                      // Wing 1 tip: 1.2, 0, -0.1. Y is 0. 
                      // Wing 2 tip: -1.2, 0, -0.1.
                      // Setting Y to flap makes them go up/down.
                      bird.geometry.attributes.position.needsUpdate = true;
                  }
              }
          });
      }
  }

  init() {
    this.createSkyShader();
    this.createSunMoon();
    this.createMoonMesh();
    this.createGodRays();
    // this.createStars(); // Removed per user request
    this.createClouds();
    this.loadBirdModel();

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

  loadBirdModel() {
      const birdPath = `${import.meta.env.BASE_URL}static/werewolf/bird.glb`;
      this.assetManager.loadGLTF(birdPath).then((gltf) => {
          this.createBirds(gltf);
      }).catch(err => {
          console.error("Failed to load bird model, falling back to procedural.", err);
          this.createBirds(null); // Fallback
      });
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
    // Group to hold moon mesh and glow (handles orbit and facing)
    this.moonMesh = new this.THREE.Group();

    // Giant blood moon sphere
    const moonGeometry = new this.THREE.SphereGeometry(80, 64, 64);
    const moonMaterial = new this.THREE.MeshStandardMaterial({
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

    // Load texture async
    const moonTexturePath = `${import.meta.env.BASE_URL}static/moon_texture.jpg`;
    this.assetManager.loadTexture(moonTexturePath).then((moonTexture) => {
        if (this.moonSphere && this.moonSphere.material) {
            this.moonSphere.material.map = moonTexture;
            this.moonSphere.material.emissiveMap = moonTexture;
            this.moonSphere.material.needsUpdate = true;
        }
    });
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

  createBirds(gltf) {
    this.birds = [];
    const birdCount = 20;

    const toonGradient = this.assetManager.getToonGradientMap();
    const material = new this.THREE.MeshToonMaterial({
        color: 0x222222,
        gradientMap: toonGradient,
        side: this.THREE.DoubleSide
    });

    // Add distance scaling to the material
    material.onBeforeCompile = (shader) => {
        shader.vertexShader = `
            varying float vDistance;
        ` + shader.vertexShader;

        shader.vertexShader = shader.vertexShader.replace(
            '#include <project_vertex>',
            `
            vec4 mvPosition = modelViewMatrix * vec4( transformed, 1.0 );
            float dist = length(mvPosition.xyz);
            vDistance = dist;
            float scaleFactor = smoothstep(30.0, 60.0, dist);
            mvPosition = modelViewMatrix * vec4( transformed * scaleFactor, 1.0 );
            gl_Position = projectionMatrix * mvPosition;
            `
        );
    };

    let birdFactory;
    
    if (gltf) {
        const sourceScene = gltf.scene;
        sourceScene.traverse((child) => {
            if (child.isMesh) {
                child.material = material;
                child.castShadow = true;
                child.receiveShadow = false;
            }
        });

        birdFactory = () => {
            const model = sourceScene.clone(true);
            
            // Fix orientation: Rotate model inside a container
            // User reports beak pointing down (likely -Y axis). Rotate -90 deg on X to point it to +Z (Forward).
            model.rotation.x = -Math.PI / 2;
            
            // Fix scaling: The model might be too large/small. 
            // Previous code had `bird.scale.setScalar(0.2)` in the loop. 
            // We can apply that to the container or model.
            
            const wing1 = model.getObjectByName('wing1');
            const wing2 = model.getObjectByName('wing2');
            
            const container = new this.THREE.Group();
            container.add(model);
            
            return { mesh: container, wing1, wing2 };
        };
    } else {
        // Fallback Geometry
        const vertices = new Float32Array([
            0, 0, 0.4, 0.15, -0.05, -0.3, -0.15, -0.05, -0.3, 
            0.05, 0, 0.1, 1.2, 0, -0.1, 0.05, 0, -0.2,       
            -0.05, 0, 0.1, -1.2, 0, -0.1, -0.05, 0, -0.2     
        ]);
        const geometry = new this.THREE.BufferGeometry();
        geometry.setAttribute('position', new this.THREE.BufferAttribute(vertices, 3));
        geometry.computeVertexNormals();
        
        birdFactory = () => {
            const bird = new this.THREE.Mesh(geometry, material);
            bird.castShadow = true;
            return { mesh: bird, wing1: null, wing2: null }; 
        };
    }

    for(let i=0; i<birdCount; i++) {
        const { mesh: bird, wing1, wing2 } = birdFactory();
        
        bird.position.set(
            (Math.random() - 0.5) * 100,
            35 + Math.random() * 20, 
            (Math.random() - 0.5) * 100
        );
        
        const speed = 0.1 + Math.random() * 0.05;
        const angle = Math.random() * Math.PI * 2;
        
        bird.userData = {
            velocity: new this.THREE.Vector3(Math.cos(angle) * speed, 0, Math.sin(angle) * speed),
            wingSpeed: 0.01 + Math.random() * 0.01,
            wingPhase: Math.random() * Math.PI * 2,
            wing1: wing1,
            wing2: wing2
        };
        
        bird.rotation.y = -angle;
        // Scale the GLTF model if needed, e.g. 0.5 if it's too big
        // For now, keep 1.0 or adjust based on visual check
        bird.scale.setScalar(0.2); 

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
