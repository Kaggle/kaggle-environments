export class CharacterManager {
  constructor(scene, THREE, SkeletonUtils, CSS2DObject, assetManager) {
    this.scene = scene;
    this.THREE = THREE;
    this.skeletonUtils = SkeletonUtils;
    this.CSS2DObject = CSS2DObject;
    this.assetManager = assetManager;

    this.playerObjects = new Map();
    this.playerGroup = new this.THREE.Group();
    this.playerGroup.name = 'playerGroup';
    this.scene.add(this.playerGroup);

    this.modelCache = new Map();
    this.animationCache = new Map();
    this.deathAnimationCompleted = new Map();

    this.roleToDirectory = {
      'Werewolf': 'werewolf',
      'Doctor': 'doctor',
      'Seer': 'seer',
      'Villager': 'villager',
      'Unknown': 'villager',
    };

    this.speakingAnimations = [];
  }

  loadCharacterModel(role) {
    const normalizedRole = this.roleToDirectory[role] || this.roleToDirectory['Villager'] || 'villager';

    if (this.modelCache.has(normalizedRole)) {
      return this.modelCache.get(normalizedRole);
    }

    const modelPath = `${import.meta.env.BASE_URL}static/werewolf/models/${normalizedRole}/${normalizedRole}.fbx`;

    const modelPromise = this.assetManager.loadFBX(modelPath)
        .then((fbx) => {
          fbx.scale.setScalar(0.05);

          const animations = {};
          if (fbx.animations && fbx.animations.length > 0) {
            fbx.animations.forEach((clip) => {
              if (clip.name.startsWith('Armature|')) return;

              let animName = clip.name;
              if (animName.toLowerCase().includes('idle') || animName.toLowerCase().includes('standing')) {
                animations['Idle'] = clip;
                clip.name = 'Idle';
              } else if (animName.toLowerCase().includes('talk')) {
                animations['Talking'] = clip;
                clip.name = 'Talking';
              } else if (animName.toLowerCase().includes('point')) {
                animations['Pointing'] = clip;
                clip.name = 'Pointing';
              } else if (animName.toLowerCase().includes('victory') || animName.toLowerCase().includes('win')) {
                animations['Victory'] = clip;
                clip.name = 'Victory';
              } else if (animName.toLowerCase().includes('defeat') || animName.toLowerCase().includes('lose')) {
                animations['Defeated'] = clip;
                clip.name = 'Defeated';
              } else if (animName.toLowerCase().includes('dying') || animName.toLowerCase().includes('death')) {
                animations['Dying'] = clip;
                clip.name = 'Dying';
              } else {
                animations[animName] = clip;
              }
            });
          }

          this.animationCache.set(normalizedRole, Promise.resolve(animations));
          return fbx;
        })
        .catch((error) => {
            throw new Error(
              `Failed to load merged model for role '${role}' (normalized: '${normalizedRole}'): ${error.message || error}`
            );
        });

    this.modelCache.set(normalizedRole, modelPromise);
    return modelPromise;
  }

  loadCharacterAnimations(role) {
    const normalizedRole = this.roleToDirectory[role] || this.roleToDirectory['Villager'] || 'villager';
    if (this.animationCache.has(normalizedRole)) {
      return this.animationCache.get(normalizedRole);
    }
    return Promise.resolve({});
  }

  async initializePlayers(gameState, playerNames, playerThumbnails, uiManager) {
    if (this.playerObjects.size > 0) {
      // console.warn('3D players already initialized');
      return;
    }

    while (this.playerGroup.children.length > 0) {
      this.playerGroup.remove(this.playerGroup.children[0]);
    }
    this.playerObjects.clear();

    const numPlayers = playerNames.length;
    const radius = 18;
    const playerHeight = 4;

    // --- Optimization: Reuse Geometries and Base Materials ---
    const orbGeometry = new this.THREE.IcosahedronGeometry(0.25, 2);
    const baseOrbMaterial = new this.THREE.MeshStandardMaterial({
      color: 0x00aa88,
      emissive: 0x00aa88,
      emissiveIntensity: 1,
      transparent: true,
      opacity: 1,
      depthTest: false,
    });

    const glowGeometry = new this.THREE.SphereGeometry(0.4, 12, 8);
    const baseGlowMaterial = new this.THREE.MeshStandardMaterial({
      color: 0x00aa88,
      emissive: 0x00aa88,
      emissiveIntensity: 0.15,
      transparent: true,
      opacity: 0.2,
      depthTest: false,
    });

    const fallbackGeometry = new this.THREE.BoxGeometry(1.5, 3, 1.5);
    const baseFallbackMaterial = new this.THREE.MeshStandardMaterial({
      roughness: 0.5,
      metalness: 0.3,
      emissiveIntensity: 0.2,
    });
    // ---------------------------------------------------------

    const playerLoadPromises = playerNames.map(async (name, i) => {
      const role = gameState.players[i].role || 'Villager';
      try {
        const fbxModel = await this.loadCharacterModel(role);
        const animations = await this.loadCharacterAnimations(role);
        return { name, i, role, fbxModel, animations, success: true };
      } catch (error) {
        console.error(`Failed to load model for ${name}:`, error);
        return { name, i, role, fbxModel: null, animations: null, success: false };
      }
    });

    const loadedPlayers = await Promise.all(playerLoadPromises);

    loadedPlayers.forEach(({ name, i, role, fbxModel, animations, success }) => {
      const displayName = gameState.players[i].display_name || '';
      const playerContainer = new this.THREE.Group();
      const angle = (i / numPlayers) * Math.PI * 2;
      const x = radius * Math.sin(angle);
      const z = radius * Math.cos(angle);
      playerContainer.position.set(x, 0, z);

      let model = null;
      let mixer = null;
      let currentAction = null;
      let modelHeight = playerHeight;

      if (success && fbxModel) {
        model = this.skeletonUtils.clone(fbxModel);
        model.position.y = 0.5;

        model.traverse((child) => {
          if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
            if (child.material) {
              const materials = Array.isArray(child.material) ? child.material : [child.material];
              materials.forEach((mat) => {
                if (mat) {
                  mat.roughness = 1.0;
                  mat.metalness = 0.0;
                  mat.envMapIntensity = 0;
                  mat.roughnessMap = null;
                  mat.metalnessMap = null;
                  mat.envMap = null;
                  if (mat.color) mat.color.multiplyScalar(1.5);
                  mat.transparent = false;
                  mat.opacity = 1.0;
                  mat.alphaTest = 0;
                  mat.needsUpdate = true;
                }
              });
            }
          }
        });
        playerContainer.add(model);
        mixer = new this.THREE.AnimationMixer(model);
        if (animations && animations['Idle']) {
          currentAction = mixer.clipAction(animations['Idle']);
          currentAction.play();
        }
        const box = new this.THREE.Box3().setFromObject(model);
        const size = box.getSize(new this.THREE.Vector3());
        modelHeight = size.y;
      } else {
        const fallbackColor =
          role === 'Werewolf' ? 0x880000 : role === 'Doctor' ? 0x008800 : role === 'Seer' ? 0x4b0082 : 0x4466ff;
        
        const fallbackMaterial = baseFallbackMaterial.clone();
        fallbackMaterial.color.setHex(fallbackColor);
        fallbackMaterial.emissive.setHex(fallbackColor);

        const fallback = new this.THREE.Mesh(fallbackGeometry, fallbackMaterial);
        fallback.position.y = 2;
        fallback.castShadow = true;
        fallback.receiveShadow = true;
        playerContainer.add(fallback);
        model = fallback;
        modelHeight = 3;
      }

      // Reuse Geometry, Clone Material
      const orb = new this.THREE.Mesh(orbGeometry, baseOrbMaterial.clone());
      orb.position.y = modelHeight + 0.8;
      orb.name = 'statusOrb';
      playerContainer.add(orb);

      const glow = new this.THREE.Mesh(glowGeometry, baseGlowMaterial.clone());
      glow.position.y = modelHeight + 0.8;
      playerContainer.add(glow);

      const orbLight = new this.THREE.PointLight(0x00aa88, 0.4, 6);
      orbLight.position.y = modelHeight + 0.8;
      orbLight.name = 'orbLight';
      orbLight.castShadow = true;
      playerContainer.add(orbLight);

      const angleToCenter = Math.atan2(playerContainer.position.x, playerContainer.position.z);
      playerContainer.rotation.y = angleToCenter + Math.PI;

      const thumbnailUrl =
        playerThumbnails[name] || `https://via.placeholder.com/60/2c3e50/ecf0f1?text=${name.charAt(0)}`;

      const focusCallback = (playerName) => {
          if (window.werewolfThreeJs && window.werewolfThreeJs.demo) {
              const leftPanel = document.querySelector('.left-panel');
              const eventPanel = document.querySelector('.event-panel');
              const leftW = (leftPanel && leftPanel.offsetParent !== null) ? leftPanel.offsetWidth : 0;
              const rightW = (eventPanel && eventPanel.offsetParent !== null) ? eventPanel.offsetWidth : 0;
              
              window.werewolfThreeJs.demo.focusOnPlayer(playerName, leftW, rightW);
          }
      };

      const playerUI = uiManager.createPlayerUI(name, displayName, thumbnailUrl, focusCallback);
      playerUI.position.set(0, modelHeight + 2.0, 0);
      playerContainer.add(playerUI);

      this.playerObjects.set(name, {
        container: playerContainer,
        model: model,
        playerUI: playerUI,
        mixer: mixer,
        animations: animations,
        currentAction: currentAction,
        orb: orb,
        glow: glow,
        orbLight: orbLight,
        originalPosition: playerContainer.position.clone(),
        baseAngle: angle,
        isAlive: true,
      });

      this.playerGroup.add(playerContainer);
    });
  }

  // ... (rest of methods: updatePlayerActive, updatePlayerStatus, playAnimation, trigger..., update, reset)
  // I will just append the methods I defined in the previous step

  updatePlayerActive(playerName) {
    const player = this.playerObjects.get(playerName);
    if (!player) return;
    const { orb, orbLight, glow, container } = player;

    orb.material.emissiveIntensity = 1;
    orbLight.intensity = 1;
    glow.material.emissiveIntensity = 0.5;
    container.scale.setScalar(1.1);
  }

  updatePlayerStatus(playerName, player_info, status, threatLevel = 0, justDied = false) {
    const player = this.playerObjects.get(playerName);
    if (!player) return;

    const { orb, orbLight, glow, container, mixer, animations, currentAction } = player;

    if (player_info && player_info.role !== 'Unknown' && player.playerUI) {
      const roleElement = player.playerUI.element.querySelector('.player-role-3d');
      if (roleElement) {
        let roleDisplay = player_info.role;
        let roleColor = '#00b894';
        if (player_info.role === 'Werewolf') {
          roleDisplay = `\u{1F43A} ${player_info.role}`;
          roleColor = '#e17055';
        } else if (player_info.role === 'Doctor') {
          roleDisplay = `\u{1FA7A} ${player_info.role}`;
          roleColor = '#6c5ce7';
        } else if (player_info.role === 'Seer') {
          roleDisplay = `\u{1F52E} ${player_info.role}`;
          roleColor = '#fd79a8';
        }
        roleElement.textContent = `${roleDisplay}`;
        roleElement.style.color = roleColor;
        roleElement.style.fontWeight = 'bold';
      }
    }

    // Reset to default
    orb.material.color.setHex(0x00aa88);
    orb.material.emissive.setHex(0x00aa88);
    orb.material.emissiveIntensity = 1.0;
    orb.material.opacity = 0.85;
    orb.visible = true;
    orbLight.color.setHex(0x00aa88);
    orbLight.intensity = 0.8;
    orbLight.visible = true;

    glow.material.color.setHex(0x00aa88);
    glow.material.emissive.setHex(0x00aa88);
    glow.material.emissiveIntensity = 0.15;
    glow.visible = true;
    
    container.scale.setScalar(1.0);
    container.position.y = 0;
    container.rotation.x = 0;

    if (player.nameplate && player.nameplate.element) {
      player.nameplate.element.style.transition = 'opacity 0.5s ease-in';
      player.nameplate.element.style.opacity = '1.0';
    }
    player.isAlive = true;

    switch (status) {
      case 'dead':
        orb.visible = false;
        orbLight.visible = false;
        glow.visible = false;
        if (player.nameplate && player.nameplate.element) {
          player.nameplate.element.style.transition = 'opacity 2s ease-out';
          player.nameplate.element.style.opacity = '0.7';
        }
        player.isAlive = false;

        mixer.stopAllAction();

        const dyingAnimation = animations ? animations['Dying'] : null;
        if (dyingAnimation) {
          const action = mixer.clipAction(dyingAnimation);
          action.setLoop(this.THREE.LoopOnce);
          action.clampWhenFinished = true;

          if (justDied) {
            // console.debug(`[DEATH TRIGGER] Playing full death animation for ${playerName}.`);
            action.reset().play();
            this.deathAnimationCompleted.set(playerName, true);
          } else {
            action.play();
            action.time = action.getClip().duration;
            this.deathAnimationCompleted.set(playerName, true);
          }
        } else {
          container.rotation.x = 0.2;
        }
        break;
      case 'werewolf':
        glow.material.color.setHex(0xaa4444);
        glow.material.emissive.setHex(0xaa4444);
        glow.material.emissiveIntensity = 0.2;
        glow.visible = true;
        break;
      case 'doctor':
        glow.material.color.setHex(0x44aa44);
        glow.material.emissive.setHex(0x44aa44);
        glow.material.emissiveIntensity = 0.2;
        glow.visible = true;
        break;
      case 'seer':
        glow.material.color.setHex(0x6644aa);
        glow.material.emissive.setHex(0x6644aa);
        glow.material.emissiveIntensity = 0.2;
        glow.visible = true;
        break;
    }

    if (threatLevel >= 1.0) {
      orb.material.color.setHex(0xaa4444);
      orb.material.emissive.setHex(0xaa4444);
      orb.material.emissiveIntensity = 1.2;
      orb.material.opacity = 0.8;
      orbLight.color.setHex(0xaa4444);
      orbLight.intensity = 0.6;
      glow.material.color.setHex(0xaa4444);
      glow.material.emissive.setHex(0xaa4444);
      glow.material.emissiveIntensity = 0.2;
    } else if (threatLevel >= 0.5) {
      orb.material.color.setHex(0xaaaa44);
      orb.material.emissive.setHex(0xaaaa44);
      orb.material.emissiveIntensity = 1.1;
      orb.material.opacity = 0.75;
      orbLight.color.setHex(0xaaaa44);
      orbLight.intensity = 0.5;
      glow.material.color.setHex(0xaaaa44);
      glow.material.emissive.setHex(0xaaaa44);
      glow.material.emissiveIntensity = 0.18;
    }

    if (player.isAlive && mixer && animations && animations['Idle']) {
      // Force reset to Idle if currently Talking (to stop loop) or if no action playing
      const isTalking = currentAction && currentAction.getClip().name === 'Talking';
      if (!currentAction || isTalking) {
        this.playAnimation(playerName, 'Idle');
      }
    }
  }

  playAnimation(playerName, animationName, options = {}) {
    const player = this.playerObjects.get(playerName);
    if (!player || !player.model) return null;

    if (this.deathAnimationCompleted.has(playerName)) {
      if (!['Victory', 'Defeated'].includes(animationName)) {
        return null;
      }
    }

    if (!player.isAlive && animationName !== 'Dying') {
      return null;
    }

    const animations = player.animations;
    if (!animations || !animations[animationName]) return null;

    const mixer = player.mixer;
    if (!mixer) return null;

    // Check if already playing
    const targetAction = mixer.clipAction(animations[animationName]);
    if (player.currentAction === targetAction && targetAction.isRunning()) {
      return targetAction;
    }

    if (player.currentAction) {
      player.currentAction.fadeOut(options.fadeOutDuration || 0.2);
    }

    const action = targetAction;
    action.reset();
    action.setLoop(this.THREE.LoopRepeat);

    if (options.loop !== undefined) action.setLoop(options.loop);
    if (options.clampWhenFinished !== undefined) action.clampWhenFinished = options.clampWhenFinished;

    action.fadeIn(options.fadeInDuration || 0.2);
    action.play();

    player.currentAction = action;
    return action;
  }

  createSoundWave() {
    const waveGeometry = new this.THREE.RingGeometry(0.5, 0.7, 32);
    const waveMaterial = new this.THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.8,
      side: this.THREE.DoubleSide,
    });
    const wave = new this.THREE.Mesh(waveGeometry, waveMaterial);
    wave.rotation.x = -Math.PI / 2;
    wave.position.y = 0.25;
    return wave;
  }

  triggerSpeakingAnimation(playerName) {
    const player = this.playerObjects.get(playerName);
    if (!player || !player.isAlive) return;

    this.playAnimation(playerName, 'Talking', { fadeInDuration: 0.2, fadeOutDuration: 0.2 });

    const wave = this.createSoundWave();
    player.container.add(wave);

    this.speakingAnimations.push({
      mesh: wave,
      startTime: performance.now(),
      duration: 1800,
    });
  }

  triggerPointingAnimation(playerName, duration = 1200) {
    const player = this.playerObjects.get(playerName);
    if (!player || !player.isAlive) return;

    this.playAnimation(playerName, 'Pointing', { fadeInDuration: 0.2, fadeOutDuration: 0.2 });

    setTimeout(() => {
      this.playAnimation(playerName, 'Idle', { fadeInDuration: 0.2 });
    }, duration);
  }

  triggerVictoryAnimation(playerName) {
    this.playAnimation(playerName, 'Victory');
  }

  triggerDefeatedAnimation(playerName) {
    this.playAnimation(playerName, 'Defeated');
  }

  update(delta, time, phaseValue) {
    this.playerObjects.forEach((player) => {
      if (player.mixer) player.mixer.update(delta);

      if (player.isAlive) {
        const floatOffset = Math.sin(time * 0.001 + player.baseAngle) * 0.2;
        const bobOffset = Math.cos(time * 0.0015 + player.baseAngle * 2) * 0.05;
        player.container.position.y = floatOffset + bobOffset;

        if (player.orb) {
          player.orb.rotation.y = time * 0.003;
          player.orb.rotation.x = Math.sin(time * 0.002) * 0.15;
          player.orb.rotation.z = Math.cos(time * 0.0025) * 0.1;
        }

        if (player.glow && player.glow.visible) {
          player.glow.rotation.y = -time * 0.002;
          const glowScale = 1 + Math.sin(time * 0.004 + player.baseAngle) * 0.15;
          player.glow.scale.setScalar(glowScale);
          player.glow.material.emissiveIntensity = 0.3 + Math.sin(time * 0.005 + player.baseAngle) * 0.1;
        }

        if (player.container.scale.x > 1.0) {
          const pulseScale = 1.05 + Math.sin(time * 0.008) * 0.08;
          player.container.scale.setScalar(pulseScale);
        }
      } else {
        if (player.orb) {
          player.orb.rotation.y = time * 0.0008;
        }
      }
    });

    const now = performance.now();
    this.speakingAnimations = this.speakingAnimations.filter((anim) => {
      const elapsedTime = now - anim.startTime;
      if (elapsedTime >= anim.duration) {
        if (anim.mesh.parent) anim.mesh.parent.remove(anim.mesh);
        anim.mesh.geometry.dispose();
        anim.mesh.material.dispose();
        return false;
      }
      const progress = elapsedTime / anim.duration;
      anim.mesh.scale.setScalar(1 + progress * 5);
      anim.mesh.material.opacity = 0.8 * (1 - progress);
      return true;
    });
  }

  reset() {
    this.deathAnimationCompleted.clear();
    this.playerObjects.forEach((player, playerName) => {
      if (player.playerUI && player.playerUI.element) {
        player.playerUI.element.classList.remove('chat-active');
      }
      player.isAlive = true;

      if (player.mixer) {
        player.mixer.stopAllAction();
        let newAction = null;
        if (player.animations && player.animations['Idle']) {
          newAction = player.mixer.clipAction(player.animations['Idle']);
        } else if (player.animations && Object.keys(player.animations).length > 0) {
          const first = Object.keys(player.animations)[0];
          newAction = player.mixer.clipAction(player.animations[first]);
        }
        if (newAction) {
          newAction.play();
          player.currentAction = newAction;
        } else {
          player.currentAction = null;
        }
      }

      if (player.orb) player.orb.visible = true;
      if (player.orbLight) player.orbLight.visible = true;
      if (player.glow) player.glow.visible = true;
      if (player.nameplate && player.nameplate.element) {
        player.nameplate.element.style.opacity = '1.0';
      }
      player.container.scale.setScalar(1.0);
      player.container.rotation.x = 0;
    });
  }
}
