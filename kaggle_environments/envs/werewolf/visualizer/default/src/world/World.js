import { SceneManager } from './SceneManager.js';
import { SkySystem } from './Environment/SkySystem.js';
import { LightingManager } from './Environment/LightingManager.js';
import { TerrainManager } from './Environment/TerrainManager.js';
import { PropsManager } from './Environment/PropsManager.js';
import { CharacterManager } from './Characters/CharacterManager.js';
import { ParticleSystem } from './Effects/ParticleSystem.js';
import { PostProcessing } from './Effects/PostProcessing.js';
import { UIManager } from './UI/UIManager.js';
import { VoteVisuals } from './Visuals/VoteVisuals.js';

export class World {
  constructor(options, modules) {
    this.options = options;
    this.modules = modules;
    this.THREE = window.THREE; // Assumed loaded

    // Initialize Managers
    this.sceneManager = new SceneManager(options, this.THREE, modules.OrbitControls, modules.CSS2DRenderer);
    const { scene, camera, renderer, labelRenderer } = this.sceneManager;

    this.skySystem = new SkySystem(scene, this.THREE, modules.Sky);
    this.lightingManager = new LightingManager(scene, this.THREE, modules.RGBELoader, renderer);
    this.terrainManager = new TerrainManager(scene, this.THREE, modules.FBXLoader, modules.GLTFLoader);
    this.propsManager = new PropsManager(scene, this.THREE, modules.VolumetricFire, camera);
    this.characterManager = new CharacterManager(scene, this.THREE, modules.FBXLoader, modules.SkeletonUtils, modules.CSS2DObject);
    // this.particleSystem = new ParticleSystem(scene, this.THREE);
    
    this.postProcessing = new PostProcessing(
        scene, camera, renderer, options.width, options.height, this.THREE,
        modules.EffectComposer, modules.RenderPass, modules.UnrealBloomPass, modules.ShaderPass, modules.FilmPass
    );

    this.uiManager = new UIManager(modules.CSS2DObject, options.width, options.height, camera, options.parent);
    
    // Groups for visuals
    this.votingArcsGroup = new this.THREE.Group();
    this.votingArcsGroup.name = 'votingArcs';
    scene.add(this.votingArcsGroup);

    this.targetRingsGroup = new this.THREE.Group();
    this.targetRingsGroup.name = 'targetRings';
    scene.add(this.targetRingsGroup);

    this.voteVisuals = new VoteVisuals(scene, this.THREE, this.votingArcsGroup, this.targetRingsGroup);

    this.phaseTransition = null;
    this.animationClock = new this.THREE.Clock();
    this.cameraAnimation = null;

    // Camera setup
    camera.position.set(25, 30, 35);
    this.sceneManager.controls.target.set(0, 8, 0);
    this.sceneManager.controls.enableDamping = true;
    this.sceneManager.controls.dampingFactor = 0.05;
    this.sceneManager.controls.minDistance = 20;
    this.sceneManager.controls.maxDistance = 80;
    this.sceneManager.controls.maxPolarAngle = Math.PI * 0.6;
    this.sceneManager.controls.update();

    this.startRenderLoop();
  }

  // Compatibility Getters for skyControlFunctions.js
  get _sky() { return this.skySystem.sky; }
  get _sunLight() { return this.skySystem.sunLight; }
  get _moonLight() { return this.skySystem.moonLight; }
  get _ambientLight() { return this.lightingManager.ambientLight; }
  get _renderer() { return this.sceneManager.renderer; }
  get _bloomPass() { return this.postProcessing.bloomPass; }
  get _clouds() { return this.skySystem.clouds; }
  get _godRayIntensity() { return this.skySystem.godRayIntensity; }
  set _godRayIntensity(v) { this.skySystem.godRayIntensity = v; }

  updatePhase(phase, currentEventIndex) {
    if (!window.werewolfGamePlayer || !window.werewolfGamePlayer.allEvents) return;

    const normalizedPhase = (phase || 'DAY').toUpperCase();
    const allEvents = window.werewolfGamePlayer.allEvents;
    if (allEvents.length === 0) return;

    const safeCurrentIndex = Math.min(Math.max(0, currentEventIndex || 0), allEvents.length - 1);

    let phaseStartIndex = 0;
    for (let i = safeCurrentIndex; i >= 0; i--) {
      const event = allEvents[i];
      if (event.event_name === 'day_start' || event.event_name === 'night_start') {
        phaseStartIndex = i;
        break;
      }
    }

    let phaseEndIndex = allEvents.length - 1;
    for (let i = safeCurrentIndex + 1; i < allEvents.length; i++) {
      const event = allEvents[i];
      if (event.event_name === 'day_start' || event.event_name === 'night_start') {
        phaseEndIndex = i;
        break;
      }
    }

    const totalPhaseEvents = phaseEndIndex - phaseStartIndex;
    const currentPhaseEvents = safeCurrentIndex - phaseStartIndex;
    let phaseProgress = 0;
    if (totalPhaseEvents > 0) {
      phaseProgress = Math.max(0, Math.min(1, currentPhaseEvents / totalPhaseEvents));
    }

    let targetPhase;
    if (normalizedPhase === 'NIGHT') {
      targetPhase = 0.5 + phaseProgress * 0.5;
    } else {
      targetPhase = phaseProgress * 0.5;
    }

    if (!this.phaseTransition) {
      this.phaseTransition = {
        current: targetPhase,
        target: targetPhase,
        speed: 0.02,
      };
      this.updateSceneForPhase(targetPhase);
    } else {
      this.phaseTransition.target = targetPhase;
    }
  }

  updateSceneForPhase(phaseValue) {
    if (this.sceneManager.renderer) {
        this.sceneManager.renderer.toneMappingExposure = 0.5 + (0.3 - phaseValue * 0.2);
    }
    
    this.skySystem.updateSkySystem(phaseValue);
    this.lightingManager.update(phaseValue);
    
    if (this.sceneManager.scene.fog) {
        if (phaseValue < 0.5) { // Strict Day Check
            this.sceneManager.scene.fog.color.setHex(0x87ceeb);
            this.sceneManager.scene.fog.density = 0.015;
        } else { // Night
            this.sceneManager.scene.fog.color.setHex(0x000000);
            this.sceneManager.scene.fog.density = 0.005;
        }
    }

    this.postProcessing.update(0, phaseValue); // Time updated in RAF
  }

  focusOnPlayer(playerName, leftPanelWidth = 0, rightPanelWidth = 0) {
    const player = this.characterManager.playerObjects.get(playerName);
    if (!player) return;

    const effectiveWidth = this.options.width - leftPanelWidth - rightPanelWidth;
    const effectiveHeight = this.options.height;

    const viewBox = new this.THREE.Box3().setFromObject(this.characterManager.playerGroup);
    const viewSize = viewBox.getSize(new this.THREE.Vector3());
    const viewCenter = viewBox.getCenter(new this.THREE.Vector3());

    const fov = this.sceneManager.camera.fov * (Math.PI / 180);
    const aspect = effectiveWidth / effectiveHeight;
    const horizontalFov = 2 * Math.atan(Math.tan(fov / 2) * aspect);

    const distV = viewSize.y / 2 / Math.tan(fov / 2);
    const distH = viewSize.x / 2 / Math.tan(horizontalFov / 2);
    let distance = Math.max(distV, distH) * 1.05;

    const playerPosition = player.container.position.clone();
    const direction = playerPosition.clone().normalize();
    const endPos = playerPosition.clone().add(direction.multiplyScalar(distance * 0.6));
    endPos.y = playerPosition.y + distance * 0.5;

    const endTarget = viewCenter;

    this.cameraAnimation = {
      startTime: performance.now(),
      duration: 1200,
      startPos: this.sceneManager.camera.position.clone(),
      endPos: endPos,
      startTarget: this.sceneManager.controls.target.clone(),
      endTarget: endTarget,
      ease: (t) => 1 - Math.pow(1 - t, 3),
    };
  }

  resetCameraView() {
      if (this.characterManager.playerGroup.children.length === 0) return;

      const box = new this.THREE.Box3().setFromObject(this.characterManager.playerGroup);
      const size = box.getSize(new this.THREE.Vector3());
      const center = box.getCenter(new this.THREE.Vector3());

      const maxDim = Math.max(size.x, size.y, size.z);
      const fov = this.sceneManager.camera.fov * (Math.PI / 180);
      let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
      cameraZ *= 1.1;

      const endPos = new this.THREE.Vector3(center.x, center.y + cameraZ / 2, center.z + cameraZ);
      const endTarget = center;

      this.cameraAnimation = {
          startTime: performance.now(),
          duration: 1200,
          startPos: this.sceneManager.camera.position.clone(),
          endPos: endPos,
          startTarget: this.sceneManager.controls.target.clone(),
          endTarget: endTarget,
          ease: (t) => 1 - Math.pow(1 - t, 3),
      };
  }

  startRenderLoop() {
    const loop = () => {
      requestAnimationFrame(loop);
      const delta = this.animationClock.getDelta();
      const time = this.animationClock.getElapsedTime() * 1000;

      if (this.phaseTransition) {
        const diff = this.phaseTransition.target - this.phaseTransition.current;
        if (Math.abs(diff) > 0.001) {
          this.phaseTransition.current += diff * this.phaseTransition.speed;
          this.updateSceneForPhase(this.phaseTransition.current);
        }
      }
      
      const phaseValue = this.phaseTransition ? this.phaseTransition.current : 0;

      // this.particleSystem.update(time, phaseValue);
      
      // Animate clouds
      if (this.skySystem.clouds) {
          this.skySystem.clouds.forEach(cloud => {
              if (cloud && cloud.userData) {
                  cloud.userData.initialAngle += cloud.userData.speed;
                  cloud.position.x = Math.cos(cloud.userData.initialAngle) * cloud.userData.radius;
                  cloud.position.z = Math.sin(cloud.userData.initialAngle) * cloud.userData.radius;
                  cloud.position.y = cloud.userData.height + Math.sin(time * 0.0005) * 2;
              }
          });
      }

      // Animate stars (Shader based)
      if (this.skySystem.stars && this.skySystem.starsMaterial) {
          this.skySystem.starsMaterial.uniforms.time.value = time;
      }

      // Animate god rays
      if (this.skySystem.godRays) {
          this.skySystem.godRays.forEach((ray, index) => {
              if (ray.userData) {
                  const pulse = Math.sin(time * 0.0008 * ray.userData.speed + ray.userData.phase);
                  if (ray.material) {
                      const baseOpacity = ray.userData.originalOpacity * this.skySystem.godRayIntensity;
                      ray.material.opacity = baseOpacity * (0.85 + pulse * 0.15);
                  }
                  ray.rotation.z = ray.userData.baseRotationZ + Math.sin(time * 0.0002 + index * 0.5) * 0.03;
                  ray.rotation.x = ray.userData.baseRotationX + Math.cos(time * 0.00025 + index * 0.3) * 0.02;
                  const lengthVariation = 1 + Math.sin(time * 0.0003 + ray.userData.phase) * 0.05;
                  ray.scale.y = lengthVariation;
              }
          });
      }

      if (this.skySystem.moonMesh && this.skySystem.moonMesh.visible) {
          this.skySystem.moonMesh.rotation.y = time * 0.00002;
      }

      // Camera Animation
      if (this.cameraAnimation) {
          const now = performance.now();
          const anim = this.cameraAnimation;
          const elapsed = now - anim.startTime;
          let progress = Math.min(elapsed / anim.duration, 1.0);
          const easedProgress = anim.ease(progress);
          this.sceneManager.camera.position.lerpVectors(anim.startPos, anim.endPos, easedProgress);
          this.sceneManager.controls.target.lerpVectors(anim.startTarget, anim.endTarget, easedProgress);
          this.sceneManager.controls.update();
          if (progress >= 1.0) this.cameraAnimation = null;
      }

      this.propsManager.update(time * 0.001);
      this.characterManager.update(delta, time, phaseValue);
      this.voteVisuals.update();
      this.postProcessing.update(time, phaseValue);

      this.postProcessing.render();
      this.sceneManager.labelRenderer.render(this.sceneManager.scene, this.sceneManager.camera);
    };
    loop();
  }

  resize(width, height) {
      this.options.width = width;
      this.options.height = height;
      this.sceneManager.resize(width, height);
      this.postProcessing.resize(width, height);
      this.uiManager.width = width;
      this.uiManager.height = height;
  }
}
