export class PropsManager {
  constructor(scene, THREE, VolumetricFire, camera) {
    this.scene = scene;
    this.THREE = THREE;
    this.VolumetricFire = VolumetricFire;
    this.camera = camera;

    this.fire = null;
    this.fireLight = null;
    this.campfireGroup = null;

    this.init();
  }

  init() {
    this.createCampfire();
    this.createMysticalCircles(15);
  }

  createCampfire() {
    console.debug('[CAMPFIRE] Creating campfire at scene center');

    if (this.VolumetricFire) {
      this.VolumetricFire.texturePath = '/static/volumetric_fire/textures/';
    }

    const campfireGroup = new this.THREE.Group();
    campfireGroup.name = 'campfire';

    const fireWidth = 5.0;
    const fireHeight = 7.0;
    const fireDepth = 5.0;
    const sliceSpacing = 0.5;

    if (this.VolumetricFire) {
      this.fire = new this.VolumetricFire(fireWidth, fireHeight, fireDepth, sliceSpacing, this.camera);
      this.fire.mesh.position.set(0, fireHeight / 2, 0);
      campfireGroup.add(this.fire.mesh);
      console.debug('[CAMPFIRE] VolumetricFire created');
    } else {
      console.warn('[CAMPFIRE] VolumetricFire not available, skipping fire effect');
    }

    const fireLight = new this.THREE.PointLight(0xff6633, 2.5, 25);
    fireLight.position.set(0, 1.5, 0);
    fireLight.castShadow = true;
    fireLight.shadow.mapSize.width = 512;
    fireLight.shadow.mapSize.height = 512;
    campfireGroup.add(fireLight);
    this.fireLight = fireLight;

    // Rock circle
    const rockCount = 8;
    const rockRadius = 2.2;
    for (let i = 0; i < rockCount; i++) {
      const angle = (i / rockCount) * Math.PI * 2;
      const x = Math.cos(angle) * rockRadius;
      const z = Math.sin(angle) * rockRadius;
      const rockSize = 0.4 + Math.random() * 0.3;
      const rockGeometry = new this.THREE.DodecahedronGeometry(rockSize, 0);
      const rockMaterial = new this.THREE.MeshStandardMaterial({
        color: 0x3a3a3a,
        roughness: 0.95,
        metalness: 0.1,
        flatShading: true,
      });
      const rock = new this.THREE.Mesh(rockGeometry, rockMaterial);
      rock.position.set(x, rockSize * 0.3, z);
      rock.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI);
      rock.castShadow = true;
      rock.receiveShadow = true;
      campfireGroup.add(rock);
    }

    // Logs
    const logCount = 6;
    const logLength = 2.0;
    const logRadius = 0.12;
    for (let i = 0; i < logCount; i++) {
      const angle = (i / logCount) * Math.PI * 2;
      const logGeometry = new this.THREE.CylinderGeometry(logRadius, logRadius * 0.9, logLength, 8);
      const logMaterial = new this.THREE.MeshStandardMaterial({
        color: 0x4a3520,
        roughness: 0.9,
        metalness: 0.0,
      });
      const log = new this.THREE.Mesh(logGeometry, logMaterial);
      const leanRadius = 0.8;
      const x = Math.cos(angle) * leanRadius;
      const z = Math.sin(angle) * leanRadius;
      log.position.set(x, logLength / 2 - 0.3, z);
      log.rotation.z = Math.PI / 6;
      log.rotation.y = angle + Math.PI / 2;
      log.castShadow = true;
      log.receiveShadow = true;
      campfireGroup.add(log);
    }

    // Kindling
    const kindlingCount = 12;
    for (let i = 0; i < kindlingCount; i++) {
      const angle = Math.random() * Math.PI * 2;
      const radius = Math.random() * 0.6;
      const kindlingGeometry = new this.THREE.CylinderGeometry(0.03, 0.025, 0.4 + Math.random() * 0.3, 6);
      const kindlingMaterial = new this.THREE.MeshStandardMaterial({
        color: 0x3a2510,
        roughness: 0.95,
        metalness: 0.0,
      });
      const kindling = new this.THREE.Mesh(kindlingGeometry, kindlingMaterial);
      kindling.position.set(Math.cos(angle) * radius, 0.2, Math.sin(angle) * radius);
      kindling.rotation.set(Math.random() * 0.5, Math.random() * Math.PI * 2, Math.random() * 0.5);
      kindling.castShadow = true;
      kindling.receiveShadow = true;
      campfireGroup.add(kindling);
    }

    campfireGroup.position.set(0, 0, 0);
    this.scene.add(campfireGroup);
    this.campfireGroup = campfireGroup;
  }

  createMysticalCircles(radius) {
    for (let i = 0; i < 3; i++) {
      const circleRadius = radius - i * 2 - 1;
      const circleGeometry = new this.THREE.RingGeometry(circleRadius - 0.1, circleRadius + 0.1, 64);
      const circleMaterial = new this.THREE.MeshStandardMaterial({
        color: new this.THREE.Color().setHSL(0.6 + i * 0.1, 0.8, 0.3 + i * 0.1),
        emissive: new this.THREE.Color().setHSL(0.6 + i * 0.1, 0.5, 0.1),
        emissiveIntensity: 0.2,
        transparent: true,
        opacity: 0.6 - i * 0.1,
        side: this.THREE.DoubleSide,
      });
      const circle = new this.THREE.Mesh(circleGeometry, circleMaterial);
      circle.rotation.x = -Math.PI / 2;
      circle.position.y = 0.01 + i * 0.001;
      this.scene.add(circle);
    }
  }

  update(time) {
    if (this.fire && this.fire.update) {
      this.fire.update(time);
    }
    if (this.fireLight) {
      this.fireLight.intensity = 4.5 + Math.sin(time * 3) * 0.5; // Adjusted timing scale assumption
    }
  }
}
