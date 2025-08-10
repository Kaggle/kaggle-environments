import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.118/build/three.module.js';

import {OrbitControls} from 'https://cdn.jsdelivr.net/npm/three@0.118/examples/jsm/controls/OrbitControls.js';
import {FBXLoader} from "https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/loaders/FBXLoader.js";
import { SkeletonUtils } from 'https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/utils/SkeletonUtils.js';
import { CSS2DRenderer, CSS2DObject } from 'https://cdn.jsdelivr.net/npm/three@0.118.1/examples/jsm/renderers/CSS2DRenderer.js';


class BasicWorldDemo {
  constructor() {
    this._Initialize();
  }

  _Initialize() {
    // WebGL Renderer
    this._threejs = new THREE.WebGLRenderer({
      antialias: true,
    });
    this._threejs.shadowMap.enabled = true;
    this._threejs.shadowMap.type = THREE.PCFSoftShadowMap;
    this._threejs.setPixelRatio(window.devicePixelRatio);
    this._threejs.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(this._threejs.domElement);

    // CSS2D Renderer
    this._labelRenderer = new CSS2DRenderer();
    this._labelRenderer.setSize(window.innerWidth, window.innerHeight);
    this._labelRenderer.domElement.style.position = 'absolute';
    this._labelRenderer.domElement.style.top = '0px';
    this._labelRenderer.domElement.style.pointerEvents = 'none'; // Allow clicks to pass through
    document.body.appendChild(this._labelRenderer.domElement);

    window.addEventListener('resize', () => {
      this._OnWindowResize();
    }, false);

    const fov = 60;
    const aspect = window.innerWidth / window.innerHeight;
    const near = 1.0;
    const far = 100000.0;
    this._camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
    this._camera.position.set(0, 0, 50);

    this._scene = new THREE.Scene();
    this._scene.background = new THREE.Color(0x7393B3);

    const hemisphereLight = new THREE.HemisphereLight(0xffffbb, 0x080820, 1.5);
    this._scene.add(hemisphereLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 2.0);
    dirLight.position.set(20, 50, 10);
    dirLight.castShadow = true;
    this._scene.add(dirLight);

    this._controls = new OrbitControls(this._camera, this._threejs.domElement);
    this._controls.target.set(0, 0, 0);
    this._controls.update();

    this._LoadModels();
    this._RAF();
  }

  _LoadModels() {
    const fbxLoader = new FBXLoader();
    fbxLoader.load('assets/Idle_stickman.fbx', (fbx) => {
        const stickmanModel = this._NormalizeModel(fbx);
        const stickmanGroup = new THREE.Group();
        const idleClip = fbx.animations[0];

        const names = ["Alice", "Bob", "Charlie", "Dave", "Eve", "Frank", "Grace", "Heidi"];
        
        const numStickmen = 8;
        const radius = 15;
        const sectorAngleDegrees = 60;
        const sectorAngleRadians = sectorAngleDegrees * (Math.PI / 180);
        const stickmanHeight = 4;

        // --- CHANGE START ---
        // Create a bounding box from the base (normalized) model to find its dimensions in local space.
        const modelBox = new THREE.Box3().setFromObject(stickmanModel);

        // Calculate the Y position for the top of the scaled stickman's head.
        // This is the max Y of the local bounding box, multiplied by the scaling factor.
        const topOfHeadY = modelBox.max.y * stickmanHeight;

        // Add a small offset to place the nameplate slightly above the head.
        const nameplateOffset = 1;
        // --- CHANGE END ---

        const startAngle = -sectorAngleRadians / 2;
        const angleIncrement = numStickmen > 1 ? sectorAngleRadians / (numStickmen - 1) : 0;

        for (let i = 0; i < numStickmen; i++) {
          const stickman = SkeletonUtils.clone(stickmanModel);
          const angle = startAngle + i * angleIncrement;

          const x = radius * Math.sin(angle);
          const z = radius * Math.cos(angle);
          const y = stickmanHeight / 2;
          stickman.position.set(x, y, z);

          stickman.scale.multiplyScalar(stickmanHeight);
          stickman.lookAt(new THREE.Vector3(0, y, 0));

          const nameplate = this._createNameplate(names[i]);

          // --- CHANGE START ---
          // Position the nameplate's 3D anchor above the top of the stickman's bounding box.
          nameplate.position.set(0, topOfHeadY + nameplateOffset, 0);
          // --- CHANGE END ---
          stickman.add(nameplate);

          stickmanGroup.add(stickman);

          if (idleClip) {
            const mixer = new THREE.AnimationMixer(stickman);
            const action = mixer.clipAction(idleClip);
            action.time = Math.random() * idleClip.duration;
            action.play();
            mixer.update(0);
          }
        }
        
        this._scene.add(stickmanGroup);
        this._FrameGroup(stickmanGroup);
      },
      undefined,
      (error) => {
        console.error("Error loading FBX model:", error);
      }
    );
  }

  _createNameplate(name) {
    const container = document.createElement('div');
    container.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    container.style.padding = '5px 10px';
    container.style.borderRadius = '5px';
    container.style.color = 'white';
    container.style.fontWeight = 'bold';
    container.style.fontSize = '18px';
    container.textContent = name;
    
    // This CSS transform shifts the element up by 100% of its own height,
    // aligning its bottom edge with the 3D anchor point.
    container.style.transform = 'translateY(-100%)';

    const label = new CSS2DObject(container);
    return label;
  }

  _FrameGroup(group) {
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

  _NormalizeModel(model) {
    // 1. Calculate the model's bounding box and center.
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());

    // 2. Instead of translating the geometry, reposition the model inside a wrapper.
    // We move the model by the negative of its center, which aligns its center
    // with the origin of the wrapper group.
    model.position.copy(center).negate();

    // 3. Create a wrapper group. This group will now act as the main object.
    const wrapper = new THREE.Group();
    wrapper.add(model);

    // 4. Scale the wrapper. This scales the model and its skeleton together,
    // preserving the animation.
    const scale = 1.0 / size.y;
    wrapper.scale.set(scale, scale, scale);

    // 5. Return the wrapper. This is now our correctly centered and scaled object.
    return wrapper;
  }

  _OnWindowResize() {
    this._camera.aspect = window.innerWidth / window.innerHeight;
    this._camera.updateProjectionMatrix();
    this._threejs.setSize(window.innerWidth, window.innerHeight);
    this._labelRenderer.setSize(window.innerWidth, window.innerHeight);
  }

  _RAF() {
    requestAnimationFrame(() => {
      this._threejs.render(this._scene, this._camera);
      this._labelRenderer.render(this._scene, this._camera);
      this._RAF();
    });
  }
}


let _APP = null;

window.addEventListener('DOMContentLoaded', () => {
  _APP = new BasicWorldDemo();
});